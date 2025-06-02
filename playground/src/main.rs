#![feature(generic_const_exprs)]
#![feature(allocator_api)]
#![feature(vec_push_within_capacity)]

mod messages;
mod old_tracer;
mod tracer;

use crate::tracer::{DelegationTracingData, MagicTracer, RamTracingData};
use cs::definitions::{timestamp_from_chunk_cycle_and_sequence, TimestampScalar};
use execution_utils::get_padded_binary;
use fft::GoodAllocator;
use gpu_prover::allocator::host::ConcurrentStaticHostAllocator;
use gpu_prover::prover::context::{MemPoolProverContext, ProverContext};
use itertools::Itertools;
use prover::risc_v_simulator::abstractions::non_determinism::{
    NonDeterminismCSRSource, QuasiUARTSource,
};
use prover::risc_v_simulator::cycle::{IMStandardIsaConfig, MachineConfig};
use prover::tracers::delegation::DelegationWitness;
use prover::tracers::main_cycle_optimized::CycleData;
use prover::tracers::oracles::chunk_lazy_init_and_teardown;
use prover::{ShuffleRamSetupAndTeardown, VectorMemoryImplWithRom};
use rayon::ThreadPoolBuilder;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::iter;
use std::process::exit;
use std::rc::Rc;
use std::sync::Arc;
use trace_and_split::setups::trace_len_for_machine;
use trace_and_split::{
    run_till_end_for_machine_config_without_tracing, setups, FinalRegisterValue, ENTRY_POINT,
};
use worker::Worker;

fn run_till_end_for_gpu_for_machine_config<
    ND: NonDeterminismCSRSource<VectorMemoryImplWithRom>,
    C: MachineConfig,
    A: GoodAllocator + 'static,
    const ROM_ADDRESS_SPACE_SECOND_WORD_BITS: usize,
>(
    num_cycles_upper_bound: usize,
    trace_size: usize,
    binary: &[u32],
    mut non_determinism: ND,
    delegation_factories: HashMap<u16, Box<dyn Fn() -> DelegationWitness<A>>>,
) -> (
    u32,
    Vec<CycleData<C, A>>,
    HashMap<u16, Vec<DelegationWitness<A>>>,
    Vec<FinalRegisterValue>,
    (
        usize, // number of empty ones to assume
        Vec<ShuffleRamSetupAndTeardown<A>>,
    ),
) {
    use setups::prover::risc_v_simulator::cycle::state_new::RiscV32StateForUnrolledProver;
    use setups::prover::risc_v_simulator::delegations::DelegationsCSRProcessor;

    assert!(trace_size.is_power_of_two());
    let rom_address_space_bound = 1usize << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS);

    let mut memory = VectorMemoryImplWithRom::new_for_byte_size(1 << 32, rom_address_space_bound); // use full RAM
    for (idx, insn) in binary.iter().enumerate() {
        memory.populate(ENTRY_POINT + idx as u32 * 4, *insn);
    }

    let cycles_per_chunk = trace_size - 1;
    let num_cycles_upper_bound = num_cycles_upper_bound.next_multiple_of(cycles_per_chunk);
    let num_circuits_upper_bound = num_cycles_upper_bound / cycles_per_chunk;

    let mut state = RiscV32StateForUnrolledProver::<C>::initial(ENTRY_POINT);

    let bookkeeping_aux_data = RamTracingData::<true, true>::new_for_ram_size_and_rom_bound(
        1 << 32,
        rom_address_space_bound,
    );
    let delegation_tracer = DelegationTracingData {
        delegation_witness_factories: delegation_factories,
        witnesses: HashMap::new(),
    };

    // important - in our memory implementation first access in every chunk is timestamped as (trace_size * circuit_idx) + 4,
    // so we take care of it

    let mut custom_csr_processor = DelegationsCSRProcessor;
    let mut chunks = vec![];
    let delegation_chunks = Rc::new(RefCell::new(HashMap::new()));
    let delegation_chunks_clone = delegation_chunks.clone();

    let initial_ts = timestamp_from_chunk_cycle_and_sequence(0, cycles_per_chunk, 0);
    let tracer = MagicTracer::<_, A, true, true, false, false>::new(
        initial_ts,
        bookkeeping_aux_data,
        delegation_tracer,
        cycles_per_chunk,
        Box::new(move |id, w| {
            delegation_chunks_clone
                .borrow_mut()
                .entry(id)
                .or_insert_with(|| vec![])
                .push(w);
        }),
    );

    let mut tracer = Some(tracer);
    let mut end_reached = false;
    let mut circuits_needed = 0;

    let now = std::time::Instant::now();

    for chunk_idx in 0..num_circuits_upper_bound {
        circuits_needed = chunk_idx + 1;

        let finished = state.run_cycles(
            &mut memory,
            tracer.as_mut().unwrap(),
            &mut non_determinism,
            &mut custom_csr_processor,
            cycles_per_chunk,
        );

        if finished {
            println!("Ended at address 0x{:08x}", state.pc);
            println!("Took {} circuits to finish execution", circuits_needed);
            end_reached = true;
            break;
        } else {
            let timestamp =
                timestamp_from_chunk_cycle_and_sequence(0, cycles_per_chunk, chunk_idx + 1);
            let (new_tracer, chunk) = tracer
                .take()
                .unwrap()
                .prepare_for_next_chunk_and_return_processed::<false>(timestamp);
            if let Some(chunk) = chunk {
                chunks.push(chunk);
            }
            tracer = Some(new_tracer);
        };
    }

    assert!(end_reached, "end of the execution was never reached");

    let elapsed = now.elapsed();
    let cycles_upper_bound = circuits_needed * cycles_per_chunk;
    let speed = (cycles_upper_bound as f64) / elapsed.as_secs_f64() / 1_000_000f64;
    println!(
        "Simulator running speed with witness tracing is {} MHz: ran {} cycles over {:?}",
        speed, cycles_upper_bound, elapsed
    );

    let MagicTracer {
        bookkeeping_aux_data,
        trace_chunk,
        delegation_tracer,
        ..
    } = tracer.unwrap();

    let instant = std::time::Instant::now();
    let iat_count = bookkeeping_aux_data.get_touched_ram_cells_count();
    println!("{iat_count} touched RAM cells");
    dbg!(instant.elapsed());
    let non_zero_pages = bookkeeping_aux_data
        .num_touched_ram_cells_in_page
        .iter()
        .cloned()
        .filter(|&x| x != 0)
        .count();
    dbg!(instant.elapsed());
    dbg!(non_zero_pages);
    for (i, count) in bookkeeping_aux_data
        .num_touched_ram_cells_in_page
        .iter()
        .cloned()
        .enumerate()
        .filter(|&(_, x)| x != 0)
    {
        println!("Page {i} touched in {count} addresses");
    }
    // put latest chunk manually in traced ones
    // chunks.push(trace_chunk);
    // assert_eq!(chunks.len(), circuits_needed);

    // let RamTracingData {
    //     register_last_live_timestamps,
    //     ram_words_last_live_timestamps,
    //     // access_bitmask,
    //     ..
    // } = bookkeeping_aux_data;

    // now we can co-join touched memory cells, their final values and timestamps

    // let memory_final_state = memory.get_final_ram_state();
    // let memory_state_ref = &memory_final_state;
    // // let memory_ref = &memory.get_final_ram_state();
    // let ram_words_last_live_timestamps_ref = &ram_words_last_live_timestamps;

    // // parallel collect
    // // first we will walk over access_bitmask and collect subparts
    // let mut teardown_chunks: Vec<Vec<(u32, (TimestampScalar, u32))>> =
    //     vec![vec![].clone(); worker.get_num_cores()];
    // let mut dst = &mut teardown_chunks[..];
    // worker.scope(ram_words_last_live_timestamps.len(), |scope, geometry| {
    //     for thread_idx in 0..geometry.len() {
    //         let chunk_size = geometry.get_chunk_size(thread_idx);
    //         let chunk_start = geometry.get_chunk_start_pos(thread_idx);
    //         let range = chunk_start..(chunk_start + chunk_size);
    //         let (el, rest) = dst.split_at_mut(1);
    //         dst = rest;
    //         let src = &ram_words_last_live_timestamps[range];
    //
    //         Worker::smart_spawn(scope, thread_idx == geometry.len() - 1, move |_| {
    //             let el = &mut el[0];
    //             for (idx, &timestamp) in src.iter().enumerate() {
    //                 if timestamp == 0 {
    //                     continue; // skip unused words
    //                 }
    //                 let word_idx = chunk_start + idx;
    //                 let phys_address = (word_idx as u32) << 2;
    //                 let word_value = memory_state_ref[word_idx];
    //                 el.push((phys_address, (timestamp, word_value)));
    //             }
    //         });
    //     }
    // });
    let alloc_fn = || {
        let mut vec =
            Vec::with_capacity_in(cycles_per_chunk, A::default());
        unsafe {
            vec.set_len(cycles_per_chunk);
        }
        Some(ShuffleRamSetupAndTeardown {
            lazy_init_data: vec,
        })
    };
    let instant = std::time::Instant::now();
    let memory_final_state = memory.get_final_ram_state();
    let (_, setups_and_teardowns_iter) = bookkeeping_aux_data.get_setup_and_teardown_chunks(
        &memory_final_state,
        cycles_per_chunk,
        iter::from_fn(alloc_fn),
    );
    let teardown_chunks = setups_and_teardowns_iter.collect_vec();
    println!(
        "Lazy init/teardown chunks collected in {:?}",
        instant.elapsed()
    );

    let RamTracingData {
        register_last_live_timestamps,
        ram_words_last_live_timestamps,
        // access_bitmask,
        ..
    } = bookkeeping_aux_data;
    
    let mut registers_final_states = Vec::with_capacity(32);
    for register_idx in 0..32 {
        let last_timestamp = register_last_live_timestamps[register_idx];
        let register_state = FinalRegisterValue {
            value: state.registers[register_idx],
            last_access_timestamp: last_timestamp,
        };
        registers_final_states.push(register_state);
    }
    
    let DelegationTracingData { witnesses, .. } = delegation_tracer;
    
    let mut merged_delegation_chunks = HashMap::new();
    
    for (&delegation_id, w) in delegation_chunks.borrow_mut().iter_mut() {
        let w = w.drain(..).collect_vec();
        merged_delegation_chunks.insert(delegation_id, w);
    }
    
    for (delegation_id, w) in witnesses.into_iter() {
        merged_delegation_chunks
            .entry(delegation_id)
            .or_insert_with(Vec::new)
            .push(w);
    }
    
    (
        state.pc,
        chunks,
        merged_delegation_chunks,
        registers_final_states,
        (circuits_needed - teardown_chunks.len(), teardown_chunks),
    )
}

fn run_and_split_for_gpu<
    ND: NonDeterminismCSRSource<VectorMemoryImplWithRom>,
    C: MachineConfig,
    A: GoodAllocator + 'static,
>(
    num_cycles_upper_bound: usize,
    binary: &[u32],
    non_determinism: ND,
    delegation_factories: HashMap<u16, Box<dyn Fn() -> DelegationWitness<A>>>,
    worker: &Worker,
) -> (
    Vec<CycleData<C, A>>,
    (
        usize, // number of empty ones to assume
        Vec<ShuffleRamSetupAndTeardown<A>>,
    ),
    HashMap<u16, Vec<DelegationWitness<A>>>,
    Vec<FinalRegisterValue>,
) {
    assert_eq!(
        setups::risc_v_cycles::ROM_ADDRESS_SPACE_SECOND_WORD_BITS,
        setups::reduced_risc_v_machine::ROM_ADDRESS_SPACE_SECOND_WORD_BITS
    );
    assert_eq!(
        setups::risc_v_cycles::ROM_ADDRESS_SPACE_SECOND_WORD_BITS,
        setups::final_reduced_risc_v_machine::ROM_ADDRESS_SPACE_SECOND_WORD_BITS
    );
    let domain_size = trace_len_for_machine::<C>();

    let (
        final_pc,
        main_circuit_traces,
        delegation_traces,
        register_final_values,
        lazy_init_teardown_data,
    ) = run_till_end_for_gpu_for_machine_config::<
        ND,
        C,
        A,
        { setups::risc_v_cycles::ROM_ADDRESS_SPACE_SECOND_WORD_BITS },
    >(
        num_cycles_upper_bound,
        domain_size,
        binary,
        non_determinism,
        delegation_factories,
    );

    // let instant = std::time::Instant::now();
    // let init_and_teardown_chunks = chunk_lazy_init_and_teardown::<A>(
    //     main_circuit_traces.len(),
    //     domain_size - 1,
    //     &lazy_init_teardown_data,
    //     worker,
    // );
    // println!(
    //     "Lazy init/teardown chunks created in {:?}",
    //     instant.elapsed()
    // );
    // 
    (
        main_circuit_traces,
        lazy_init_teardown_data,
        delegation_traces,
        register_final_values,
    )
}

fn main() {
    type C = IMStandardIsaConfig;
    type A = ConcurrentStaticHostAllocator;
    MemPoolProverContext::initialize_host_allocator(4, 1 << 8, 22).unwrap();
    let worker = Worker::new();
    let worker_ref = &worker;
    let mut binary = vec![];
    std::fs::File::open("examples/hashed_fibonacci/app.bin")
        .unwrap()
        .read_to_end(&mut binary)
        .unwrap();
    let binary = get_padded_binary(&binary);
    let non_determinism_source = QuasiUARTSource::new_with_reads(vec![1 << 22, 0]);
    let cycles_per_circuit = setups::num_cycles_for_machine::<C>();
    let trace_len = setups::trace_len_for_machine::<C>();
    assert_eq!(cycles_per_circuit + 1, trace_len);
    let num_instances_upper_bound = 64;
    let max_cycles_to_run = num_instances_upper_bound * cycles_per_circuit;

    let binary = Arc::new(binary);

    let delegation_factories = setups::delegation_factories_for_machine::<C, A>();
    let (
        main_circuits_witness,
        inits_and_teardowns,
        delegation_circuits_witness,
        final_register_values,
    ) = run_and_split_for_gpu::<_, C, _>(
        max_cycles_to_run,
        // 16,
        &binary,
        non_determinism_source.clone(),
        delegation_factories,
        worker_ref,
    );

    let (
        old_main_circuits_witness,
        old_inits_and_teardowns,
        old_delegation_circuits_witness,
        old_final_register_values,
    ) = old_tracer::trace_execution_for_gpu::<_, C, A>(
        num_instances_upper_bound,
        &binary,
        non_determinism_source.clone(),
        worker_ref,
    );

    // assert_eq!(main_circuits_witness.len(), old_main_circuits_witness.len());
    // for (i, (new, old)) in main_circuits_witness
    //     .into_iter()
    //     .zip(old_main_circuits_witness.into_iter())
    //     .enumerate()
    // {
    //     assert_eq!(
    //         new.per_cycle_data, old.per_cycle_data,
    //         "Main circuit witness mismatch at index {i}"
    //     );
    // }

    assert_eq!(
        inits_and_teardowns.0, old_inits_and_teardowns.0,
        "Init/teardown empty chunks count mismatch"
    );

    assert_eq!(
        inits_and_teardowns.1.len(),
        old_inits_and_teardowns.1.len(),
        "Init/teardown chunks count mismatch"
    );

    for (i, (new, old)) in inits_and_teardowns
        .1
        .into_iter()
        .zip(old_inits_and_teardowns.1.into_iter())
        .enumerate()
    {
        assert_eq!(
            new.lazy_init_data, old.lazy_init_data,
            "Init/teardown chunk mismatch at index {i}"
        );
    }

    // assert_eq!(
    //     delegation_circuits_witness.len(),
    //     old_delegation_circuits_witness.len(),
    //     "Delegation circuits witness count mismatch"
    // );
    // 
    // for (i, ((new_idx, new_witnesses), (old_idx, old_witnesses))) in delegation_circuits_witness
    //     .into_iter()
    //     .zip(old_delegation_circuits_witness.into_iter())
    //     .enumerate()
    // {
    //     assert_eq!(
    //         new_idx, old_idx,
    //         "Delegation circuit index mismatch at index {i}"
    //     );
    //     assert_eq!(
    //         new_witnesses.len(),
    //         old_witnesses.len(),
    //         "Delegation witness count mismatch at index {i}"
    //     );
    //     for (j, (new_witness, old_witness)) in new_witnesses
    //         .into_iter()
    //         .zip(old_witnesses.into_iter())
    //         .enumerate()
    //     {
    //         assert_eq!(new_witness.write_timestamp, old_witness.write_timestamp);
    //         assert_eq!(new_witness.indirect_reads, old_witness.indirect_reads);
    //         assert_eq!(new_witness.indirect_writes, old_witness.indirect_writes);
    //         assert_eq!(new_witness.register_accesses, old_witness.register_accesses);
    //     }
    // }

    assert_eq!(final_register_values, old_final_register_values);

    // let pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    // pool.scope(|s| {
    //     let instant = std::time::Instant::now();
    //     for i in 0..8 {
    //         let binary = binary.clone();
    //         let mut non_determinism_source = non_determinism_source.clone();
    //         println!("[{i}] spawning {:?}", instant.elapsed());
    //         s.spawn(move |_| {
    //             let thread_instant = std::time::Instant::now();
    //             println!("[{i}] running {:?}", instant.elapsed());
    //             let delegation_factories = setups::delegation_factories_for_machine::<C, A>();
    //             // let _ = run_without_tracing::<_, C, A>(max_cycles_to_run, &binary, &mut non_determinism_source);
    //             let (
    //                 _final_pc,
    //                 // main_circuits_witness,
    //                 // delegation_circuits_witness,
    //                 // final_register_values,
    //                 // inits_and_teardowns,
    //             ) = run_and_split_for_gpu::<_, C, _>(
    //                 max_cycles_to_run,
    //                 &binary,
    //                 &mut non_determinism_source,
    //                 delegation_factories,
    //                 worker_ref,
    //             );
    //             println!(
    //                 "[{i}] finished {:?} took {:?}",
    //                 instant.elapsed(),
    //                 thread_instant.elapsed()
    //             );
    //         });
    //         println!("[{i}] spawned {:?}", instant.elapsed());
    //     }
    // });
    // let _ = run_without_tracing::<_, C, A>(max_cycles_to_run, &binary, &mut non_determinism_source);

    // let (
    //     final_pc,
    //     main_circuits_witness,
    //     delegation_circuits_witness,
    //     final_register_values,
    //     inits_and_teardowns,
    // ) = run_and_split_for_gpu::<_, C, _>(
    //     max_cycles_to_run,
    //     &binary,
    //     &mut non_determinism_source,
    //     delegation_factories,
    //     &worker,
    // );

    // // we just need to chunk inits/teardowns
    //
    // let inits_and_teardowns = chunk_lazy_init_and_teardown(
    //     main_circuits_witness.len(),
    //     cycles_per_circuit,
    //     &inits_and_teardowns,
    //     &worker,
    // );
}
