#![feature(generic_const_exprs)]

use cs::definitions::{timestamp_from_chunk_cycle_and_sequence, TimestampScalar};
use execution_utils::get_padded_binary;
use fft::GoodAllocator;
use gpu_prover::allocator::host::ConcurrentStaticHostAllocator;
use gpu_prover::prover::context::{MemPoolProverContext, ProverContext};
use prover::risc_v_simulator::abstractions::non_determinism::{NonDeterminismCSRSource, QuasiUARTSource};
use prover::risc_v_simulator::cycle::{IMStandardIsaConfig, MachineConfig};
use prover::tracers::delegation::DelegationWitness;
use prover::tracers::main_cycle_optimized::CycleData;
use prover::VectorMemoryImplWithRom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use rayon::ThreadPoolBuilder;
use trace_and_split::setups::trace_len_for_machine;
use trace_and_split::{run_till_end_for_machine_config_without_tracing, setups, FinalRegisterValue, ENTRY_POINT};
use worker::Worker;

fn run_till_end_for_gpu_for_machine_config<
    ND: NonDeterminismCSRSource<VectorMemoryImplWithRom>,
    C: MachineConfig,
    A: GoodAllocator,
    const ROM_ADDRESS_SPACE_SECOND_WORD_BITS: usize,
>(
    num_cycles_upper_bound: usize,
    trace_size: usize,
    binary: &[u32],
    non_determinism: &mut ND,
    delegation_factories: HashMap<u16, Box<dyn Fn() -> DelegationWitness<A>>>,
    worker: &Worker,
) -> (
    u32,
    Vec<CycleData<C, A>>,
    HashMap<u16, Vec<DelegationWitness<A>>>,
    Vec<FinalRegisterValue>,
    Vec<Vec<(u32, (TimestampScalar, u32))>>, // lazy iniy/teardown data - all unique words touched, sorted ascending, but not in one vector
) {
    use prover::tracers::main_cycle_optimized::DelegationTracingData;
    use prover::tracers::main_cycle_optimized::GPUFriendlyTracer;
    use prover::tracers::main_cycle_optimized::RamTracingData;
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

    let bookkeeping_aux_data =
        RamTracingData::<true>::new_for_ram_size_and_rom_bound(1 << 32, rom_address_space_bound);
    let delegation_tracer = DelegationTracingData {
        all_per_type_logs: HashMap::new(),
        delegation_witness_factories: delegation_factories,
        current_per_type_logs: HashMap::new(),
        num_traced_registers: 0,
        mem_reads_offset: 0,
        mem_writes_offset: 0,
    };

    // important - in our memory implementation first access in every chunk is timestamped as (trace_size * circuit_idx) + 4,
    // so we take care of it

    let mut custom_csr_processor = DelegationsCSRProcessor;

    let initial_ts = timestamp_from_chunk_cycle_and_sequence(0, cycles_per_chunk, 0);
    let mut tracer = GPUFriendlyTracer::<_, _, true, true, true>::new(
        initial_ts,
        bookkeeping_aux_data,
        delegation_tracer,
        cycles_per_chunk,
        num_circuits_upper_bound,
    );

    let mut end_reached = false;
    let mut circuits_needed = 0;

    let now = std::time::Instant::now();

    for chunk_idx in 0..num_circuits_upper_bound {
        circuits_needed = chunk_idx + 1;
        if chunk_idx != 0 {
            let timestamp = timestamp_from_chunk_cycle_and_sequence(0, cycles_per_chunk, chunk_idx);
            tracer.prepare_for_next_chunk(timestamp);
        }

        let finished = state.run_cycles(
            &mut memory,
            &mut tracer,
            non_determinism,
            &mut custom_csr_processor,
            cycles_per_chunk,
        );

        if finished {
            println!("Ended at address 0x{:08x}", state.pc);
            println!("Took {} circuits to finish execution", circuits_needed);
            end_reached = true;
            break;
        };
    }

    assert!(end_reached, "end of the execution was never reached");

    let GPUFriendlyTracer {
        bookkeeping_aux_data,
        trace_chunk,
        traced_chunks,
        delegation_tracer,
        ..
    } = tracer;

    // put latest chunk manually in traced ones
    let mut traced_chunks = traced_chunks;
    traced_chunks.push(trace_chunk);
    // assert_eq!(traced_chunks.len(), circuits_needed);

    let elapsed = now.elapsed();
    let cycles_upper_bound = circuits_needed * cycles_per_chunk;
    let speed = (cycles_upper_bound as f64) / elapsed.as_secs_f64() / 1_000_000f64;
    println!(
        "Simulator running speed with witness tracing is {} MHz: ran {} cycles over {:?}",
        speed, cycles_upper_bound, elapsed
    );

    let RamTracingData {
        register_last_live_timestamps,
        ram_words_last_live_timestamps,
        access_bitmask,
        ..
    } = bookkeeping_aux_data;

    // now we can co-join touched memory cells, their final values and timestamps

    let memory_final_state = memory.get_final_ram_state();
    let memory_state_ref = &memory_final_state;
    // let memory_ref = &memory.get_final_ram_state();
    let ram_words_last_live_timestamps_ref = &ram_words_last_live_timestamps;

    // parallel collect
    // first we will walk over access_bitmask and collect subparts
    let mut chunks: Vec<Vec<(u32, (TimestampScalar, u32))>> =
        vec![vec![].clone(); worker.get_num_cores()];
    // let mut dst = &mut chunks[..];
    // worker.scope(access_bitmask.len(), |scope, geometry| {
    //     for thread_idx in 0..geometry.len() {
    //         let chunk_size = geometry.get_chunk_size(thread_idx);
    //         let chunk_start = geometry.get_chunk_start_pos(thread_idx);
    //         let range = chunk_start..(chunk_start + chunk_size);
    //         let (el, rest) = dst.split_at_mut(1);
    //         dst = rest;
    //         let src = &access_bitmask[range];
    //
    //         Worker::smart_spawn(scope, thread_idx == geometry.len() - 1, move |_| {
    //             let el = &mut el[0];
    //             for (idx, &word) in src.iter().enumerate() {
    //                 if word == 0 {
    //                     continue; // skip empty words
    //                 }
    //                 for bit_idx in 0..usize::BITS {
    //                     let word_idx =
    //                         (chunk_start + idx) * (usize::BITS as usize) + (bit_idx as usize);
    //                     let word_is_used = word & (1 << bit_idx) > 0;
    //                     if word_is_used {
    //                         let phys_address = word_idx << 2;
    //                         let word_value = memory_state_ref[word_idx];
    //                         // let word_value = unsafe { *(memory_state_ref.as_ptr().add(phys_address) as *const u32) };
    //                         let last_timestamp: TimestampScalar =
    //                             ram_words_last_live_timestamps_ref[word_idx];
    //                         el.push((phys_address as u32, (last_timestamp, word_value)));
    //                     }
    //                 }
    //             }
    //         });
    //     }
    // });

    let mut registers_final_states = Vec::with_capacity(32);
    for register_idx in 0..32 {
        let last_timestamp = register_last_live_timestamps[register_idx];
        let register_state = FinalRegisterValue {
            value: state.registers[register_idx],
            last_access_timestamp: last_timestamp,
        };
        registers_final_states.push(register_state);
    }

    let DelegationTracingData {
        all_per_type_logs,
        current_per_type_logs,
        ..
    } = delegation_tracer;

    let mut all_per_type_logs = all_per_type_logs;
    for (delegation_type, current_data) in current_per_type_logs.into_iter() {
        // We decide whether we do or not do delegation by comparing length, so we do NOT pad here.
        // GPU also benefits from little less transfer, and pads for another convantion by itself

        // let mut current_data = current_data;
        // current_data.pad();

        if current_data.is_empty() == false {
            all_per_type_logs
                .entry(delegation_type)
                .or_insert(vec![])
                .push(current_data);
        }
    }

    // assert_eq!(circuits_needed, traced_chunks.len());

    (
        state.pc,
        traced_chunks,
        all_per_type_logs,
        registers_final_states,
        chunks,
    )
}


fn run_and_split_for_gpu<
    ND: NonDeterminismCSRSource<VectorMemoryImplWithRom>,
    C: MachineConfig,
    A: GoodAllocator,
>(
    num_cycles_upper_bound: usize,
    binary: &[u32],
    non_determinism: &mut ND,
    delegation_factories: HashMap<u16, Box<dyn Fn() -> DelegationWitness<A>>>,
    worker: &Worker,
) -> (
    u32,
    Vec<CycleData<C, A>>,
    HashMap<u16, Vec<DelegationWitness<A>>>,
    Vec<FinalRegisterValue>,
    Vec<Vec<(u32, (TimestampScalar, u32))>>,
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
        worker,
    );

    (
        final_pc,
        main_circuit_traces,
        delegation_traces,
        register_final_values,
        lazy_init_teardown_data,
    )
}

pub fn run_without_tracing<
    ND: NonDeterminismCSRSource<VectorMemoryImplWithRom>,
    C: MachineConfig,
    A: GoodAllocator,
>(
    num_cycles_upper_bound: usize,
    binary: &[u32],
    non_determinism: &mut ND,
) -> (u32, [u32; 32]) {
    assert_eq!(
        setups::risc_v_cycles::ROM_ADDRESS_SPACE_SECOND_WORD_BITS,
        setups::reduced_risc_v_machine::ROM_ADDRESS_SPACE_SECOND_WORD_BITS
    );
    assert_eq!(
        setups::risc_v_cycles::ROM_ADDRESS_SPACE_SECOND_WORD_BITS,
        setups::final_reduced_risc_v_machine::ROM_ADDRESS_SPACE_SECOND_WORD_BITS
    );
    let domain_size = trace_len_for_machine::<C>();

    run_till_end_for_machine_config_without_tracing::<
        ND,
        C,
        A,
        { setups::risc_v_cycles::ROM_ADDRESS_SPACE_SECOND_WORD_BITS },
    >(
        num_cycles_upper_bound,
        domain_size,
        binary,
        non_determinism,
    )
}

fn main() {
    type C = IMStandardIsaConfig;
    type A = ConcurrentStaticHostAllocator;
    MemPoolProverContext::initialize_host_allocator(16, 1 << 8, 22).unwrap();
    let worker = Worker::new();
    let worker_ref = &worker;
    let mut binary = vec![];
    std::fs::File::open("examples/hashed_fibonacci/app.bin")
        .unwrap()
        .read_to_end(&mut binary)
        .unwrap();
    let binary = get_padded_binary(&binary);
    let mut non_determinism_source = QuasiUARTSource::new_with_reads(vec![1 << 16, 1 << 15]);
    let cycles_per_circuit = setups::num_cycles_for_machine::<C>();
    let trace_len = setups::trace_len_for_machine::<C>();
    assert_eq!(cycles_per_circuit + 1, trace_len);
    let max_cycles_to_run = 10 * cycles_per_circuit;

    let cycles_per_circuit = setups::num_cycles_for_machine::<C>();
    let max_cycles_to_run = max_cycles_to_run * cycles_per_circuit;

    // let delegation_factories =
    //     setups::delegation_factories_for_machine::<C, A>();

    let binary = Arc::new(binary);
    let pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    pool.scope(|s| {
        let instant = std::time::Instant::now();
        for i in 0..8 {
            let binary = binary.clone();
            let mut non_determinism_source = non_determinism_source.clone();
            println!("[{i}] spawning {:?}", instant.elapsed());
            s.spawn(move |_| {
                let thread_instant = std::time::Instant::now();
                println!("[{i}] running {:?}", instant.elapsed());
                let delegation_factories =
                    setups::delegation_factories_for_machine::<C, A>();
                // let _ = run_without_tracing::<_, C, A>(max_cycles_to_run, &binary, &mut non_determinism_source);
                let (
                    final_pc,
                    main_circuits_witness,
                    delegation_circuits_witness,
                    final_register_values,
                    inits_and_teardowns,
                ) = run_and_split_for_gpu::<_, C, _>(
                    max_cycles_to_run,
                    &binary,
                    &mut non_determinism_source,
                    delegation_factories,
                    worker_ref,
                );
                dbg!(main_circuits_witness.len());
                dbg!(delegation_circuits_witness.len());
                println!("[{i}] finished {:?} took {:?}", instant.elapsed(), thread_instant.elapsed());
            });
            println!("[{i}] spawned {:?}", instant.elapsed());
        }
    });
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
