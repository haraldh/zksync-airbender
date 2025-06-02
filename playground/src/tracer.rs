// - rs1 is register and read only
// - rs2 is register or RAM read
// - rd is register or RAM write

use cs::definitions::{TimestampData, TimestampScalar, TIMESTAMP_STEP};
use fft::GoodAllocator;
use itertools::{enumerate, Itertools};
use prover::definitions::LazyInitAndTeardown;
use prover::tracers::delegation::DelegationWitness;
use prover::tracers::main_cycle_optimized::{
    CycleData, RegIndexOrMemWordIndex, SingleCycleTracingData, EMPTY_SINGLE_CYCLE_TRACING_DATA,
};
use prover::ShuffleRamSetupAndTeardown;
use risc_v_simulator::abstractions::tracer::{
    RegisterOrIndirectReadData, RegisterOrIndirectReadWriteData, Tracer,
};
use risc_v_simulator::cycle::state::RiscV32State;
use risc_v_simulator::cycle::state_new::RiscV32StateForUnrolledProver;
use risc_v_simulator::cycle::{IMStandardIsaConfig, MachineConfig};
use std::alloc::Global;
use std::collections::HashMap;
use std::time::Instant;
use worker::Worker;
// NOTE: this tracer ALLOWS for delegations to initialize memory, so we should use enough cycles
// to eventually perform all the inits

const PAGE_WORDS_LOG_SIZE: usize = 10; // 4 KiB page size, 4 bytes per word
// const PAGE_WORDS_SIZE: usize = 1 << PAGE_WORDS_LOG_SIZE;

#[derive(Clone, Debug)]
pub struct RamTracingData<const TRACE_TOUCHED_RAM: bool, const TRACE_TIMESTAMPS: bool> {
    pub register_last_live_timestamps: [TimestampScalar; 32],
    pub ram_words_last_live_timestamps: Vec<TimestampScalar>,
    // pub access_bitmask: BitVec,
    // pub num_touched_ram_cells: usize,
    pub num_touched_ram_cells_in_page: Vec<u32>,
    pub num_touched_ram_cells_total: u32,
    pub rom_bound: usize,
}

impl<const TRACE_TOUCHED_RAM: bool, const TRACE_TIMESTAMPS: bool>
    RamTracingData<TRACE_TOUCHED_RAM, TRACE_TIMESTAMPS>
{
    pub fn new_for_ram_size_and_rom_bound(ram_size: usize, rom_bound: usize) -> Self {
        if TRACE_TOUCHED_RAM {
            assert!(
                TRACE_TIMESTAMPS,
                "touched RAM  tracing requires timestamps tracing to be enabled"
            );
        }
        assert_eq!(ram_size % 4, 0);
        assert_eq!(rom_bound % 4, 0);

        let num_words = ram_size / 4;
        let ram_words_last_live_timestamps = if TRACE_TIMESTAMPS {
            vec![0; num_words]
        } else {
            vec![]
        };
        // let access_bitmask = if TRACE_TOUCHED_RAM {
        //     let num_pages = num_words.next_multiple_of(1 << PAGE_WORDS_LOG_SIZE);
        //     bitvec![0; num_pages]
        // } else {
        //     BitVec::new()
        // };

        let num_touched_ram_cells_in_page = if TRACE_TOUCHED_RAM {
            let num_pages = num_words.div_ceil(1 << PAGE_WORDS_LOG_SIZE);
            vec![0; num_pages]
        } else {
            vec![]
        };

        // let bitmask_len = if TRACE_TOUCHED_RAM {
        //     num_words.div_ceil(usize::BITS as usize)
        // } else {
        //     0
        // };

        Self {
            register_last_live_timestamps: [0; 32],
            ram_words_last_live_timestamps,
            // access_bitmask,
            // num_touched_ram_cells: 0,
            num_touched_ram_cells_in_page,
            num_touched_ram_cells_total: 0,
            rom_bound,
        }
    }

    #[inline(always)]
    pub(crate) fn mark_register_use(
        &mut self,
        reg_idx: u32,
        write_timestamp: TimestampScalar,
    ) -> TimestampScalar {
        if !TRACE_TIMESTAMPS {
            return 0;
        }
        unsafe {
            let read_timestamp = core::mem::replace(
                self.register_last_live_timestamps
                    .get_unchecked_mut(reg_idx as usize),
                write_timestamp,
            );
            debug_assert!(read_timestamp < write_timestamp);

            read_timestamp
        }
    }

    #[inline(always)]
    pub(crate) fn mark_ram_slot_use(
        &mut self,
        phys_word_idx: u32,
        write_timestamp: TimestampScalar,
    ) -> TimestampScalar {
        // if TRACE_TOUCHED_RAM {
        //     // mark memory slot as touched
        //     //     let bookkeeping_word_idx = (phys_word_idx / usize::BITS) as usize;
        //     //     let bit_idx = phys_word_idx % usize::BITS;
        //     //     unsafe {
        //     //         let is_new_cell = (*self.access_bitmask.get_unchecked(bookkeeping_word_idx)
        //     //             & (1 << bit_idx))
        //     //             == 0;
        //     //         *self.access_bitmask.get_unchecked_mut(bookkeeping_word_idx) |= 1 << bit_idx;
        //     //         self.num_touched_ram_cells += is_new_cell as usize;
        //     //     }
        //
        //     let page_idx = (phys_word_idx >> PAGE_WORDS_LOG_SIZE) as usize;
        //     unsafe { self.access_bitmask.set_unchecked(page_idx, true) };
        // }

        if !TRACE_TIMESTAMPS {
            return 0;
        }
        let read_timestamp = unsafe {
            core::mem::replace(
                self.ram_words_last_live_timestamps
                    .get_unchecked_mut(phys_word_idx as usize),
                write_timestamp,
            )
        };

        if TRACE_TOUCHED_RAM {
            if read_timestamp == 0 {
                // this is a new cell
                let page_idx = (phys_word_idx >> PAGE_WORDS_LOG_SIZE) as usize;
                unsafe { *self.num_touched_ram_cells_in_page.get_unchecked_mut(page_idx) += 1 };
                self.num_touched_ram_cells_total += 1;
            }
        }

        debug_assert!(read_timestamp < write_timestamp);
        read_timestamp
    }

    pub fn get_touched_ram_cells_count(&self) -> u32 {
        assert!(TRACE_TOUCHED_RAM);
        self.num_touched_ram_cells_in_page.iter().copied().sum::<u32>()
    }

    pub fn get_setup_and_teardown_chunks<'a, A: GoodAllocator + 'a>(
        &'a self,
        memory: &'a [u32],
        chunk_size: usize,
        available_chunks: impl IntoIterator<Item = ShuffleRamSetupAndTeardown<A>> + 'a,
    ) -> (
        usize,
        impl Iterator<Item = ShuffleRamSetupAndTeardown<A>> + 'a,
    ) {
        assert!(TRACE_TOUCHED_RAM);
        let page_indexes = self
            .num_touched_ram_cells_in_page
            .iter()
            .copied()
            .enumerate()
            .filter_map(move |(index, count)| if count == 0 { None } else { Some(index) })
            .collect_vec();
        let mut src = page_indexes.into_iter()
            .flat_map(move |index| {
                let range_start = index << PAGE_WORDS_LOG_SIZE;
                let range_end = index + 1 << PAGE_WORDS_LOG_SIZE;
                let values = &memory[range_start..range_end];
                let timestamps = &self.ram_words_last_live_timestamps[range_start..range_end];
                timestamps
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(move |(index, timestamp)| {
                        if timestamp != 0 {
                            let result = LazyInitAndTeardown {
                                address: (range_start as u32 + index as u32) << 2,
                                teardown_value: unsafe { *values.get_unchecked(index) },
                                teardown_timestamp: TimestampData::from_scalar(timestamp),
                            };
                            Some(result)
                        } else {
                            None
                        }
                    })
            });
        let touched_ram_cells_count = self.get_touched_ram_cells_count() as usize;
        let chunks_needed = touched_ram_cells_count.div_ceil(chunk_size);
        let padding_size = chunks_needed * chunk_size - touched_ram_cells_count;
        let chunks = available_chunks
            .into_iter()
            .take(chunks_needed)
            .enumerate()
            .map(move |(index, mut chunk)| {
                let data = &mut chunk.lazy_init_data;
                assert_eq!(data.len(), chunk_size);
                let (padding, dst) = data.split_at_mut(if index == 0 { padding_size } else { 0 });
                padding.fill(LazyInitAndTeardown::default());
                dst.fill_with(|| unsafe { src.next().unwrap_unchecked() });
                chunk
            });
        (chunks_needed, chunks)
    }
}

pub struct DelegationTracingData<A: GoodAllocator = Global> {
    pub witnesses: HashMap<u16, DelegationWitness<A>>,
    pub delegation_witness_factories: HashMap<u16, Box<dyn Fn() -> DelegationWitness<A>>>,
}

pub struct MagicTracer<
    C: MachineConfig = IMStandardIsaConfig,
    A: GoodAllocator = Global,
    const TRACE_TOUCHED_RAM: bool = false,
    const TRACE_TIMESTAMPS: bool = false,
    const TRACE_CYCLES: bool = false,
    const TRACE_DELEGATIONS: bool = false,
> {
    pub bookkeeping_aux_data: RamTracingData<TRACE_TOUCHED_RAM, TRACE_TIMESTAMPS>,
    pub trace_chunk: CycleData<C, A>,
    pub delegation_tracer: DelegationTracingData<A>,
    pub chunk_size: usize,
    pub current_timestamp: TimestampScalar,
    emit_delegation_trace_chunk_fn: Box<dyn Fn(u16, DelegationWitness<A>)>,
}

const RS1_ACCESS_IDX: TimestampScalar = 0;
const RS2_ACCESS_IDX: TimestampScalar = 1;
const RD_ACCESS_IDX: TimestampScalar = 2;
const DELEGATION_ACCESS_IDX: TimestampScalar = 3;

const RAM_READ_ACCESS_IDX: TimestampScalar = RS2_ACCESS_IDX;
const RAM_WRITE_ACCESS_IDX: TimestampScalar = RD_ACCESS_IDX;

impl<
        C: MachineConfig,
        A: GoodAllocator,
        const TRACE_TOUCHED_RAM: bool,
        const TRACE_TIMESTAMPS: bool,
        const TRACE_CYCLES: bool,
        const TRACE_DELEGATIONS: bool,
    > MagicTracer<C, A, TRACE_TOUCHED_RAM, TRACE_TIMESTAMPS, TRACE_CYCLES, TRACE_DELEGATIONS>
{
    pub fn new(
        initial_timestamp: TimestampScalar,
        bookkeeping_aux_data: RamTracingData<TRACE_TOUCHED_RAM, TRACE_TIMESTAMPS>,
        delegation_tracer: DelegationTracingData<A>,
        chunk_size: usize,
        emit_delegation_trace_chunk_fn: Box<dyn Fn(u16, DelegationWitness<A>)>,
    ) -> Self {
        if TRACE_CYCLES || TRACE_DELEGATIONS {
            assert!(
                TRACE_TIMESTAMPS,
                "RAM timestamps bookkeeping is needed for cycles or delegation tracing"
            );
        }

        assert!((chunk_size + 1).is_power_of_two());

        let trace_chunk = if TRACE_CYCLES {
            CycleData::<C, A>::new_with_cycles_capacity(chunk_size)
        } else {
            CycleData::<C, A>::dummy()
        };

        Self {
            bookkeeping_aux_data,
            trace_chunk,
            delegation_tracer,
            chunk_size,
            current_timestamp: initial_timestamp,
            emit_delegation_trace_chunk_fn,
        }
    }

    pub fn prepare_for_next_chunk_and_return_processed<const NEXT_TRACE_CYCLES: bool>(
        self,
        timestamp: TimestampScalar,
    ) -> (
        MagicTracer<
            C,
            A,
            TRACE_TOUCHED_RAM,
            TRACE_TIMESTAMPS,
            NEXT_TRACE_CYCLES,
            TRACE_DELEGATIONS,
        >,
        Option<CycleData<C, A>>,
    ) {
        let Self {
            bookkeeping_aux_data,
            trace_chunk,
            delegation_tracer,
            chunk_size,
            current_timestamp,
            emit_delegation_trace_chunk_fn,
        } = self;
        let tracer = MagicTracer::new(
            timestamp,
            bookkeeping_aux_data,
            delegation_tracer,
            chunk_size,
            emit_delegation_trace_chunk_fn,
        );
        let data = if TRACE_CYCLES {
            trace_chunk.assert_at_capacity();
            Some(trace_chunk)
        } else {
            None
        };
        (tracer, data)
    }
}

// impl<C: MachineConfig, A: GoodAllocator, const TRACE_DELEGATIONS: bool>
//     MagicTracer<C, A, true, true, TRACE_DELEGATIONS>
// {
//     pub fn skip_tracing_chunk(
//         self,
//         timestamp: TimestampScalar,
//     ) -> (
//         MagicTracer<C, A, true, false, TRACE_DELEGATIONS>,
//         CycleData<C, A>,
//     ) {
//         let Self {
//             bookkeeping_aux_data,
//             trace_chunk,
//             traced_chunks,
//             delegation_tracer,
//             chunk_size,
//             current_timestamp,
//         } = self;
//         assert!(traced_chunks.is_empty(), "chunks must not be accumulated");
//         let _ = current_timestamp;
//
//         trace_chunk.assert_at_capacity();
//
//         let new_self = MagicTracer {
//             bookkeeping_aux_data,
//             trace_chunk: CycleData::<C, A>::dummy(),
//             traced_chunks,
//             delegation_tracer,
//             chunk_size,
//             current_timestamp: timestamp,
//         };
//
//         (new_self, trace_chunk)
//     }
// }
//
// impl<C: MachineConfig, A: GoodAllocator, const TRACE_DELEGATIONS: bool>
//     MagicTracer<C, A, true, false, TRACE_DELEGATIONS>
// {
//     pub fn start_tracing_chunk(
//         self,
//         timestamp: TimestampScalar,
//     ) -> MagicTracer<C, A, true, true, TRACE_DELEGATIONS> {
//         let Self {
//             bookkeeping_aux_data,
//             trace_chunk,
//             traced_chunks,
//             delegation_tracer,
//             chunk_size,
//             current_timestamp,
//         } = self;
//         assert!(traced_chunks.is_empty(), "chunks must not be accumulated");
//         let _ = current_timestamp;
//         let _ = trace_chunk;
//
//         MagicTracer {
//             bookkeeping_aux_data,
//             trace_chunk: CycleData::<C, A>::new_with_cycles_capacity(self.chunk_size),
//             traced_chunks,
//             delegation_tracer,
//             chunk_size,
//             current_timestamp: timestamp,
//         }
//     }
// }

impl<
        C: MachineConfig,
        A: GoodAllocator,
        const TRACE_TOUCHED_RAM: bool,
        const TRACE_TIMESTAMPS: bool,
        const TRACE_CYCLES: bool,
        const TRACE_DELEGATIONS: bool,
    > Tracer<C>
    for MagicTracer<C, A, TRACE_TOUCHED_RAM, TRACE_TIMESTAMPS, TRACE_CYCLES, TRACE_DELEGATIONS>
{
    #[inline(always)]
    fn at_cycle_start(&mut self, current_state: &RiscV32State<C>) {
        if !TRACE_CYCLES {
            return;
        }
        unsafe {
            self.trace_chunk
                .per_cycle_data
                .push_within_capacity(EMPTY_SINGLE_CYCLE_TRACING_DATA)
                .unwrap_unchecked();
            self.trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked()
                .pc = current_state.pc;
        }
    }

    #[inline(always)]
    fn at_cycle_end(&mut self, _current_state: &RiscV32State<C>) {
        if !TRACE_TIMESTAMPS {
            return;
        }
        self.current_timestamp += TIMESTAMP_STEP;
    }

    #[inline(always)]
    fn at_cycle_start_ext(&mut self, current_state: &RiscV32StateForUnrolledProver<C>) {
        if !TRACE_CYCLES {
            return;
        }
        unsafe {
            self.trace_chunk
                .per_cycle_data
                .push_within_capacity(EMPTY_SINGLE_CYCLE_TRACING_DATA)
                .unwrap_unchecked();
            self.trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked()
                .pc = current_state.pc;
        }
    }

    #[inline(always)]
    fn at_cycle_end_ext(&mut self, _current_state: &RiscV32StateForUnrolledProver<C>) {
        if !TRACE_TIMESTAMPS {
            return;
        }
        self.current_timestamp += TIMESTAMP_STEP;
    }

    #[inline(always)]
    fn trace_opcode_read(&mut self, _phys_address: u64, _read_value: u32) {
        // Nothing, opcodes are expected to be read from ROM
    }

    #[inline(always)]
    fn trace_rs1_read(&mut self, reg_idx: u32, read_value: u32) {
        // dbg!(reg_idx, read_value);
        // dbg!(TRACE_TIMESTAMPS);
        if !TRACE_TIMESTAMPS {
            return;
        }
        debug_assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);

        let write_timestamp = self.current_timestamp + RS1_ACCESS_IDX;

        let read_timestamp = self
            .bookkeeping_aux_data
            .mark_register_use(reg_idx, write_timestamp);

        if !TRACE_CYCLES {
            return;
        }
        unsafe {
            let dst = self
                .trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked();
            dst.rs1_read_value = read_value;
            dst.rs1_read_timestamp = TimestampData::from_scalar(read_timestamp);
            dst.rs1_reg_idx = reg_idx as u16;
        }
    }

    #[inline(always)]
    fn trace_rs2_read(&mut self, reg_idx: u32, read_value: u32) {
        // dbg!(reg_idx, read_value);
        if !TRACE_TIMESTAMPS {
            return;
        }
        debug_assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);

        let write_timestamp = self.current_timestamp + RS2_ACCESS_IDX;

        let read_timestamp = self
            .bookkeeping_aux_data
            .mark_register_use(reg_idx, write_timestamp);

        // NOTE: we reuse this access for RAM LOAD, but it's not traced if LOAD op happens

        if !TRACE_CYCLES {
            return;
        }
        unsafe {
            let dst = self
                .trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked();
            dst.rs2_or_mem_word_read_value = read_value;
            dst.rs2_or_mem_word_address = RegIndexOrMemWordIndex::register(reg_idx as u8);
            dst.rs2_or_mem_read_timestamp = TimestampData::from_scalar(read_timestamp);
        }
    }

    #[inline(always)]
    fn trace_rd_write(&mut self, reg_idx: u32, read_value: u32, written_value: u32) {
        // dbg!(reg_idx, read_value, written_value);
        if !TRACE_TIMESTAMPS {
            return;
        }
        debug_assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);

        // this happens only if RAM write didn't happen (and opcodes like BRANCH write 0 into x0)

        let write_timestamp = self.current_timestamp + RD_ACCESS_IDX;

        let read_timestamp = self
            .bookkeeping_aux_data
            .mark_register_use(reg_idx, write_timestamp);

        if !TRACE_CYCLES {
            return;
        }
        unsafe {
            let dst = self
                .trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked();

            let mut written_value = written_value;
            if reg_idx == 0 {
                assert_eq!(read_value, 0);
                // we can flush anything here
                written_value = 0;
            }

            dst.rd_or_mem_word_read_value = read_value;
            dst.rd_or_mem_word_write_value = written_value;
            dst.rd_or_mem_word_address = RegIndexOrMemWordIndex::register(reg_idx as u8);
            dst.rd_or_mem_read_timestamp = TimestampData::from_scalar(read_timestamp);
        }
    }

    #[inline(always)]
    fn trace_non_determinism_read(&mut self, read_value: u32) {
        if !TRACE_TIMESTAMPS {
            return;
        }

        debug_assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);

        if !TRACE_CYCLES {
            return;
        }

        unsafe {
            let dst = self
                .trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked();
            dst.non_determinism_read = read_value;
        }
    }

    #[inline(always)]
    fn trace_non_determinism_write(&mut self, _written_value: u32) {
        // do nothing
    }

    #[inline(always)]
    fn trace_ram_read(&mut self, phys_address: u64, read_value: u32) {
        if !(TRACE_TOUCHED_RAM || TRACE_TIMESTAMPS) {
            return;
        }

        if TRACE_TIMESTAMPS {
            debug_assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);
        }

        assert!(phys_address < (1u64 << 32));
        assert_eq!(phys_address % 4, 0);

        let (address, read_value) = if phys_address < self.bookkeeping_aux_data.rom_bound as u64 {
            // ROM read, substituted as read 0 from 0
            (0, 0)
        } else {
            (phys_address, read_value)
        };

        let write_timestamp = if TRACE_TIMESTAMPS {
            self.current_timestamp + RAM_READ_ACCESS_IDX
        } else {
            0
        };

        let phys_word_idx = address / 4;
        let read_timestamp = self
            .bookkeeping_aux_data
            .mark_ram_slot_use(phys_word_idx as u32, write_timestamp);

        if !TRACE_CYCLES {
            return;
        }
        unsafe {
            let dst = self
                .trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked();
            // record
            dst.rs2_or_mem_word_read_value = read_value;
            dst.rs2_or_mem_word_address = RegIndexOrMemWordIndex::memory(address as u32);
            dst.rs2_or_mem_read_timestamp = TimestampData::from_scalar(read_timestamp);
        }
    }

    #[inline(always)]
    fn trace_ram_read_write(&mut self, phys_address: u64, read_value: u32, written_value: u32) {
        if !(TRACE_TOUCHED_RAM || TRACE_TIMESTAMPS) {
            return;
        }
        if TRACE_TIMESTAMPS {
            debug_assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);
        }
        assert!(phys_address < (1u64 << 32));
        assert_eq!(phys_address % 4, 0);

        assert!(
            phys_address >= self.bookkeeping_aux_data.rom_bound as u64,
            "Cannot write to ROM"
        );

        // RAM write happens BEFORE rd write

        let write_timestamp = if TRACE_TIMESTAMPS {
            self.current_timestamp + RAM_WRITE_ACCESS_IDX
        } else {
            0
        };

        let phys_word_idx = phys_address / 4;
        let read_timestamp = self
            .bookkeeping_aux_data
            .mark_ram_slot_use(phys_word_idx as u32, write_timestamp);

        if !TRACE_CYCLES {
            return;
        }
        // record
        unsafe {
            let dst = self
                .trace_chunk
                .per_cycle_data
                .last_mut()
                .unwrap_unchecked();
            dst.rd_or_mem_word_read_value = read_value;
            dst.rd_or_mem_word_write_value = written_value;
            dst.rd_or_mem_word_address = RegIndexOrMemWordIndex::memory(phys_address as u32);
            dst.rd_or_mem_read_timestamp = TimestampData::from_scalar(read_timestamp);
        }
    }

    #[inline(always)]
    fn trace_address_translation(
        &mut self,
        _satp_value: u32,
        _virtual_address: u64,
        _phys_address: u64,
    ) {
        // nothing
    }

    fn record_delegation(
        &mut self,
        access_id: u32,
        base_register: u32,
        register_accesses: &mut [RegisterOrIndirectReadWriteData],
        indirect_read_addresses: &[u32],
        indirect_reads: &mut [RegisterOrIndirectReadData],
        indirect_write_addresses: &[u32],
        indirect_writes: &mut [RegisterOrIndirectReadWriteData],
    ) {
        if !(TRACE_TOUCHED_RAM || TRACE_TIMESTAMPS) {
            return;
        }
        if TRACE_TIMESTAMPS {
            assert_eq!(self.current_timestamp % TIMESTAMP_STEP, 0);
        }
        assert_eq!(indirect_read_addresses.len(), indirect_reads.len());
        assert_eq!(indirect_write_addresses.len(), indirect_writes.len());

        let delegation_type = access_id as u16;

        if TRACE_CYCLES {
            // mark as delegation
            unsafe {
                let dst = self
                    .trace_chunk
                    .per_cycle_data
                    .last_mut()
                    .unwrap_unchecked();
                dst.delegation_request = delegation_type;
            }
        }

        let write_timestamp = if TRACE_TIMESTAMPS {
            self.current_timestamp + DELEGATION_ACCESS_IDX
        } else {
            0
        };

        if TRACE_DELEGATIONS {
            let current_tracer = self
                .delegation_tracer
                .witnesses
                .entry(delegation_type)
                .or_insert_with(|| {
                    let new_tracer = (self
                        .delegation_tracer
                        .delegation_witness_factories
                        .get(&delegation_type)
                        .unwrap())();

                    new_tracer
                });

            assert_eq!(current_tracer.base_register_index, base_register);

            unsafe {
                // trace register part
                let mut register_index = base_register;
                for dst in register_accesses.iter_mut() {
                    let read_timestamp = self
                        .bookkeeping_aux_data
                        .mark_register_use(register_index, write_timestamp);
                    dst.timestamp = TimestampData::from_scalar(read_timestamp);

                    register_index += 1;
                }

                // formal reads and writes
                for (phys_address, dst) in indirect_read_addresses
                    .iter()
                    .zip(indirect_reads.iter_mut())
                {
                    let phys_address = *phys_address;
                    let phys_word_idx = phys_address / 4;

                    let read_timestamp = self
                        .bookkeeping_aux_data
                        .mark_ram_slot_use(phys_word_idx as u32, write_timestamp);

                    dst.timestamp = TimestampData::from_scalar(read_timestamp);
                }

                for (phys_address, dst) in indirect_write_addresses
                    .iter()
                    .zip(indirect_writes.iter_mut())
                {
                    let phys_address = *phys_address;
                    let phys_word_idx = phys_address / 4;

                    let read_timestamp = self
                        .bookkeeping_aux_data
                        .mark_ram_slot_use(phys_word_idx as u32, write_timestamp);

                    dst.timestamp = TimestampData::from_scalar(read_timestamp);
                }
            }

            current_tracer
                .register_accesses
                .extend_from_slice(&*register_accesses);
            current_tracer
                .indirect_reads
                .extend_from_slice(&*indirect_reads);
            current_tracer
                .indirect_writes
                .extend_from_slice(&*indirect_writes);
            current_tracer
                .write_timestamp
                .push_within_capacity(TimestampData::from_scalar(write_timestamp))
                .unwrap();

            // swap if needed
            // assert that all lengths are the same
            current_tracer.assert_consistency();
            let should_replace = current_tracer.at_capacity();
            if should_replace {
                let new_tracer = (self
                    .delegation_tracer
                    .delegation_witness_factories
                    .get(&delegation_type)
                    .unwrap())();
                let current_tracer = core::mem::replace(
                    self.delegation_tracer
                        .witnesses
                        .get_mut(&delegation_type)
                        .unwrap(),
                    new_tracer,
                );
                (self.emit_delegation_trace_chunk_fn)(delegation_type, current_tracer);
            }
        } else {
            // we only need to mark RAM and register use

            if TRACE_TIMESTAMPS {
                // trace register part
                let mut register_index = base_register;
                for _reg in register_accesses.iter() {
                    let _read_timestamp = self
                        .bookkeeping_aux_data
                        .mark_register_use(register_index, write_timestamp);

                    register_index += 1;
                }
            }

            // formal reads and writes
            for phys_address in indirect_read_addresses.iter() {
                let phys_address = *phys_address;
                let phys_word_idx = phys_address / 4;

                let _read_timestamp = self
                    .bookkeeping_aux_data
                    .mark_ram_slot_use(phys_word_idx as u32, write_timestamp);
            }

            for phys_address in indirect_write_addresses.iter() {
                let phys_address = *phys_address;
                let phys_word_idx = phys_address / 4;

                let _read_timestamp = self
                    .bookkeeping_aux_data
                    .mark_ram_slot_use(phys_word_idx as u32, write_timestamp);
            }
        }
    }
}
