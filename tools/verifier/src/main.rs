#![no_std]
#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![no_main]

extern "C" {
    // Boundaries of the heap
    static mut _sheap: usize;
    static mut _eheap: usize;

    // Boundaries of the stack
    static mut _sstack: usize;
    static mut _estack: usize;
}

core::arch::global_asm!(include_str!("asm/asm_reduced.S"));

#[no_mangle]
extern "C" fn eh_personality() {}

#[link_section = ".init.rust"]
#[export_name = "_start_rust"]
unsafe extern "C" fn start_rust() -> ! {
    main()
}

#[export_name = "_setup_interrupts"]
pub unsafe fn custom_setup_interrupts() {
    extern "C" {
        fn _machine_start_trap();
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct MachineTrapFrame {
    pub registers: [u32; 32],
}

/// Exception (trap) handler in rust.
/// Called from the asm/asm.S
#[link_section = ".trap.rust"]
#[export_name = "_machine_start_trap_rust"]
pub extern "C" fn machine_start_trap_rust(_trap_frame: *mut MachineTrapFrame) -> usize {
    {
        unsafe { core::hint::unreachable_unchecked() }
    }
}

#[cfg(feature = "panic_output")]
#[macro_export]
macro_rules! print
{
	($($args:tt)+) => ({
		use core::fmt::Write;
		let _ = write!(riscv_common::QuasiUART::new(), $($args)+);
	});
}

#[cfg(feature = "panic_output")]
#[macro_export]
macro_rules! println
{
	() => ({
		crate::print!("\r\n")
	});
	($fmt:expr) => ({
		crate::print!(concat!($fmt, "\r\n"))
	});
	($fmt:expr, $($args:tt)+) => ({
		crate::print!(concat!($fmt, "\r\n"), $($args)+)
	});
}

#[cfg(feature = "panic_output")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    print_panic(_info);

    riscv_common::zksync_os_finish_error()
}

#[cfg(feature = "panic_output")]
fn print_panic(_info: &core::panic::PanicInfo) {
    print!("Aborting: ");
    if let Some(p) = _info.location() {
        println!("line {}, file {}", p.line(), p.file(),);

        if let Some(m) = _info.message().as_str() {
            println!("line {}, file {}: {}", p.line(), p.file(), m,);
        } else {
            println!(
                "line {}, file {}, message:\n{}",
                p.line(),
                p.file(),
                _info.message()
            );
        }
    } else {
        println!("no information available");
    }
}

#[cfg(feature = "base_layer")]
unsafe fn workload() -> ! {
    let output = full_statement_verifier::verify_base_layer();
    riscv_common::zksync_os_finish_success_extended(&output);
}

#[cfg(any(feature = "recursion_step", feature = "recursion_step_no_delegation"))]
unsafe fn workload() -> ! {
    let output = full_statement_verifier::verify_recursion_layer();
    riscv_common::zksync_os_finish_success_extended(&output);
}

#[cfg(feature = "final_recursion_step")]
unsafe fn workload() -> ! {
    let output = full_statement_verifier::verify_final_recursion_layer();
    riscv_common::zksync_os_finish_success_extended(&output);
}

#[cfg(any(
    feature = "universal_circuit",
    feature = "universal_circuit_no_delegation"
))]
// This verifier can handle any circuit and any layer.
// It uses the first word in the input to determine which circuit to verify.
unsafe fn workload() -> ! {
    let metadata = riscv_common::csr_read_word();

    // These values should match VerifierCircuitsIdentifiers.
    match metadata {
        0 => {
            let output = full_statement_verifier::verify_base_layer();
            riscv_common::zksync_os_finish_success_extended(&output);
        }
        1 => {
            let output = full_statement_verifier::verify_recursion_layer();
            riscv_common::zksync_os_finish_success_extended(&output);
        }
        2 => {
            let output = full_statement_verifier::verify_final_recursion_layer();
            riscv_common::zksync_os_finish_success_extended(&output);
        }
        3 => {
            full_statement_verifier::RISC_V_VERIFIER_PTR(
                &mut core::mem::MaybeUninit::uninit().assume_init_mut(),
                &mut full_statement_verifier::verifier_common::ProofPublicInputs::uninit(),
            );
            riscv_common::zksync_os_finish_success(&[1, 2, 3, 0, 0, 0, 0, 0]);
        }
        other => {
            let Some(pos) =
                full_statement_verifier::RECURSION_LAYER_CIRCUITS_VERIFICATION_PARAMETERS
                    .iter()
                    .position(|el| el.0 == other)
            else {
                riscv_common::zksync_os_finish_error();
            };
            let verification_ptr =
                full_statement_verifier::RECURSION_LAYER_CIRCUITS_VERIFICATION_PARAMETERS[pos].3;
            (verification_ptr)(
                &mut core::mem::MaybeUninit::uninit().assume_init_mut(),
                &mut full_statement_verifier::verifier_common::ProofPublicInputs::uninit(),
            );
            riscv_common::zksync_os_finish_success(&[1, 1, 1, 1, 1, 1, 1, 1]);
        }
    }
}

#[cfg(feature = "verifier_tests")]
unsafe fn workload() -> ! {
    use core::mem::MaybeUninit;
    use verifier::concrete::size_constants::*;
    use verifier::verify;
    use verifier::ProofPublicInputs;

    use verifier::verifier_common::ProofOutput;

    let mut proof_output: ProofOutput<TREE_CAP_SIZE, NUM_COSETS, NUM_DELEGATION_CHALLENGES, 1> =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut state_variables = ProofPublicInputs::uninit();

    unsafe { verify(&mut proof_output, &mut state_variables) };

    let mut output = [0u32; 16];
    for i in 0..16 {
        output[i] = i as u32;
    }
    riscv_common::zksync_os_finish_success_extended(&output)
}

#[inline(never)]
fn main() -> ! {
    unsafe { workload() }
}
