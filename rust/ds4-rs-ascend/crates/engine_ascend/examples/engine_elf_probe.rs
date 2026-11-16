//! Register a device ELF with the serving engine and report which kernels bound.
//!
//! `device.rs` binds by MANGLED symbol and the engine already tolerates a partial ELF
//! (`[device] <symbol> not in ELF (rc=507000) -- left on CPU`), so pointing it at an ELF built from
//! the Rust-emitted kernels measures how many slots it can take from it, with the rest falling back
//! rather than failing.
//!
//! usage: engine_elf_probe <elf path>

use engine_ascend::dispatcher::AscendDispatcher;
use engine_core::dispatch::CpuDispatcher;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: engine_elf_probe <elf path>"))?;
    let d = AscendDispatcher::new(CpuDispatcher).with_devices(std::path::Path::new(&path), &[0])?;
    let off = d.offloaded_by();
    println!("elf: {path}");
    println!("kernels the engine bound from it: {}", off.len());
    for k in &off {
        println!("  {k}");
    }
    Ok(())
}
