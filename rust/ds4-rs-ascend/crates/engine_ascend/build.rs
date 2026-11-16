//! Point the linker at the CANN runtime for the `ascend` feature.
//!
//! `device.rs` carries `#[link(name = "runtime")]` and `#[link(name = "ascendcl")]`, but nothing told
//! the linker where those live: `LIBRARY_PATH` and `RUSTFLAGS` both failed to add the directory, and
//! the feature simply would not link. A build script is the supported place for this, and it makes
//! the feature buildable anywhere CANN is installed rather than only where the environment happens
//! to be set up by hand.

fn main() {
    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    if std::env::var_os("CARGO_FEATURE_ASCEND").is_none() {
        return;
    }
    // `ASCEND_HOME_PATH` first; then the conventional roots, newest toolkit first, because the
    // variable is often unset in a fresh shell while the toolkit is installed.
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Some(home) = std::env::var_os("ASCEND_HOME_PATH") {
        roots.push(home.into());
    }
    roots.push("/usr/local/Ascend/ascend-toolkit/latest".into());
    for entry in std::fs::read_dir("/usr/local/Ascend").into_iter().flatten().flatten() {
        roots.push(entry.path());
    }

    let mut added = 0;
    for root in &roots {
        // The toolkit nests the libraries one or two levels down (`<root>/aarch64-linux/lib64`,
        // `<root>/*/aarch64-linux/lib64`), so ask the filesystem rather than guessing a layout.
        for sub in ["", "aarch64-linux/lib64", "lib64", "runtime/lib64"] {
            let dir = if sub.is_empty() { root.clone() } else { root.join(sub) };
            if dir.join("libruntime.so").exists() || dir.join("libascendcl.so").exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
                added += 1;
            }
        }
        if let Ok(entries) = std::fs::read_dir(root) {
            for e in entries.flatten() {
                for sub in ["aarch64-linux/lib64", "lib64"] {
                    let dir = e.path().join(sub);
                    if dir.join("libruntime.so").exists() {
                        println!("cargo:rustc-link-search=native={}", dir.display());
                        added += 1;
                    }
                }
            }
        }
    }
    // `libruntime.so` itself has unresolved `cce::runtime::*` references that CANN's other libraries
    // provide at run time. Refusing those at LINK time makes the feature unbuildable against a
    // vendor .so that is meant to be resolved dynamically, so allow them here.
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");

    if added == 0 {
        println!(
            "cargo:warning=ascend feature: no CANN lib directory found (libruntime.so); \
             the link will fail until ASCEND_HOME_PATH points at an install"
        );
    }
}
