//! Embeds the built dashboard (`dashboard/dist/`, committed) into the binary:
//! writes `$OUT_DIR/dashboard_assets.rs`, a table of every file under it with
//! its path relative to `dist/`, its content type and its bytes
//! (`include_bytes!`), which `src/serve/dashboard.rs` serves. `cargo build`
//! never needs node; a stale `dist/` is caught by the dashboard's own check
//! (`npm --prefix dashboard run check:dist`).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The content types the dashboard's files may have, by extension. A file
/// with any other extension fails the build: serving it as a guess would be
/// worse than noticing.
const TYPES: &[(&str, &str)] = &[
    ("html", "text/html; charset=utf-8"),
    ("js", "text/javascript; charset=utf-8"),
    ("css", "text/css; charset=utf-8"),
    ("woff2", "font/woff2"),
    ("txt", "text/plain; charset=utf-8"),
    ("svg", "image/svg+xml"),
    ("json", "application/json"),
    ("png", "image/png"),
];

fn main() {
    let root =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let dist = root.join("dashboard").join("dist");
    println!("cargo:rerun-if-changed={}", dist.display());
    if !dist.join("index.html").is_file() {
        panic!(
            "{} is missing: build the dashboard first with `npm --prefix dashboard run build` \
             (dashboard/dist/ is committed; a fresh checkout has it)",
            dist.join("index.html").display()
        );
    }
    let mut files = Vec::new();
    collect(&dist, &mut files);
    files.sort();
    let mut table = String::from(
        "/// Every file of `dashboard/dist/`: path relative to it, content type, bytes.\n\
         pub static FILES: &[(&str, &str, &[u8])] = &[\n",
    );
    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path
            .strip_prefix(&dist)
            .expect("under dist")
            .to_str()
            .unwrap_or_else(|| panic!("{} is not UTF-8", path.display()))
            .replace('\\', "/");
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let Some((_, content_type)) = TYPES.iter().find(|(e, _)| *e == extension)
        else {
            panic!(
                "{}: no content type for .{extension} files; add it to build.rs",
                path.display()
            );
        };
        writeln!(
            table,
            "    ({relative:?}, {content_type:?}, include_bytes!({:?})),",
            path.display().to_string()
        )
        .expect("write to a string");
    }
    table.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    std::fs::write(out.join("dashboard_assets.rs"), table)
        .expect("writing dashboard_assets.rs");
    vulkan_kernels(&root, &out);
}

/// Linux: compiles the Vulkan kernels (`src/kernels/vulkan/*.comp`, GLSL) to
/// SPIR-V with `glslc` and writes `$OUT_DIR/vulkan_kernels.rs`, the table
/// `KERNELS` of each kernel's name and SPIR-V, which `src/vulkan.rs` builds
/// its pipelines from. A file holds the kernels of one Metal source: every
/// `KERNEL_<name>` it mentions is a kernel `<name>` (the Metal function's
/// name), compiled from the file with `-DKERNEL_<name>`.
/// `src/kernels/vulkan/PORTING.md` has the rules the shaders follow.
fn vulkan_kernels(root: &Path, out: &Path) {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let dir = root.join("src").join("kernels").join("vulkan");
    println!("cargo:rerun-if-changed={}", dir.display());
    // The tests' own kernels (`tests/metal/*.metal` on Metal) are
    // `tests/vulkan/*.comp`, built the same way and with the same includes.
    let tests = root.join("tests").join("vulkan");
    let mut sources: Vec<PathBuf> = Vec::new();
    for from in [&dir, &tests] {
        if !from.is_dir() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", from.display());
        sources.extend(
            std::fs::read_dir(from)
                .unwrap_or_else(|e| panic!("listing {}: {e}", from.display()))
                .map(|entry| entry.expect("directory entry").path())
                .filter(|path| path.is_file()),
        );
    }
    sources.sort();
    let mut kernels: Vec<(String, PathBuf)> = Vec::new();
    for path in &sources {
        println!("cargo:rerun-if-changed={}", path.display());
        if path.extension().and_then(|e| e.to_str()) != Some("comp") {
            continue;
        }
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let mut names: Vec<&str> = text
            .match_indices("KERNEL_")
            .map(|(at, prefix)| {
                let rest = &text[at + prefix.len()..];
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                &rest[..end]
            })
            .filter(|name| !name.is_empty())
            .collect();
        names.sort();
        names.dedup();
        for name in names {
            if let Some((_, other)) = kernels.iter().find(|(n, _)| n == name) {
                panic!("kernel {name} is in {} and {}", other.display(), path.display());
            }
            kernels.push((name.to_owned(), path.clone()));
        }
    }
    let spirv = out.join("vulkan");
    std::fs::create_dir_all(&spirv).expect("creating the SPIR-V directory");
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    // Kernels of a file that requires an extension lily can run without
    // (OPTIONAL_EXTENSIONS) are left out when glslc cannot compile them;
    // their callers dispatch them only where the device has the extension
    // and the kernel was built.
    let skipped = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((name, path)) = kernels.get(i) else { break };
                    let result = std::process::Command::new(&glslc)
                        .args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute"])
                        .arg(format!("-DKERNEL_{name}"))
                        .arg("-I")
                        .arg(&dir)
                        .arg(path)
                        .arg("-o")
                        .arg(spirv.join(format!("{name}.spv")))
                        .output();
                    let failure = match result {
                        Ok(output) if output.status.success() => continue,
                        Ok(output) => String::from_utf8_lossy(&output.stderr).into_owned(),
                        Err(e) => format!("running {glslc}: {e}"),
                    };
                    if requires_optional_extension(path) {
                        skipped.lock().expect("skipped").push((name.clone(), failure));
                        continue;
                    }
                    failures
                        .lock()
                        .expect("failures")
                        .push(format!("kernel {name} ({}):\n{failure}", path.display()));
                }
            });
        }
    });
    let failures = failures.into_inner().expect("failures");
    if !failures.is_empty() {
        panic!("{} Vulkan kernels failed to compile:\n{}", failures.len(), failures.join("\n"));
    }
    let skipped = skipped.into_inner().expect("skipped");
    for (name, failure) in &skipped {
        let first = failure.lines().next().unwrap_or("");
        println!("cargo:warning=Vulkan kernel {name} left out (needs an extension glslc lacks): {first}");
    }
    kernels.retain(|(name, _)| !skipped.iter().any(|(s, _)| s == name));
    let mut table = String::from(
        "/// Every Vulkan kernel: its name (the Metal kernel function's) and SPIR-V.\n\
         static KERNELS: &[(&str, &[u8])] = &[\n",
    );
    for (name, _) in &kernels {
        writeln!(
            table,
            "    ({name:?}, include_bytes!({:?})),",
            spirv.join(format!("{name}.spv")).display().to_string()
        )
        .expect("write to a string");
    }
    table.push_str("];\n");
    std::fs::write(out.join("vulkan_kernels.rs"), table)
        .expect("writing vulkan_kernels.rs");
}

/// Shader extensions lily runs without: a kernel file that requires one is
/// built only where glslc has it.
const OPTIONAL_EXTENSIONS: &[&str] = &["GL_NV_cooperative_matrix2"];

/// Whether the kernel file `path` requires one of [`OPTIONAL_EXTENSIONS`].
fn requires_optional_extension(path: &Path) -> bool {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("#extension")
            && line.contains(": require")
            && OPTIONAL_EXTENSIONS.iter().any(|ext| line.contains(ext))
    })
}

/// Every regular file below `dir`, directories descended into (rerun on each,
/// so an added or removed file is noticed). Hidden files (a Finder
/// `.DS_Store`) are not part of the build output and are skipped.
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("listing {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("directory entry");
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.is_file() {
            files.push(path);
        }
    }
}
