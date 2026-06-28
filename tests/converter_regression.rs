use std::path::Path;
use std::process::Command;

#[test]
fn bullet_converter_accounts_for_v10_extended_header() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("training/convert_bullet_to_nnue.py");
    let python = r#"
import importlib.util
import os
import struct

script = os.environ["CONVERTER_SCRIPT"]
spec = importlib.util.spec_from_file_location("converter", script)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

header = struct.pack("<II", 0x4E4E5545, 10)
header += bytes([0x80 | 0x01])
header += struct.pack("<HHH", 1024, 0, 0)
header += bytes([16, 1])
header += bytes([0])

expected = 18
got = converter.get_header_size(header + b"\0" * 8)
assert got == expected, (
    f"expected extended v10 header size {expected}, got {got}; "
    "converter would start reading weights from header bytes"
)
"#;

    let output = if cfg!(target_os = "windows") {
        Command::new("python")
    } else {
        Command::new("python3")
    }
        .arg("-c")
        .arg(python)
        .env("CONVERTER_SCRIPT", script)
        .output()
        .expect("python3 should run converter regression");

    assert!(
        output.status.success(),
        "converter should account for v10 extended header bytes\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
