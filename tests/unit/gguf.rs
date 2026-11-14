use super::*;

/// `tests/goldens/gguf_dequant.bin`, written by
/// `tools/reference/glm53/gguf_dequant_golden.c`: records of (type id, block
/// count, raw blocks, decoded f32 values), the blocks from ds4's quantizer
/// and from random bytes, the values from ds4's reference formulas.
const GOLDEN: &[u8] = include_bytes!("../goldens/gguf_dequant.bin");

struct Record<'a> {
    ty: GgmlType,
    raw: &'a [u8],
    values: Vec<f32>,
}

fn records() -> Vec<Record<'static>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < GOLDEN.len() {
        let id = u32::from_le_bytes(GOLDEN[at..at + 4].try_into().unwrap());
        let n = u32::from_le_bytes(GOLDEN[at + 4..at + 8].try_into().unwrap()) as usize;
        at += 8;
        let ty = GgmlType::from_id(id).expect("golden type");
        let raw = &GOLDEN[at..at + n * ty.block_bytes()];
        at += raw.len();
        let count = n * ty.block_values();
        let values = GOLDEN[at..at + 4 * count]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        at += 4 * count;
        out.push(Record { ty, raw, values });
    }
    out
}

#[test]
fn dequantize_matches_ds4_bit_for_bit() {
    let records = records();
    let mut seen = std::collections::BTreeSet::new();
    for r in &records {
        let mut out = vec![0f32; r.values.len()];
        dequantize(r.ty, r.raw, &mut out).unwrap();
        for (i, (a, b)) in out.iter().zip(&r.values).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{:?} value {i}: lily {a} ds4 {b}",
                r.ty
            );
        }
        seen.insert(r.ty.name());
    }
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        ["IQ2_XXS", "Q2_K", "Q4_K", "Q8_0"],
        "every quantized type is covered"
    );
}

#[test]
fn ksigns_matches_ggml_table() {
    // The first and last rows of ggml's literal table.
    assert_eq!(
        KSIGNS_IQ2XS[..16],
        [0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15]
    );
    assert_eq!(
        KSIGNS_IQ2XS[112..],
        [
            240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126,
            255
        ]
    );
}

#[test]
fn row_bytes_of_each_type() {
    assert_eq!(GgmlType::Q8_0.row_bytes(4096).unwrap(), 4352);
    assert_eq!(GgmlType::Q4K.row_bytes(4096).unwrap(), 2304);
    assert_eq!(GgmlType::Q2K.row_bytes(2048).unwrap(), 672);
    assert_eq!(GgmlType::Iq2Xxs.row_bytes(4096).unwrap(), 1056);
    assert_eq!(GgmlType::Bf16.row_bytes(4096).unwrap(), 8192);
    assert!(GgmlType::Q4K.row_bytes(128).is_err());
}

/// Writes a small GGUF with every metadata kind and two tensors and reads
/// it back.
#[test]
fn open_reads_header_and_tensor_table() {
    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }
    let mut f = Vec::new();
    f.extend(b"GGUF");
    f.extend(3u32.to_le_bytes());
    f.extend(2u64.to_le_bytes()); // tensors
    f.extend(4u64.to_le_bytes()); // metadata
    s(&mut f, "general.architecture");
    f.extend(8u32.to_le_bytes());
    s(&mut f, "glm5-next");
    s(&mut f, "glm5-next.block_count");
    f.extend(4u32.to_le_bytes());
    f.extend(46u32.to_le_bytes());
    s(&mut f, "glm5-next.eps");
    f.extend(6u32.to_le_bytes());
    f.extend(1e-5f32.to_le_bytes());
    s(&mut f, "glm5-next.layer_types");
    f.extend(9u32.to_le_bytes());
    f.extend(5u32.to_le_bytes()); // i32 elements
    f.extend(3u64.to_le_bytes());
    for v in [0i32, 0, 1] {
        f.extend(v.to_le_bytes());
    }
    s(&mut f, "a.weight");
    f.extend(2u32.to_le_bytes());
    f.extend(32u64.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    f.extend(8u32.to_le_bytes()); // Q8_0
    f.extend(0u64.to_le_bytes());
    s(&mut f, "b.weight");
    f.extend(1u32.to_le_bytes());
    f.extend(3u64.to_le_bytes());
    f.extend(0u32.to_le_bytes()); // F32
    f.extend(96u64.to_le_bytes());
    while !f.len().is_multiple_of(32) {
        f.push(0);
    }
    let data = f.len() as u64;
    // Two Q8_0 rows: scale 0.5, values 0..32 and -(0..32).
    for sign in [1i8, -1] {
        f.extend(half::f16::from_f32(0.5).to_bits().to_le_bytes());
        f.extend((0..32).map(|i| (sign * i as i8) as u8));
    }
    f.resize(data as usize + 96, 0);
    for v in [1.5f32, -2.0, 3.25] {
        f.extend(v.to_le_bytes());
    }
    let dir =
        std::env::temp_dir().join(format!("lily-gguf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.gguf");
    std::fs::write(&path, &f).unwrap();

    let g = Gguf::open(&path).unwrap();
    assert_eq!(g.architecture().unwrap(), "glm5-next");
    assert_eq!(g.u64("glm5-next.block_count").unwrap(), 46);
    assert_eq!(g.f32("glm5-next.eps").unwrap(), 1e-5);
    let Some(Value::Array(types)) = g.get("glm5-next.layer_types") else { panic!() };
    assert_eq!(
        types.iter().map(|v| v.as_u64().unwrap()).collect::<Vec<_>>(),
        [0, 0, 1]
    );
    let a = g.require("a.weight").unwrap();
    assert_eq!(
        (a.ty, a.offset, a.bytes, a.rows(), a.row_len()),
        (GgmlType::Q8_0, data, 68, 2, 32)
    );
    let rows = g.read_rows_f32(a, 1..2).unwrap();
    assert_eq!(rows, (0..32).map(|i| -0.5 * i as f32).collect::<Vec<_>>());
    let b = g.require("b.weight").unwrap();
    assert_eq!(g.read_rows_f32(b, 0..1).unwrap(), [1.5, -2.0, 3.25]);
    assert!(g.tensor("c.weight").is_none());
    std::fs::remove_dir_all(&dir).ok();
}

/// Bytes of a GGUF v3 file: header counts, then `body` (metadata and the
/// tensor table), padded with zeros to `len`.
fn gguf_bytes(n_tensors: u64, n_kv: u64, body: &[u8], len: usize) -> Vec<u8> {
    let mut f = Vec::from(&b"GGUF"[..]);
    f.extend(3u32.to_le_bytes());
    f.extend(n_tensors.to_le_bytes());
    f.extend(n_kv.to_le_bytes());
    f.extend(body);
    if f.len() < len {
        f.resize(len, 0);
    }
    f
}

/// One tensor entry: name `t`, the dimensions, type F32, `offset`.
fn tensor_entry(dims: &[u64], offset: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(1u64.to_le_bytes());
    b.push(b't');
    b.extend((dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend(d.to_le_bytes());
    }
    b.extend(0u32.to_le_bytes());
    b.extend(offset.to_le_bytes());
    b
}

/// Opens `bytes` as a file and returns the error (the open must fail, and
/// must not panic or abort, in debug and release builds alike).
fn open_err(name: &str, bytes: &[u8]) -> String {
    let dir =
        std::env::temp_dir().join(format!("lily-gguf-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.gguf"));
    std::fs::write(&path, bytes).unwrap();
    let r = std::panic::catch_unwind(|| Gguf::open(&path));
    std::fs::remove_file(&path).ok();
    match r {
        Ok(Ok(_)) => panic!("{name}: a malformed file was accepted"),
        Ok(Err(e)) => format!("{e:#}"),
        Err(_) => panic!("{name}: the reader panicked"),
    }
}

#[test]
fn malformed_tensor_tables_are_errors() {
    // The review's cases (128-byte files, one F32 tensor, data at 64).
    let cases: [(&str, Vec<u64>, u64, &str); 5] = [
        ("offset-wrap", vec![1], u64::MAX - 63, "overflow"),
        ("row-size-wrap", vec![1u64 << 62], 0, "overflow"),
        ("rows-wrap", vec![1, 1u64 << 63, 2], 0, "overflow"),
        ("unaligned", vec![1], 1, "multiple"),
        ("zero-dim", vec![0], 0, "zero dimension"),
    ];
    for (name, dims, offset, want) in cases {
        let e = open_err(name, &gguf_bytes(1, 0, &tensor_entry(&dims, offset), 128));
        assert!(e.contains(want), "{name}: {e}");
    }
    // A tensor past the end, and a duplicate name.
    let e = open_err("past-end", &gguf_bytes(1, 0, &tensor_entry(&[64], 0), 128));
    assert!(e.contains("past the file"), "{e}");
    let mut two = tensor_entry(&[1], 0);
    two.extend(tensor_entry(&[1], 32));
    let e = open_err("duplicate", &gguf_bytes(2, 0, &two, 256));
    assert!(e.contains("appears twice"), "{e}");
}

#[test]
fn malformed_metadata_is_an_error() {
    fn kv(key: &str, ty: u32, value: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend((key.len() as u64).to_le_bytes());
        b.extend(key.as_bytes());
        b.extend(ty.to_le_bytes());
        b.extend(value);
        b
    }
    // Truncated: a header that stops inside its counts, and one inside a key.
    let full = gguf_bytes(
        0,
        1,
        &kv("general.name", 8, &[3, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c']),
        0,
    );
    for cut in [10, 20, 30, full.len() - 1] {
        let e = open_err(&format!("truncated-{cut}"), &full[..cut]);
        assert!(
            e.contains("ends early")
                || e.contains("past the end")
                || e.contains("does not fit"),
            "{e}"
        );
    }
    // Impossible counts.
    let e = open_err("huge-tensor-count", &gguf_bytes(u64::MAX, 0, &[], 64));
    assert!(e.contains("count"), "{e}");
    let e = open_err("huge-kv-count", &gguf_bytes(0, u64::MAX / 2, &[], 64));
    assert!(e.contains("count"), "{e}");
    // A string longer than the file, an array longer than the file.
    let e = open_err(
        "huge-string",
        &gguf_bytes(0, 1, &kv("k", 8, &u64::MAX.to_le_bytes()), 64),
    );
    assert!(e.contains("past the end"), "{e}");
    let mut arr = 4u32.to_le_bytes().to_vec();
    arr.extend(((1u64 << 40) as u64).to_le_bytes());
    let e = open_err("huge-array", &gguf_bytes(0, 1, &kv("k", 9, &arr), 64));
    assert!(e.contains("does not fit"), "{e}");
    // Nested arrays, however small.
    let mut nested = 9u32.to_le_bytes().to_vec();
    nested.extend(1u64.to_le_bytes());
    nested.extend(9u32.to_le_bytes());
    nested.extend(1u64.to_le_bytes());
    nested.extend(4u32.to_le_bytes());
    nested.extend(1u64.to_le_bytes());
    nested.extend(7u32.to_le_bytes());
    let e = open_err("nested", &gguf_bytes(0, 1, &kv("k", 9, &nested), 128));
    assert!(e.contains("nested"), "{e}");
    // Unknown types, a bad alignment, a repeated key.
    let e = open_err("bad-type", &gguf_bytes(0, 1, &kv("k", 99, &[0; 8]), 64));
    assert!(e.contains("unknown"), "{e}");
    let e = open_err(
        "bad-alignment",
        &gguf_bytes(0, 1, &kv("general.alignment", 4, &3u32.to_le_bytes()), 64),
    );
    assert!(e.contains("alignment"), "{e}");
    let mut twice = kv("k", 4, &1u32.to_le_bytes());
    twice.extend(kv("k", 4, &2u32.to_le_bytes()));
    let e = open_err("repeated-key", &gguf_bytes(0, 2, &twice, 64));
    assert!(e.contains("twice"), "{e}");
    // A string that is not UTF-8.
    let e = open_err(
        "not-utf8",
        &gguf_bytes(0, 1, &kv("k", 8, &[2, 0, 0, 0, 0, 0, 0, 0, 0xc3, 0x28]), 64),
    );
    assert!(e.contains("UTF-8"), "{e}");
}

#[test]
fn a_header_past_the_memory_budget_is_refused() {
    // 2^28 one-byte array elements fit a (sparse) 256 MiB file but would
    // parse into several GiB of values.
    let mut b = gguf_bytes(0, 1, &[], 0);
    b.extend(1u64.to_le_bytes());
    b.push(b'k');
    b.extend(9u32.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend((1u64 << 28).to_le_bytes());
    let dir =
        std::env::temp_dir().join(format!("lily-gguf-budget-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("budget.gguf");
    std::fs::write(&path, &b).unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(b.len() as u64 + (1 << 28))
        .unwrap();
    let r = Gguf::open(&path);
    std::fs::remove_dir_all(&dir).ok();
    let e = format!("{:#}", r.err().expect("accepted"));
    assert!(e.contains("more memory"), "{e}");
}
