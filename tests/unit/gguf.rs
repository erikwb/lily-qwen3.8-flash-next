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
