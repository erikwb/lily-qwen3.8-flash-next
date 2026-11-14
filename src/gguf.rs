//! GGUF files: the header (metadata and tensor table) and a CPU decoder for
//! the block formats GLM 5.3 Flash's published Q2 file uses (F32, F16, BF16,
//! Q8_0, Q4_K, Q2_K, IQ2_XXS).
//!
//! The block layouts and the value formulas follow ggml as antirez's ds4
//! (MIT) implements them (`ds4.c`: `q2_k_value_f32`, `ds4_vec_dot_q4_K_f32`,
//! `ds4_vec_dot_iq2_xxs_f32`, the IQ2 grid and sign tables). Every decoded
//! value is a product of small exact factors, so the decode is exact and the
//! GPU kernels must reproduce it bit for bit (`tests/unit/gguf.rs` pins it to
//! values ds4's code computed).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use half::{bf16, f16};

/// The ggml block types lily decodes. The discriminants are the GGUF ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q2K = 10,
    Q4K = 12,
    Iq2Xxs = 16,
    Bf16 = 30,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            12 => Self::Q4K,
            16 => Self::Iq2Xxs,
            30 => Self::Bf16,
            _ => return None,
        })
    }

    /// Values per block.
    pub fn block_values(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q8_0 => 32,
            Self::Q2K | Self::Q4K | Self::Iq2Xxs => QK_K,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q4K => 144,
            Self::Iq2Xxs => 66,
        }
    }

    /// Bytes of a row of `n` values (`n` a multiple of the block).
    pub fn row_bytes(self, n: usize) -> Result<usize> {
        ensure!(
            n.is_multiple_of(self.block_values()),
            "{self:?} row of {n} values is not whole blocks of {}",
            self.block_values()
        );
        (n / self.block_values())
            .checked_mul(self.block_bytes())
            .with_context(|| format!("{self:?} row of {n} values overflows"))
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q4K => "Q4_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Bf16 => "BF16",
        }
    }
}

/// Values per K-quant super-block.
pub const QK_K: usize = 256;

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    /// An unsigned integer of any width.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor of the table.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the row length (contiguous). Every
    /// dimension is at least 1.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset in the file, a multiple of the file's alignment.
    pub offset: u64,
    pub bytes: u64,
    row_len: usize,
    rows: usize,
    row_bytes: usize,
}

impl TensorInfo {
    /// Values per row (`dims[0]`).
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Rows: the product of the other dimensions.
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
}

/// Whether `path` names a GGUF file (lily takes one as `--model`).
pub fn is_gguf(path: &Path) -> bool {
    path.is_file() && path.extension().is_some_and(|e| e == "gguf")
}

/// A parsed GGUF header. The tensor data stays in the file.
pub struct Gguf {
    pub path: PathBuf,
    pub file_len: u64,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    index: BTreeMap<String, usize>,
}

/// Memory the parsed header may take: strings, arrays, the metadata map,
/// the tensor table and its index. Each is charged before it is reserved,
/// with a conservative estimate of the map nodes and other overhead
/// ([`ENTRY_OVERHEAD`]); a header that would need more is refused. It is an
/// estimate of the parsed header's size, not a bound on allocator use.
const HEADER_BUDGET: u64 = 2 << 30;

/// Charged per string, map entry and table row beyond its own bytes (the
/// allocation and B-tree node share; generous, so the budget errs low).
const ENTRY_OVERHEAD: u64 = 64;

/// Largest `general.alignment` accepted.
const MAX_ALIGNMENT: u64 = 1 << 16;

/// The header being read: every count and length is checked against the
/// bytes left in the file and charged against [`HEADER_BUDGET`] before
/// memory is reserved for it, and the large reservations are fallible, so a
/// corrupt or truncated file is an error rather than an unbounded
/// allocation.
struct HeaderReader {
    r: BufReader<File>,
    pos: u64,
    len: u64,
    budget: u64,
}

impl HeaderReader {
    fn remaining(&self) -> u64 {
        self.len - self.pos
    }

    fn charge(&mut self, bytes: u64) -> Result<()> {
        self.budget = self
            .budget
            .checked_sub(bytes)
            .context("the GGUF header needs more memory than lily allows (2 GiB)")?;
        Ok(())
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        ensure!(N as u64 <= self.remaining(), "the GGUF header ends early");
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    /// `n` items of at least `min_bytes` each still fit in the file.
    fn count(&self, n: u64, min_bytes: u64, what: &str) -> Result<usize> {
        let need = n.checked_mul(min_bytes);
        ensure!(
            need.is_some_and(|b| b <= self.remaining()),
            "GGUF {what} count {n} does not fit the {} bytes left",
            self.remaining()
        );
        Ok(usize::try_from(n)?)
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        ensure!(
            n <= self.remaining(),
            "GGUF string of {n} bytes past the end of the file"
        );
        self.charge(n.saturating_add(ENTRY_OVERHEAD))?;
        let n = usize::try_from(n)?;
        let mut b = Vec::new();
        b.try_reserve_exact(n)?;
        b.resize(n, 0);
        self.r.read_exact(&mut b)?;
        self.pos += n as u64;
        String::from_utf8(b).context("GGUF string is not UTF-8")
    }

    /// One metadata value of type `ty`; `nested` is set inside an array,
    /// where an array is refused (GGUF files lily reads have none).
    fn value(&mut self, ty: u32, nested: bool) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.fixed::<1>()?[0]),
            1 => Value::I8(self.fixed::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.fixed()?)),
            3 => Value::I16(i16::from_le_bytes(self.fixed()?)),
            4 => Value::U32(u32::from_le_bytes(self.fixed()?)),
            5 => Value::I32(i32::from_le_bytes(self.fixed()?)),
            6 => Value::F32(f32::from_le_bytes(self.fixed()?)),
            7 => Value::Bool(self.fixed::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                ensure!(!nested, "nested GGUF arrays are not supported");
                let elem = self.u32()?;
                let min = match elem {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8 | 10..=12 => 8,
                    9 => bail!("nested GGUF arrays are not supported"),
                    other => bail!("unknown GGUF array element type {other}"),
                };
                let n = self.u64()?;
                let n = self.count(n, min, "array")?;
                self.charge((n as u64).saturating_mul(size_of::<Value>() as u64))?;
                let mut v = Vec::new();
                v.try_reserve_exact(n)?;
                for _ in 0..n {
                    v.push(self.value(elem, true)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(u64::from_le_bytes(self.fixed()?)),
            11 => Value::I64(i64::from_le_bytes(self.fixed()?)),
            12 => Value::F64(f64::from_le_bytes(self.fixed()?)),
            other => bail!("unknown GGUF metadata type {other}"),
        })
    }
}

impl Gguf {
    /// Reads the header of `path` (the tensor data may still be arriving:
    /// only the header and the file's length are checked). Every size is
    /// computed with checked arithmetic; dimensions must be nonzero, tensor
    /// names unique, offsets multiples of the alignment, and every tensor
    /// inside the file.
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut h = HeaderReader {
            r: BufReader::with_capacity(1 << 20, file),
            pos: 0,
            len: file_len,
            budget: HEADER_BUDGET,
        };
        let magic = h
            .fixed::<4>()
            .with_context(|| format!("{} is too short", path.display()))?;
        ensure!(&magic == b"GGUF", "{} is not a GGUF file", path.display());
        let version = h.u32()?;
        ensure!(version == 3, "GGUF version {version}; lily reads version 3");
        let n_tensors = h.u64()?;
        let n_kv = h.u64()?;
        // A key/value pair takes at least 13 bytes (key length, type, one
        // byte of value), a tensor entry 32 (name length, rank, one
        // dimension, type, offset).
        let n_kv = h.count(n_kv, 13, "metadata")?;
        let n_tensors = h.count(n_tensors, 32, "tensor")?;
        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            h.charge(size_of::<(String, Value)>() as u64 + ENTRY_OVERHEAD)?;
            let key = h.string()?;
            let ty = h.u32()?;
            let value = h
                .value(ty, false)
                .with_context(|| format!("reading metadata {key}"))?;
            ensure!(
                !metadata.contains_key(&key),
                "GGUF metadata key {key} appears twice"
            );
            metadata.insert(key, value);
        }
        // The raw table, then the parsed one and its index (a second copy
        // of each name) beside it.
        let row = size_of::<(String, Vec<u64>, u32, u64)>()
            + size_of::<TensorInfo>()
            + size_of::<(String, usize)>();
        h.charge((n_tensors as u64).saturating_mul(row as u64 + 3 * ENTRY_OVERHEAD))?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(n_tensors)?;
        for _ in 0..n_tensors {
            let name = h.string()?;
            let nd = h.u32()?;
            ensure!((1..=4).contains(&nd), "tensor {name} has {nd} dimensions");
            // The dimensions, twice, and the index's copy of the name.
            h.charge(16 * u64::from(nd) + name.len() as u64)?;
            let mut dims = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                let d = h.u64()?;
                ensure!(d >= 1, "tensor {name} has a zero dimension");
                dims.push(d);
            }
            let ty = h.u32()?;
            let offset = h.u64()?;
            raw.push((name, dims, ty, offset));
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(v) => v.as_u64().context("general.alignment is not an integer")?,
        };
        ensure!(
            alignment.is_power_of_two() && alignment <= MAX_ALIGNMENT,
            "GGUF alignment {alignment}"
        );
        let data = h
            .pos
            .checked_next_multiple_of(alignment)
            .context("GGUF data offset overflows")?;
        let mut tensors = Vec::new();
        tensors.try_reserve_exact(raw.len())?;
        let mut index = BTreeMap::new();
        for (name, dims, ty_id, offset) in raw {
            let info =
                tensor_info(&name, &dims, ty_id, offset, data, alignment, file_len)
                    .with_context(|| format!("tensor {name}"))?;
            ensure!(
                index.insert(name.clone(), tensors.len()).is_none(),
                "tensor {name} appears twice"
            );
            tensors.push(info);
        }
        Ok(Self { path: path.to_path_buf(), file_len, metadata, tensors, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn require(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor(name)
            .with_context(|| format!("{} has no tensor {name}", self.path.display()))
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("GGUF metadata {key} missing or not an integer"))
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        self.get(key)
            .and_then(Value::as_f32)
            .with_context(|| format!("GGUF metadata {key} missing or not a float"))
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        self.get(key)
            .and_then(Value::as_str)
            .with_context(|| format!("GGUF metadata {key} missing or not a string"))
    }

    /// The architecture name (`general.architecture`).
    pub fn architecture(&self) -> Result<&str> {
        self.str("general.architecture")
    }

    /// Reads `len` bytes at absolute offset `offset` (inside the file).
    pub fn read_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let end = offset.checked_add(len as u64);
        ensure!(
            end.is_some_and(|e| e <= self.file_len),
            "{len} bytes at {offset} past the end of {}",
            self.path.display()
        );
        let file = File::open(&self.path)?;
        let mut b = vec![0u8; len];
        file.read_exact_at(&mut b, offset)?;
        Ok(b)
    }

    /// Decodes rows `rows` of tensor `t` to f32 (`rows.len() * row_len`).
    pub fn read_rows_f32(
        &self,
        t: &TensorInfo,
        rows: std::ops::Range<usize>,
    ) -> Result<Vec<f32>> {
        ensure!(
            rows.start <= rows.end && rows.end <= t.rows(),
            "rows {rows:?} of {} ({} rows)",
            t.name,
            t.rows()
        );
        let rb = t.row_bytes();
        // In range: rows.end * rb <= t.bytes, checked at open.
        let raw =
            self.read_bytes(t.offset + (rows.start * rb) as u64, rows.len() * rb)?;
        let mut out = vec![0f32; rows.len() * t.row_len()];
        dequantize(t.ty, &raw, &mut out)?;
        Ok(out)
    }
}

/// A tensor entry checked: sizes without overflow, the offset a multiple
/// of the alignment, the data inside the file.
fn tensor_info(
    name: &str,
    dims: &[u64],
    ty_id: u32,
    offset: u64,
    data: u64,
    alignment: u64,
    file_len: u64,
) -> Result<TensorInfo> {
    let ty = GgmlType::from_id(ty_id)
        .with_context(|| format!("unsupported ggml type {ty_id}"))?;
    let row_len = usize::try_from(dims[0])?;
    let rows = dims[1..]
        .iter()
        .try_fold(1u64, |a, &d| a.checked_mul(d))
        .context("dimensions overflow")?;
    let rows = usize::try_from(rows)?;
    row_len.checked_mul(rows).context("element count overflows")?;
    let row_bytes = ty.row_bytes(row_len)?;
    let bytes = row_bytes.checked_mul(rows).context("size overflows")?;
    let bytes = u64::try_from(bytes)?;
    ensure!(
        offset.is_multiple_of(alignment),
        "offset {offset} is not a multiple of {alignment}"
    );
    let start = data.checked_add(offset).context("offset overflows")?;
    let end = start.checked_add(bytes).context("end overflows")?;
    ensure!(end <= file_len, "ends at {end} past the file's {file_len} bytes");
    Ok(TensorInfo {
        name: name.to_string(),
        dims: dims.to_vec(),
        ty,
        offset: start,
        bytes,
        row_len,
        rows,
        row_bytes,
    })
}

fn le_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn f16_at(b: &[u8], at: usize) -> f32 {
    f16::from_bits(le_u16(b, at)).to_f32()
}

/// Decodes whole blocks of `ty` from `raw` into `out` (one value per
/// element; `out.len()` must be the blocks' value count).
pub fn dequantize(ty: GgmlType, raw: &[u8], out: &mut [f32]) -> Result<()> {
    let bv = ty.block_values();
    let bb = ty.block_bytes();
    ensure!(
        out.len().is_multiple_of(bv) && raw.len() == out.len() / bv * bb,
        "{} bytes of {ty:?} do not decode to {} values",
        raw.len(),
        out.len()
    );
    for (blk, dst) in raw.chunks_exact(bb).zip(out.chunks_exact_mut(bv)) {
        match ty {
            GgmlType::F32 => dst[0] = f32::from_le_bytes(blk.try_into().unwrap()),
            GgmlType::F16 => dst[0] = f16_at(blk, 0),
            GgmlType::Bf16 => dst[0] = bf16::from_bits(le_u16(blk, 0)).to_f32(),
            GgmlType::Q8_0 => dequant_q8_0(blk, dst),
            GgmlType::Q4K => dequant_q4_k(blk, dst),
            GgmlType::Q2K => dequant_q2_k(blk, dst),
            GgmlType::Iq2Xxs => dequant_iq2_xxs(blk, dst),
        }
    }
    Ok(())
}

/// Q8_0: f16 scale, 32 signed bytes.
fn dequant_q8_0(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for (i, o) in out.iter_mut().enumerate() {
        *o = d * f32::from(b[2 + i] as i8);
    }
}

/// The 6-bit scale and min of sub-block `j` of a Q4_K block's 12 scale bytes.
pub fn q4_k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of
/// nibbles; sub-block j of 32 takes the low (even j) or high nibbles of
/// bytes `32 * (j / 2)..`.
fn dequant_q4_k(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    for j in 0..8 {
        let (sc, m) = q4_k_scale_min(j, scales);
        let scale = d * f32::from(sc);
        let minv = dmin * f32::from(m);
        let off = (j >> 1) * 32;
        let shift = (j & 1) * 4;
        for l in 0..32 {
            let q = (qs[off + l] >> shift) & 0xf;
            out[j * 32 + l] = scale * f32::from(q) - minv;
        }
    }
}

/// Q2_K: 16 scale bytes (low nibble scale, high nibble min), 64 bytes of
/// 2-bit codes, f16 d, f16 dmin. Group g of 16 values reads bytes
/// `32 * (g / 8) + 16 * (g & 1)..` at bit `2 * ((g / 2) & 3)`.
fn dequant_q2_k(b: &[u8], out: &mut [f32]) {
    let scales = &b[0..16];
    let qs = &b[16..80];
    let d = f16_at(b, 80);
    let dmin = f16_at(b, 82);
    for g in 0..16 {
        let base = 32 * (g / 8) + 16 * (g & 1);
        let shift = ((g / 2) & 3) * 2;
        let sc = scales[g];
        let scale = d * f32::from(sc & 0xf);
        let minv = dmin * f32::from(sc >> 4);
        for l in 0..16 {
            let q = (qs[base + l] >> shift) & 3;
            out[g * 16 + l] = scale * f32::from(q) - minv;
        }
    }
}

/// IQ2_XXS: f16 d, then per 32 values two u32 words: four grid indices
/// (bytes of the first), four 7-bit sign-pattern indices and a 4-bit scale
/// (the second). Value = `0.125 * d * (2 * s + 1) * (+/-)grid`.
fn dequant_iq2_xxs(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let w = &b[2 + 8 * ib..2 + 8 * ib + 8];
        let aux0 = u32::from_le_bytes(w[0..4].try_into().unwrap());
        let aux1 = u32::from_le_bytes(w[4..8].try_into().unwrap());
        let scale = 0.125 * d * ((2 * (aux1 >> 28) + 1) as f32);
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((aux0 >> (8 * l)) & 0xff) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                let v = i32::from(grid[j]);
                let v = if signs & (1 << j) != 0 { -v } else { v };
                out[ib * 32 + l * 8 + j] = scale * v as f32;
            }
        }
    }
}

/// ggml's IQ2 sign patterns: entry s is the 7 bits of s with an eighth
/// parity bit that makes the count of set bits even.
pub const KSIGNS_IQ2XS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut s = 0;
    while s < 128 {
        let parity = (s as u8).count_ones() & 1;
        t[s] = s as u8 | ((parity as u8) << 7);
        s += 1;
    }
    t
};

/// ggml's IQ2_XXS grid: 256 vectors of eight magnitudes from {8, 25, 43},
/// one byte each, little-endian in the u64.
pub const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

#[cfg(test)]
#[path = "../tests/unit/gguf.rs"]
mod tests;
