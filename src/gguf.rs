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
        Ok(n / self.block_values() * self.block_bytes())
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
    /// ggml order: `dims[0]` is the row length (contiguous).
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    /// Values per row (`dims[0]`).
    pub fn row_len(&self) -> usize {
        self.dims[0] as usize
    }

    /// Rows: the product of the other dimensions.
    pub fn rows(&self) -> usize {
        self.dims[1..].iter().product::<u64>() as usize
    }

    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.row_len()).expect("validated at open")
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

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let n = read_u64(r)?;
    ensure!(n < 1 << 30, "GGUF string of {n} bytes");
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn read_value(r: &mut impl Read, ty: u32) -> Result<Value> {
    let mut b8 = [0u8; 8];
    let mut fixed = |n: usize, r: &mut dyn Read| -> Result<[u8; 8]> {
        b8 = [0; 8];
        r.read_exact(&mut b8[..n])?;
        Ok(b8)
    };
    Ok(match ty {
        0 => Value::U8(fixed(1, r)?[0]),
        1 => Value::I8(fixed(1, r)?[0] as i8),
        2 => Value::U16(u16::from_le_bytes(fixed(2, r)?[..2].try_into()?)),
        3 => Value::I16(i16::from_le_bytes(fixed(2, r)?[..2].try_into()?)),
        4 => Value::U32(u32::from_le_bytes(fixed(4, r)?[..4].try_into()?)),
        5 => Value::I32(i32::from_le_bytes(fixed(4, r)?[..4].try_into()?)),
        6 => Value::F32(f32::from_le_bytes(fixed(4, r)?[..4].try_into()?)),
        7 => Value::Bool(fixed(1, r)?[0] != 0),
        8 => Value::Str(read_string(r)?),
        9 => {
            let elem = read_u32(r)?;
            let n = read_u64(r)?;
            ensure!(n < 1 << 28, "GGUF array of {n} elements");
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                v.push(read_value(r, elem)?);
            }
            Value::Array(v)
        }
        10 => Value::U64(u64::from_le_bytes(fixed(8, r)?)),
        11 => Value::I64(i64::from_le_bytes(fixed(8, r)?)),
        12 => Value::F64(f64::from_le_bytes(fixed(8, r)?)),
        other => bail!("unknown GGUF metadata type {other}"),
    })
}

impl Gguf {
    /// Reads the header of `path` (the tensor data may still be arriving:
    /// only the header and the file's length are checked).
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 20, file);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        ensure!(&magic == b"GGUF", "{} is not a GGUF file", path.display());
        let version = read_u32(&mut r)?;
        ensure!(version == 3, "GGUF version {version}; lily reads version 3");
        let n_tensors = read_u64(&mut r)?;
        let n_kv = read_u64(&mut r)?;
        let mut metadata = BTreeMap::new();
        let mut pos = 24u64;
        let mut counted = CountingReader { inner: &mut r, n: 0 };
        for _ in 0..n_kv {
            let key = read_string(&mut counted)?;
            let ty = read_u32(&mut counted)?;
            let value = read_value(&mut counted, ty)
                .with_context(|| format!("reading metadata {key}"))?;
            metadata.insert(key, value);
        }
        let mut raw = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(&mut counted)?;
            let nd = read_u32(&mut counted)?;
            ensure!((1..=4).contains(&nd), "tensor {name} has {nd} dimensions");
            let mut dims = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                dims.push(read_u64(&mut counted)?);
            }
            let ty = read_u32(&mut counted)?;
            let offset = read_u64(&mut counted)?;
            raw.push((name, dims, ty, offset));
        }
        pos += counted.n;
        let alignment =
            metadata.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
        ensure!(alignment.is_power_of_two(), "GGUF alignment {alignment}");
        let data = pos.next_multiple_of(alignment);
        let mut tensors = Vec::with_capacity(raw.len());
        let mut index = BTreeMap::new();
        for (name, dims, ty_id, offset) in raw {
            let ty = GgmlType::from_id(ty_id).with_context(|| {
                format!("tensor {name} has unsupported ggml type {ty_id}")
            })?;
            let row = dims[0] as usize;
            let rows: u64 = dims[1..].iter().product();
            let bytes = ty.row_bytes(row).with_context(|| format!("tensor {name}"))?
                as u64
                * rows;
            let offset = data + offset;
            ensure!(
                offset + bytes <= file_len,
                "tensor {name} ends at {} past the file's {file_len} bytes",
                offset + bytes
            );
            index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo { name, dims, ty, offset, bytes });
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

    /// Reads `len` bytes at absolute offset `offset`.
    pub fn read_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
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
            rows.end <= t.rows(),
            "rows {rows:?} of {} ({} rows)",
            t.name,
            t.rows()
        );
        let rb = t.row_bytes();
        let raw =
            self.read_bytes(t.offset + (rows.start * rb) as u64, rows.len() * rb)?;
        let mut out = vec![0f32; rows.len() * t.row_len()];
        dequantize(t.ty, &raw, &mut out)?;
        Ok(out)
    }
}

/// A reader that counts the bytes it hands out (the header's length).
struct CountingReader<'r, R: Read> {
    inner: &'r mut R,
    n: u64,
}

impl<R: Read> Read for CountingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.n += n as u64;
        Ok(n)
    }
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
