use crate::board::BoardState;
use crate::types::*;
use std::fs::File;
use std::io::{BufReader, Read as IoRead};

pub const PSQ_INPUTS_PER_BUCKET: usize = 768;
pub const NNUE_OUTPUT_BUCKETS: usize = 8;
pub const MAX_HIDDEN_SIZE: usize = 2048;

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;
const EVAL_SCALE: i32 = 400;
const FT_SHIFT: i32 = 9;
const NNUE_NUM_PIECE_TYPES: usize = 12;
const NNUE_MAGIC: u32 = 0x4E4E5545;
const COMPACT_NNUE_MAGIC: u32 = 0x314E4345;
const COMPACT_NNUE_VERSION_ROWS: u32 = 1;
const COMPACT_NNUE_VERSION_PACKED: u32 = 2;
const COMPACT_ZERO_ROW: u16 = u16::MAX;

pub fn convert(sq: u8) -> u8 {
    sq ^ 56
}

const CONSENSUS_BUCKETS: [[usize; 8]; 4] = [
    [0, 4, 8, 8, 12, 12, 14, 14],
    [1, 5, 9, 9, 12, 12, 14, 14],
    [2, 6, 10, 10, 13, 13, 15, 15],
    [3, 7, 11, 11, 13, 13, 15, 15],
];

#[derive(Copy, Clone, Debug)]
pub enum KbLayout {
    Uniform = 0,
    Consensus = 1,
    Reckless = 2,
}

impl KbLayout {
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(KbLayout::Uniform),
            1 => Some(KbLayout::Consensus),
            2 => Some(KbLayout::Reckless),
            _ => None,
        }
    }
}

pub fn compute_king_buckets(layout: KbLayout) -> ([usize; 64], [bool; 64]) {
    let mut kb = [0; 64];
    let mut km = [false; 64];

    for sq in 0..64 {
        let f = sq % 8;
        let r = sq / 8;
        let (mf, mirror) = if f >= 4 { (7 - f, true) } else { (f, false) };

        kb[sq] = match layout {
            KbLayout::Uniform => mf * 4 + r / 2,
            KbLayout::Consensus => CONSENSUS_BUCKETS[mf][r],
            KbLayout::Reckless => {
                let t = [
                    0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 6, 7, 7, 6, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                ];
                t[sq]
            }
        };
        km[sq] = mirror;
    }
    (kb, km)
}

fn halfka_idx(
    kb: &[usize; 64],
    km: &[bool; 64],
    persp: u8,
    ks: u8,
    pc: u8,
    pt: u8,
    ps: u8,
) -> usize {
    let mut k = ks as usize;
    let mut p = ps as usize;
    let mut pi = (pc as usize) * 6 + pt as usize;

    if persp == 1 {
        k ^= 56;
        p ^= 56;
        pi = if pi >= 6 { pi - 6 } else { pi + 6 };
    }
    if km[k] {
        p = (p & !7) | (7 - (p & 7));
    }
    kb[k] * (NNUE_NUM_PIECE_TYPES * 64) + pi * 64 + p
}

pub fn output_bucket(pc: u32) -> usize {
    let b = (pc as i32 - 2) / 4;
    b.clamp(0, 7) as usize
}

pub struct NNUENet {
    pub hidden_size: usize,
    pub input_weights: Vec<i16>,
    pub input_row_map: Vec<u16>,
    pub input_biases: Vec<i16>,
    pub output_weights: Vec<i16>,
    pub output_bias: [i32; NNUE_OUTPUT_BUCKETS],
    pub use_screlu: bool,
    pub use_pairwise: bool,
    pub l1_size: usize,
    pub l1_per_bucket: usize,
    pub bucketed_hidden: bool,
    pub l1_scale: i32,
    pub l2_size: usize,
    pub l2_per_bucket: usize,
    pub l1_weights: Vec<i16>,
    pub l1_biases: Vec<i16>,
    pub l2_weights_f: Vec<f32>,
    pub l2_biases_f: Vec<f32>,
    pub out_weights_f: Vec<f32>,
    pub out_bias_f: Vec<f32>,
    pub dual_l1: bool,
    pub num_king_buckets: usize,
    pub kb_layout: KbLayout,
    pub king_bucket: [usize; 64],
    pub king_mirror: [bool; 64],
}

struct VersionFlags {
    screlu: bool,
    pairwise: bool,
    l1s: usize,
    l2s: usize,
    l1sc: i32,
    bucketed: bool,
    dual: bool,
    nkb: usize,
    layout: KbLayout,
    ft: usize,
}

type FeatureWeights = (Vec<i16>, Vec<u16>, Vec<i16>);
type HiddenLayers = (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>);

struct CompactFeaturePayload {
    dense_len: usize,
    dense_header: Vec<u8>,
    input_weights: Vec<i16>,
    input_row_map: Vec<u16>,
    tail: Vec<u8>,
    virtual_rows: usize,
    hidden_size: usize,
}

impl VersionFlags {
    fn l1_scale_f32(&self) -> f32 {
        if self.l1sc != 0 {
            self.l1sc as f32
        } else {
            QA as f32
        }
    }
}

impl NNUENet {
    pub fn halfka(&self, persp: u8, ks: u8, pc: u8, pt: u8, ps: u8) -> usize {
        halfka_idx(&self.king_bucket, &self.king_mirror, persp, ks, pc, pt, ps)
    }

    pub fn input_row(&self, idx: usize) -> Option<&[i16]> {
        let physical_row = *self.input_row_map.get(idx)?;
        if physical_row == COMPACT_ZERO_ROW {
            return None;
        }

        let physical_row = physical_row as usize;
        let start = physical_row * self.hidden_size;
        Some(&self.input_weights[start..start + self.hidden_size])
    }

    #[inline(always)]
    fn input_row_fast(&self, idx: usize) -> Option<&[i16]> {
        debug_assert!(idx < self.input_row_map.len());
        let physical_row = unsafe { *self.input_row_map.get_unchecked(idx) };
        if physical_row == COMPACT_ZERO_ROW {
            return None;
        }

        let start = physical_row as usize * self.hidden_size;
        debug_assert!(start + self.hidden_size <= self.input_weights.len());
        let row = unsafe {
            std::slice::from_raw_parts(self.input_weights.as_ptr().add(start), self.hidden_size)
        };
        Some(row)
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let len = std::fs::metadata(path)
            .map_err(|e| format!("stat: {}", e))?
            .len();
        let f = File::open(path).map_err(|e| format!("open: {}", e))?;
        let mut r = BufReader::new(f);
        Self::load_reader(&mut r, len, path)
    }

    pub fn load_from_bytes(data: &[u8], name: &str) -> Result<Self, String> {
        let len = data.len() as u64;
        let mut r = std::io::Cursor::new(data);
        Self::load_reader(&mut r, len, name)
    }

    pub fn load_compact_from_bytes(data: &[u8], name: &str) -> Result<Self, String> {
        let compact = Self::read_compact_payload(data)?;
        Self::load_compact_payload(compact, name)
    }

    fn load_reader(r: &mut impl IoRead, data_len: u64, name: &str) -> Result<Self, String> {
        let ver = Self::read_header(r)?;
        let flags = Self::read_version_flags(r, ver)?;
        let hs = Self::infer_hidden_size(ver, &flags, data_len)?;
        if hs > MAX_HIDDEN_SIZE {
            return Err(format!("hs {} too large", hs));
        }

        let (iw, input_row_map, ib) = Self::read_feature_weights(r, hs, &flags)?;
        let (l1w, l1b, l2w_raw, l2b_raw) = Self::read_hidden_layers(r, hs, &flags)?;
        let (outw, outb) = Self::read_output_weights(r, hs, &flags)?;

        Self::finish_load(
            ver,
            name,
            hs,
            flags,
            iw,
            input_row_map,
            ib,
            l1w,
            l1b,
            l2w_raw,
            l2b_raw,
            outw,
            outb,
        )
    }

    fn load_compact_payload(compact: CompactFeaturePayload, name: &str) -> Result<Self, String> {
        let mut header = std::io::Cursor::new(compact.dense_header);
        let ver = Self::read_header(&mut header)?;
        let flags = Self::read_version_flags(&mut header, ver)?;
        let hs = Self::infer_hidden_size(ver, &flags, compact.dense_len as u64)?;
        if hs != compact.hidden_size {
            return Err(format!(
                "compact NNUE hidden size {} does not match dense header {}",
                compact.hidden_size, hs
            ));
        }
        let psq = flags.nkb * PSQ_INPUTS_PER_BUCKET;
        if psq != compact.virtual_rows {
            return Err(format!(
                "compact NNUE row count {} does not match dense header {}",
                compact.virtual_rows, psq
            ));
        }

        let mut tail = std::io::Cursor::new(compact.tail);
        let mut ib = vec![0i16; hs];
        read_i16s(&mut tail, &mut ib)?;
        let (l1w, l1b, l2w_raw, l2b_raw) = Self::read_hidden_layers(&mut tail, hs, &flags)?;
        let (outw, outb) = Self::read_output_weights(&mut tail, hs, &flags)?;

        let mut trailing = [0u8; 1];
        if tail.read(&mut trailing).map_err(|e| e.to_string())? != 0 {
            return Err("compact NNUE dense tail has trailing bytes".into());
        }

        Self::finish_load(
            ver,
            name,
            hs,
            flags,
            compact.input_weights,
            compact.input_row_map,
            ib,
            l1w,
            l1b,
            l2w_raw,
            l2b_raw,
            outw,
            outb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_load(
        ver: u32,
        name: &str,
        hs: usize,
        flags: VersionFlags,
        input_weights: Vec<i16>,
        input_row_map: Vec<u16>,
        input_biases: Vec<i16>,
        l1w: Vec<i16>,
        l1b: Vec<i16>,
        l2w_raw: Vec<i16>,
        l2b_raw: Vec<i16>,
        outw: Vec<i16>,
        outb: [i32; NNUE_OUTPUT_BUCKETS],
    ) -> Result<Self, String> {
        let (kbt, kmt) = compute_king_buckets(flags.layout);

        Self::print_load_info(ver, name, hs, &flags);

        let l2w_f = Self::convert_to_f32(&l2w_raw, flags.l1_scale_f32());
        let l2b_f = Self::convert_to_f32(&l2b_raw, flags.l1_scale_f32());
        let ow_f = Self::convert_to_f32(&outw, QB as f32);
        let ob_f: Vec<f32> = outb
            .iter()
            .map(|&b| b as f32 / (flags.l1_scale_f32() * QB as f32))
            .collect();

        let _l1t = Self::transpose_l1_weights(hs, &flags, &l1w);

        Ok(NNUENet {
            hidden_size: hs,
            input_weights,
            input_row_map,
            input_biases,
            output_weights: outw,
            output_bias: outb,
            use_screlu: flags.screlu,
            use_pairwise: flags.pairwise,
            l1_size: flags.l1s,
            l1_per_bucket: flags.l1s,
            bucketed_hidden: flags.bucketed,
            l1_scale: flags.l1sc,
            l2_size: flags.l2s,
            l2_per_bucket: flags.l2s,
            l1_weights: l1w,
            l1_biases: l1b,
            l2_weights_f: l2w_f,
            l2_biases_f: l2b_f,
            out_weights_f: ow_f,
            out_bias_f: ob_f,
            dual_l1: flags.dual,
            num_king_buckets: flags.nkb,
            kb_layout: flags.layout,
            king_bucket: kbt,
            king_mirror: kmt,
        })
    }

    fn read_compact_payload(data: &[u8]) -> Result<CompactFeaturePayload, String> {
        let mut r = std::io::Cursor::new(data);
        let magic = read_u32(&mut r)?;
        if magic != COMPACT_NNUE_MAGIC {
            return Err("bad compact NNUE magic".into());
        }
        let version = read_u32(&mut r)?;
        if !matches!(
            version,
            COMPACT_NNUE_VERSION_ROWS | COMPACT_NNUE_VERSION_PACKED
        ) {
            return Err(format!("unsupported compact NNUE v{}", version));
        }

        let dense_len = usize::try_from(read_u64(&mut r)?)
            .map_err(|_| "compact NNUE dense length too large".to_string())?;
        let header_len = read_u32(&mut r)? as usize;
        let virtual_rows = read_u32(&mut r)? as usize;
        let physical_rows = read_u32(&mut r)? as usize;
        let hidden_size = read_u32(&mut r)? as usize;

        if virtual_rows > COMPACT_ZERO_ROW as usize {
            return Err(format!(
                "compact NNUE virtual row count {} does not fit in u16",
                virtual_rows
            ));
        }
        if physical_rows > COMPACT_ZERO_ROW as usize {
            return Err(format!(
                "compact NNUE physical row count {} does not fit in u16",
                physical_rows
            ));
        }
        if hidden_size == 0 || hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!("compact NNUE hidden size {} invalid", hidden_size));
        }

        let row_bytes = hidden_size
            .checked_mul(2)
            .ok_or("compact NNUE row size overflow")?;
        let dense_feature_bytes = virtual_rows
            .checked_mul(row_bytes)
            .ok_or("compact NNUE feature size overflow")?;
        let dense_prefix_len = header_len
            .checked_add(dense_feature_bytes)
            .ok_or("compact NNUE dense prefix overflow")?;
        let tail_len = dense_len
            .checked_sub(dense_prefix_len)
            .ok_or("compact NNUE dense length is too small")?;

        let mut dense_header = vec![0u8; header_len];
        r.read_exact(&mut dense_header).map_err(|e| e.to_string())?;

        let mut input_row_map = vec![0u16; virtual_rows];
        for row in &mut input_row_map {
            *row = read_u16(&mut r)?;
            if *row != COMPACT_ZERO_ROW && *row as usize >= physical_rows {
                return Err(format!("compact NNUE row map points past row {}", *row));
            }
        }

        let input_weights = if version == COMPACT_NNUE_VERSION_ROWS {
            let compact_values = physical_rows
                .checked_mul(hidden_size)
                .ok_or("compact NNUE physical feature size overflow")?;
            let mut input_weights = vec![0i16; compact_values];
            read_i16s(&mut r, &mut input_weights)?;
            input_weights
        } else {
            Self::read_packed_feature_weights(&mut r, physical_rows, hidden_size)?
        };

        let mut tail = vec![0u8; tail_len];
        r.read_exact(&mut tail).map_err(|e| e.to_string())?;

        let mut trailing = [0u8; 1];
        if r.read(&mut trailing).map_err(|e| e.to_string())? != 0 {
            return Err("compact NNUE has trailing bytes".into());
        }

        Ok(CompactFeaturePayload {
            dense_len,
            dense_header,
            input_weights,
            input_row_map,
            tail,
            virtual_rows,
            hidden_size,
        })
    }

    fn read_packed_feature_weights(
        r: &mut impl IoRead,
        physical_rows: usize,
        hidden_size: usize,
    ) -> Result<Vec<i16>, String> {
        let compact_values = physical_rows
            .checked_mul(hidden_size)
            .ok_or("compact NNUE physical feature size overflow")?;

        let mut base_weights = vec![0u8; compact_values];
        r.read_exact(&mut base_weights).map_err(|e| e.to_string())?;

        let mut correction_offsets = vec![0u32; physical_rows + 1];
        for offset in &mut correction_offsets {
            *offset = read_u32(r)?;
        }
        validate_correction_offsets(&correction_offsets)?;

        let correction_count = *correction_offsets
            .last()
            .ok_or("compact NNUE correction offsets are empty")?
            as usize;
        let mut correction_indices = vec![0u16; correction_count];
        for index in &mut correction_indices {
            *index = read_u16(r)?;
            if *index as usize >= hidden_size {
                return Err(format!("compact NNUE correction index {} invalid", *index));
            }
        }

        let mut correction_deltas = vec![0i16; correction_count];
        read_i16s(r, &mut correction_deltas)?;

        let mut input_weights = Vec::with_capacity(compact_values);
        input_weights.extend(base_weights.into_iter().map(|weight| weight as i8 as i16));

        for row in 0..physical_rows {
            let row_start = row * hidden_size;
            let begin = correction_offsets[row] as usize;
            let end = correction_offsets[row + 1] as usize;
            for correction in begin..end {
                let lane = correction_indices[correction] as usize;
                let value = input_weights[row_start + lane]
                    .checked_add(correction_deltas[correction])
                    .ok_or("compact NNUE correction overflow")?;
                input_weights[row_start + lane] = value;
            }
        }

        Ok(input_weights)
    }

    fn read_header(r: &mut impl IoRead) -> Result<u32, String> {
        let magic = read_u32(r)?;
        if magic != NNUE_MAGIC {
            return Err("bad magic".into());
        }
        read_u32(r)
    }

    fn read_version_flags(r: &mut impl IoRead, ver: u32) -> Result<VersionFlags, String> {
        let mut flags = VersionFlags {
            screlu: false,
            pairwise: false,
            l1s: 0,
            l2s: 0,
            l1sc: QA,
            bucketed: false,
            dual: false,
            nkb: 16,
            layout: KbLayout::Uniform,
            ft: 0,
        };

        match ver {
            5 | 6 => {
                if ver == 6 {
                    let f = read_u8(r)?;
                    flags.screlu = f & 1 != 0;
                    flags.pairwise = f & 2 != 0;
                    if f & 32 != 0 {
                        flags.layout = KbLayout::Consensus;
                    }
                }
            }
            7..=10 => {
                let f = read_u8(r)?;
                flags.screlu = f & 1 != 0;
                flags.pairwise = f & 2 != 0;
                if f & 4 != 0 {
                    flags.l1sc = 64;
                }
                flags.bucketed = f & 8 != 0;
                flags.dual = f & 16 != 0;
                let ext = f & 128 != 0;
                let cons_inline = if !ext { f & 32 != 0 } else { false };

                flags.ft = read_u16(r)? as usize;
                flags.l1s = read_u16(r)? as usize;
                flags.l2s = read_u16(r)? as usize;

                if ext {
                    flags.nkb = read_u8(r)? as usize;
                    flags.layout = KbLayout::from_id(read_u8(r)?).ok_or("bad layout")?;
                } else if cons_inline {
                    flags.layout = KbLayout::Consensus;
                }

                if ver >= 10 {
                    let _ = read_u8(r)?;
                }
            }
            _ => return Err(format!("unsupported v{}", ver)),
        }
        Ok(flags)
    }

    fn infer_hidden_size(ver: u32, flags: &VersionFlags, data_len: u64) -> Result<usize, String> {
        match ver {
            5 => {
                let body = data_len - 8;
                let num = body - 32;
                let den = 2 * (12288 + 1 + 16);
                if !num.is_multiple_of(den) {
                    return Err("cannot infer h".into());
                }
                Ok((num / den) as usize)
            }
            6 => {
                let body = data_len - 9;
                let om: u64 = if flags.pairwise { 8 } else { 16 };
                let num = body - 32;
                let den = 2 * (12288 + 1 + om);
                if !num.is_multiple_of(den) {
                    return Err("cannot infer h".into());
                }
                Ok((num / den) as usize)
            }
            _ => Ok(flags.ft),
        }
    }

    fn read_feature_weights(
        r: &mut impl IoRead,
        hs: usize,
        flags: &VersionFlags,
    ) -> Result<FeatureWeights, String> {
        let psq = flags.nkb * PSQ_INPUTS_PER_BUCKET;
        let mut dense_weights = vec![0i16; psq * hs];
        read_i16s(r, &mut dense_weights)?;
        let (iw, input_row_map) = Self::compact_input_weights(dense_weights, hs, psq)?;
        let mut ib = vec![0i16; hs];
        read_i16s(r, &mut ib)?;
        Ok((iw, input_row_map, ib))
    }

    fn compact_input_weights(
        dense_weights: Vec<i16>,
        hs: usize,
        virtual_rows: usize,
    ) -> Result<(Vec<i16>, Vec<u16>), String> {
        if virtual_rows > COMPACT_ZERO_ROW as usize {
            return Err(format!(
                "NNUE virtual row count {} does not fit in u16",
                virtual_rows
            ));
        }
        if hs == 0 {
            return Err("NNUE hidden size must be nonzero".into());
        }
        if dense_weights.len() != virtual_rows * hs {
            return Err("NNUE feature matrix size mismatch".into());
        }

        let mut compact_weights = Vec::with_capacity(dense_weights.len());
        let mut input_row_map = Vec::with_capacity(virtual_rows);
        for row in dense_weights.chunks_exact(hs) {
            if row.iter().all(|&value| value == 0) {
                input_row_map.push(COMPACT_ZERO_ROW);
                continue;
            }

            let physical_row = compact_weights.len() / hs;
            if physical_row >= COMPACT_ZERO_ROW as usize {
                return Err(format!(
                    "NNUE physical row count {} does not fit in u16",
                    physical_row + 1
                ));
            }
            input_row_map.push(physical_row as u16);
            compact_weights.extend_from_slice(row);
        }

        Ok((compact_weights, input_row_map))
    }

    fn read_hidden_layers(
        r: &mut impl IoRead,
        hs: usize,
        flags: &VersionFlags,
    ) -> Result<HiddenLayers, String> {
        let bl1 = if flags.bucketed {
            NNUE_OUTPUT_BUCKETS * flags.l1s
        } else {
            flags.l1s
        };
        let bl2 = if flags.bucketed {
            NNUE_OUTPUT_BUCKETS * flags.l2s
        } else {
            flags.l2s
        };
        let mut l1w = Vec::new();
        let mut l1b = Vec::new();
        let mut l2w_raw = Vec::new();
        let mut l2b_raw = Vec::new();

        if flags.l1s > 0 {
            let li = if flags.pairwise { hs } else { 2 * hs };
            l1w = vec![0i16; li * bl1];
            read_i16s(r, &mut l1w)?;
            l1b = vec![0i16; bl1];
            read_i16s(r, &mut l1b)?;
        }
        if flags.l2s > 0 {
            let l2i = if flags.dual { flags.l1s * 2 } else { flags.l1s };
            l2w_raw = vec![0i16; l2i * bl2];
            read_i16s(r, &mut l2w_raw)?;
            l2b_raw = vec![0i16; bl2];
            read_i16s(r, &mut l2b_raw)?;
        }
        Ok((l1w, l1b, l2w_raw, l2b_raw))
    }

    fn read_output_weights(
        r: &mut impl IoRead,
        hs: usize,
        flags: &VersionFlags,
    ) -> Result<(Vec<i16>, [i32; NNUE_OUTPUT_BUCKETS]), String> {
        let ow = if flags.l2s > 0 {
            flags.l2s
        } else if flags.l1s > 0 {
            flags.l1s
        } else if flags.pairwise {
            hs
        } else {
            2 * hs
        };
        let mut outw = vec![0i16; NNUE_OUTPUT_BUCKETS * ow];
        read_i16s(r, &mut outw)?;
        let mut outb = [0i32; NNUE_OUTPUT_BUCKETS];
        for bias in &mut outb {
            *bias = read_i32(r)?;
        }
        Ok((outw, outb))
    }

    fn print_load_info(ver: u32, name: &str, hs: usize, flags: &VersionFlags) {
        let act = if flags.pairwise {
            "pairwise"
        } else if flags.screlu {
            "SCReLU"
        } else {
            "CReLU"
        };
        println!(
            "info string Loaded NNUE v{} {} {} (FT={} L1={} L2={})",
            ver, name, act, hs, flags.l1s, flags.l2s
        );
    }

    fn convert_to_f32(src: &[i16], divisor: f32) -> Vec<f32> {
        if src.is_empty() {
            Vec::new()
        } else {
            src.iter().map(|&v| v as f32 / divisor).collect()
        }
    }

    fn transpose_l1_weights(hs: usize, flags: &VersionFlags, l1w: &[i16]) -> Vec<i16> {
        if flags.l1s == 0 {
            return Vec::new();
        }
        let bl1 = if flags.bucketed {
            NNUE_OUTPUT_BUCKETS * flags.l1s
        } else {
            flags.l1s
        };
        let l1 = bl1;
        let pp = if flags.pairwise { hs / 2 } else { hs };
        let total = if flags.pairwise { hs } else { 2 * hs };
        let mut wt = vec![0i16; l1 * total];
        for i in 0..l1 {
            for j in 0..pp {
                wt[i * pp + j] = l1w[j * l1 + i];
            }
        }
        let no = l1 * pp;
        for i in 0..l1 {
            for j in 0..pp {
                wt[no + i * pp + j] = l1w[(pp + j) * l1 + i];
            }
        }
        wt
    }

    pub fn forward(&self, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        let bucket = output_bucket(piece_count);
        let out_w = self.output_weight_row(bucket);

        let (stm_acc, ntm_acc) = if stm == WHITE {
            (acc.white(), acc.black())
        } else {
            (acc.black(), acc.white())
        };

        if self.l1_size > 0 && self.use_pairwise {
            return self.forward_l1_pairwise(stm_acc, ntm_acc, bucket);
        }
        if self.use_pairwise {
            return self.forward_v6_pairwise(stm_acc, ntm_acc, bucket, out_w);
        }
        self.forward_base(stm_acc, ntm_acc, bucket, out_w)
    }

    fn output_weight_row(&self, bucket: usize) -> &[i16] {
        let w = if self.l2_per_bucket > 0 {
            self.l2_per_bucket
        } else if self.l1_per_bucket > 0 {
            self.l1_per_bucket
        } else if self.use_pairwise {
            self.hidden_size
        } else {
            2 * self.hidden_size
        };
        &self.output_weights[bucket * w..bucket * w + w]
    }

    #[inline(always)]
    fn crelu_i64(value: i16) -> i64 {
        if value <= 0 {
            0
        } else if value >= QA as i16 {
            QA as i64
        } else {
            value as i64
        }
    }

    fn forward_base(&self, stm: &[i16], ntm: &[i16], bucket: usize, out_w: &[i16]) -> i32 {
        let h = self.hidden_size;
        let mut output = self.output_bias[bucket] as i64;

        if self.use_screlu {
            for i in 0..h {
                let v = Self::crelu_i64(stm[i]);
                output += v * v * out_w[i] as i64;
            }
            for i in 0..h {
                let v = Self::crelu_i64(ntm[i]);
                output += v * v * out_w[h + i] as i64;
            }
            output /= QA as i64;
        } else {
            for i in 0..h {
                let v = Self::crelu_i64(stm[i]);
                output += v * out_w[i] as i64;
            }
            for i in 0..h {
                let v = Self::crelu_i64(ntm[i]);
                output += v * out_w[h + i] as i64;
            }
        }

        let mut result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
        if self.use_screlu {
            result = result * 4 / 5;
        }
        result
    }

    fn forward_v6_pairwise(&self, stm: &[i16], ntm: &[i16], bucket: usize, out_w: &[i16]) -> i32 {
        let pw = self.hidden_size / 2;
        let mut sum: i64 = 0;

        for i in 0..pw {
            let a = (stm[i] as i32).clamp(0, QA);
            let b = (stm[i + pw] as i32).clamp(0, QA);
            sum += (a * b) as i64 * out_w[i] as i64;
        }
        for i in 0..pw {
            let a = (ntm[i] as i32).clamp(0, QA);
            let b = (ntm[i + pw] as i32).clamp(0, QA);
            sum += (a * b) as i64 * out_w[pw + i] as i64;
        }

        let output = sum / QA as i64 + self.output_bias[bucket] as i64;
        (output * EVAL_SCALE as i64 / QAB as i64) as i32
    }

    fn forward_l1_pairwise(&self, stm: &[i16], ntm: &[i16], bucket: usize) -> i32 {
        let pw = self.hidden_size / 2;
        let l1_total = self.l1_size;
        let l1_pb = self.l1_per_bucket;
        let qa_l1 = self.l1_scale;

        let l1_off = if self.bucketed_hidden {
            bucket * l1_pb
        } else {
            0
        };
        let l1 = if self.bucketed_hidden {
            l1_pb
        } else {
            l1_total
        };

        let sp = Self::pairwise_pack(stm, pw);
        let np = Self::pairwise_pack(ntm, pw);

        let pw_scale = (QA * QA) >> FT_SHIFT;
        let hidden32 = self.l1_matmul(&sp, &np, l1_total, l1, l1_off, pw, pw_scale);
        let l1_out = Self::screlu_activation(&hidden32, pw_scale, qa_l1);

        if self.l2_per_bucket > 0 {
            self.forward_l2(l1_out, bucket, l1)
        } else {
            self.forward_l1_output(l1_out, bucket, l1)
        }
    }

    fn pairwise_pack(input: &[i16], pw: usize) -> Vec<u8> {
        let mut result = vec![0u8; pw];
        for i in 0..pw {
            let a = (input[i] as i32).clamp(0, QA);
            let b = (input[i + pw] as i32).clamp(0, QA);
            result[i] = ((a * b) >> FT_SHIFT) as u8;
        }
        result
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn l1_matmul(
        &self,
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
    ) -> Vec<i32> {
        let mut hidden32 = vec![0i32; l1];
        for i in 0..l1 {
            hidden32[i] = self.l1_biases[l1_off + i] as i32 * pw_scale;
        }
        for i in 0..l1 {
            let gi = l1_off + i;
            for j in 0..pw {
                hidden32[i] += sp[j] as i32 * self.l1_weights[j * l1_total + gi] as i32;
            }
            for j in 0..pw {
                hidden32[i] += np[j] as i32 * self.l1_weights[(pw + j) * l1_total + gi] as i32;
            }
        }
        hidden32
    }

    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32) -> Vec<f32> {
        let qf = qa_l1 as f32;
        let qsq = qf * qf;
        let mut out = vec![0.0f32; hidden.len()];
        for i in 0..hidden.len() {
            let v = (hidden[i] / pw_scale).clamp(0, qa_l1);
            out[i] = (v * v) as f32 / qsq;
        }
        out
    }

    fn forward_l2(&self, l1_out: Vec<f32>, bucket: usize, _l1: usize) -> i32 {
        let l2_pb = self.l2_per_bucket;
        let l2_total = self.l2_size;
        let l2_off = if self.bucketed_hidden {
            bucket * l2_pb
        } else {
            0
        };
        let l2 = if self.bucketed_hidden {
            l2_pb
        } else {
            l2_total
        };

        let mut h2 = vec![0.0f32; l2];
        h2.copy_from_slice(&self.l2_biases_f[l2_off..l2_off + l2]);
        for (i, &l1_value) in l1_out.iter().enumerate() {
            if l1_value == 0.0 {
                continue;
            }
            for (k, h2_value) in h2.iter_mut().enumerate().take(l2) {
                *h2_value += l1_value * self.l2_weights_f[i * l2_total + l2_off + k];
            }
        }
        for h2_value in h2.iter_mut().take(l2) {
            *h2_value = h2_value.clamp(0.0, 1.0);
            *h2_value *= *h2_value;
        }

        let ow = &self.out_weights_f[bucket * l2_pb..bucket * l2_pb + l2_pb];
        let mut of = self.out_bias_f[bucket];
        for k in 0..l2 {
            of += h2[k] * ow[k];
        }
        (of * EVAL_SCALE as f32) as i32
    }

    fn forward_l1_output(&self, l1_out: Vec<f32>, bucket: usize, l1: usize) -> i32 {
        let l1_pb = self.l1_per_bucket;
        let ow = &self.out_weights_f[bucket * l1_pb..bucket * l1_pb + l1_pb];
        let mut of = self.out_bias_f[bucket];
        for i in 0..l1 {
            of += l1_out[i] * ow[i];
        }
        (of * EVAL_SCALE as f32) as i32
    }
}

#[derive(Clone)]
pub struct NNUEAccumulator {
    white: Vec<i16>,
    black: Vec<i16>,
    pub hs: usize,
    pub wk: u8,
    pub bk: u8,
}

impl NNUEAccumulator {
    pub fn new(hs: usize) -> Self {
        NNUEAccumulator {
            white: vec![0i16; hs],
            black: vec![0i16; hs],
            hs,
            wk: 0,
            bk: 0,
        }
    }

    pub fn white(&self) -> &[i16] {
        &self.white
    }
    pub fn black(&self) -> &[i16] {
        &self.black
    }

    #[inline(always)]
    fn add_row(acc: &mut [i16], row: &[i16]) {
        let len = acc.len();
        debug_assert_eq!(len, row.len());
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
                *acc.get_unchecked_mut(i + 1) += *row.get_unchecked(i + 1);
                *acc.get_unchecked_mut(i + 2) += *row.get_unchecked(i + 2);
                *acc.get_unchecked_mut(i + 3) += *row.get_unchecked(i + 3);
            }
            i += 4;
        }
        while i < len {
            unsafe {
                *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
            }
            i += 1;
        }
    }

    #[inline(always)]
    fn remove_row(acc: &mut [i16], row: &[i16]) {
        let len = acc.len();
        debug_assert_eq!(len, row.len());
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                *acc.get_unchecked_mut(i) -= *row.get_unchecked(i);
                *acc.get_unchecked_mut(i + 1) -= *row.get_unchecked(i + 1);
                *acc.get_unchecked_mut(i + 2) -= *row.get_unchecked(i + 2);
                *acc.get_unchecked_mut(i + 3) -= *row.get_unchecked(i + 3);
            }
            i += 4;
        }
        while i < len {
            unsafe {
                *acc.get_unchecked_mut(i) -= *row.get_unchecked(i);
            }
            i += 1;
        }
    }

    #[inline(always)]
    fn add_feature(acc: &mut [i16], net: &NNUENet, idx: usize) {
        if let Some(row) = net.input_row_fast(idx) {
            Self::add_row(acc, row);
        }
    }

    #[inline(always)]
    fn remove_feature(acc: &mut [i16], net: &NNUENet, idx: usize) {
        if let Some(row) = net.input_row_fast(idx) {
            Self::remove_row(acc, row);
        }
    }

    pub fn refresh(&mut self, net: &NNUENet, st: &BoardState) {
        let h = self.hs;
        let wk = convert(st.king_sq(true) as u8);
        let bk = convert(st.king_sq(false) as u8);
        self.wk = wk;
        self.bk = bk;

        self.white.copy_from_slice(&net.input_biases[..h]);
        self.black.copy_from_slice(&net.input_biases[..h]);

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let mut bb = st.bb[(if color == 0 { 0 } else { 6 }) + pt as usize];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    let csq = convert(sq);

                    Self::add_feature(&mut self.white, net, net.halfka(0, wk, color, pt, csq));
                    Self::add_feature(&mut self.black, net, net.halfka(1, bk, color, pt, csq));
                }
            }
        }
    }

    fn add_piece(&mut self, net: &NNUENet, color: u8, pt: u8, sq: u8) {
        let csq = convert(sq);

        Self::add_feature(&mut self.white, net, net.halfka(0, self.wk, color, pt, csq));
        Self::add_feature(&mut self.black, net, net.halfka(1, self.bk, color, pt, csq));
    }

    fn remove_piece(&mut self, net: &NNUENet, color: u8, pt: u8, sq: u8) {
        let csq = convert(sq);

        Self::remove_feature(&mut self.white, net, net.halfka(0, self.wk, color, pt, csq));
        Self::remove_feature(&mut self.black, net, net.halfka(1, self.bk, color, pt, csq));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_move(
        &mut self,
        net: &NNUENet,
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        use crate::board::{is_white_piece, piece_on, piece_type, sq, EMPTY_SQ};

        let from = sq(sr, sc);
        let to = sq(er, ec);
        let mover_pi = piece_on(&st_before.bb, from);
        if mover_pi == EMPTY_SQ {
            return true;
        }

        let mover_type = piece_type(mover_pi);
        let white = is_white_piece(mover_pi);
        let color: u8 = if white { 0 } else { 1 };

        if mover_type == 5 {
            return false;
        }

        self.remove_piece(net, color, mover_type, from as u8);

        let cap_pi = piece_on(&st_before.bb, to);
        if cap_pi != EMPTY_SQ {
            let cap_color: u8 = if is_white_piece(cap_pi) { 0 } else { 1 };
            let cap_type = piece_type(cap_pi);
            self.remove_piece(net, cap_color, cap_type, to as u8);
        }

        if mover_type == 0 && Some(to) == st_before.ep && sc != ec {
            let cap_sq = if white { to + 8 } else { to - 8 };
            let ep_color: u8 = if white { 1 } else { 0 };
            self.remove_piece(net, ep_color, 0, cap_sq as u8);
        }

        if mover_type == 5 && sc == 4 && (ec == 6 || ec == 2) {
            if ec == 6 {
                self.remove_piece(net, color, 3, sq(sr, 7) as u8);
                self.add_piece(net, color, 3, sq(sr, 5) as u8);
            } else {
                self.remove_piece(net, color, 3, sq(sr, 0) as u8);
                self.add_piece(net, color, 3, sq(sr, 3) as u8);
            }
        }

        if mover_type == 0 && (er == 0 || er == 7) {
            let promo_type = match promotion.to_ascii_uppercase() {
                b'Q' => 4u8,
                b'R' => 3,
                b'B' => 2,
                b'N' => 1,
                _ => 4,
            };
            self.add_piece(net, color, promo_type, to as u8);
        } else {
            self.add_piece(net, color, mover_type, to as u8);
        }

        true
    }
}

fn read_u8(r: &mut impl IoRead) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b[0])
}

fn read_u16(r: &mut impl IoRead) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(r: &mut impl IoRead) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl IoRead) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

fn read_i32(r: &mut impl IoRead) -> Result<i32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(i32::from_le_bytes(b))
}

fn read_i16s(r: &mut impl IoRead, buf: &mut [i16]) -> Result<(), String> {
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 2) };
    r.read_exact(bytes).map_err(|e| format!("i16s: {}", e))?;
    Ok(())
}

fn validate_correction_offsets(offsets: &[u32]) -> Result<(), String> {
    let mut prev = 0;
    for (idx, &offset) in offsets.iter().enumerate() {
        if idx == 0 && offset != 0 {
            return Err("compact NNUE correction offsets must start at zero".into());
        }
        if offset < prev {
            return Err("compact NNUE correction offsets must be sorted".into());
        }
        prev = offset;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;

    fn nnue_score(net: &NNUENet, fen: &str) -> i32 {
        let mut engine = Engine::new();
        engine.set_fen(fen);

        let mut acc = NNUEAccumulator::new(net.hidden_size);
        acc.refresh(net, &engine.st);
        let stm = if engine.st.w { WHITE } else { BLACK };
        let piece_count = (0..12).map(|i| engine.st.bb[i].count_ones()).sum();
        let score = net.forward(&acc, stm, piece_count);
        if stm == WHITE {
            score
        } else {
            -score
        }
    }

    #[test]
    fn compact_embedded_nnue_matches_dense_scores() {
        let dense = NNUENet::load_from_bytes(include_bytes!("net.nnue"), "<dense test>")
            .expect("dense NNUE should load");
        let compact =
            NNUENet::load_compact_from_bytes(include_bytes!("net.compact.nnue"), "<compact test>")
                .expect("compact NNUE should load");

        assert!(
            include_bytes!("net.compact.nnue").len() + 3_000_000 < include_bytes!("net.nnue").len(),
            "compact embedded NNUE should remove the zero feature rows"
        );
        assert_eq!(dense.input_row_map, compact.input_row_map);
        assert_eq!(dense.input_weights, compact.input_weights);
        assert_eq!(
            compact
                .input_row_map
                .iter()
                .filter(|&&row| row == COMPACT_ZERO_ROW)
                .count(),
            1712
        );
        assert_eq!(compact.input_weights.len() / compact.hidden_size, 10576);

        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
            "8/8/8/R2pP1k/8/8/6Q1/4K3 w - d6 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 b - - 0 1",
            "8/P4k2/8/8/8/8/8/6K1 w - - 0 1",
        ] {
            assert_eq!(
                nnue_score(&dense, fen),
                nnue_score(&compact, fen),
                "compact NNUE score mismatch for {fen}"
            );
        }
    }
}
