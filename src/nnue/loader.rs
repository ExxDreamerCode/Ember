use super::other_nets::{load_other_net, OtherNetInfo};
use super::{
    compute_king_buckets, threat_feature_count, KbLayout, NNUENet, COMPACT_ZERO_ROW,
    MAX_HIDDEN_SIZE, NNUE_OUTPUT_BUCKETS, PSQ_INPUTS_PER_BUCKET, QA, QB,
};
use std::io::Read as IoRead;

const NNUE_MAGIC: u32 = 0x4E4E5545;
const COMPACT_NNUE_MAGIC: u32 = 0x314E4345;
const COMPACT_NNUE_VERSION_ROWS: u32 = 1;
const COMPACT_NNUE_VERSION_PACKED: u32 = 2;

pub(super) fn read_u8(r: &mut impl IoRead) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b[0])
}

pub(super) fn read_u16(r: &mut impl IoRead) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u16::from_le_bytes(b))
}

pub(super) fn read_u32(r: &mut impl IoRead) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

pub(super) fn read_u64(r: &mut impl IoRead) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

pub(super) fn read_i32(r: &mut impl IoRead) -> Result<i32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(i32::from_le_bytes(b))
}

pub(super) fn read_i16s(r: &mut impl IoRead, buf: &mut [i16]) -> Result<(), String> {
    let mut bytes = vec![0u8; std::mem::size_of_val(buf)];
    r.read_exact(&mut bytes)
        .map_err(|e| format!("i16s: {}", e))?;
    let (raw_values, []) = bytes.as_chunks::<2>() else {
        unreachable!("i16 byte buffer length is always even");
    };
    for (value, raw) in buf.iter_mut().zip(raw_values) {
        *value = i16::from_le_bytes([raw[0], raw[1]]);
    }
    Ok(())
}

pub(super) fn validate_correction_offsets(offsets: &[u32]) -> Result<(), String> {
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

struct VersionFlags {
    screlu: bool,
    pairwise: bool,
    l1s: usize,
    l2s: usize,
    l1sc: i32,
    bucketed: bool,
    dual: bool,
    hl_crelu: bool,
    nkb: usize,
    layout: KbLayout,
    ft: usize,
    has_threats: bool,
    num_threat_features: usize,
    xray_threats: bool,
}

type FeatureWeights = (Vec<i16>, Vec<u16>, Vec<i16>);
type ThreatWeights = Vec<i8>;
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
    fn max_abs_output_weight(outw: &[i16]) -> i64 {
        outw.iter()
            .map(|&weight| (weight as i64).abs())
            .max()
            .unwrap_or(0)
    }

    fn screlu_i32_output_safe(outw: &[i16]) -> bool {
        const I32_DOT_LANES: i64 = 8;
        let max_abs_weight = Self::max_abs_output_weight(outw);
        max_abs_weight * QA as i64 * QA as i64 * I32_DOT_LANES <= i32::MAX as i64
    }

    fn screlu_i32_accumulator_safe(hidden_size: usize, outw: &[i16]) -> bool {
        const X86_V3_VALUES_PER_CHUNK: usize = 16;
        const TERMS_PER_CHUNK_PER_LANE_FOR_BOTH_SIDES: i64 = 4;
        let chunks = (hidden_size / X86_V3_VALUES_PER_CHUNK) as i64;
        let max_abs_weight = Self::max_abs_output_weight(outw);
        max_abs_weight * QA as i64 * QA as i64 * chunks * TERMS_PER_CHUNK_PER_LANE_FOR_BOTH_SIDES
            <= i32::MAX as i64
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
        Self::load_from_bytes(&data, path)
    }

    pub fn load_from_bytes(data: &[u8], name: &str) -> Result<Self, String> {
        if OtherNetInfo::is_format(data) {
            let other = load_other_net(data)?;
            return Err(format!(
                "external-format net detected and decoded ({}) but cannot be returned as a native NNUENet ({}); load it through evaluate::init_nnue or the UCI NNUE option",
                other.overview, name
            ));
        }
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
        Self::validate_threat_header(&flags)?;

        let (iw, input_row_map, ib) = Self::read_feature_weights(r, hs, &flags)?;
        let threat_weights = Self::read_threat_weights(r, hs, &flags)?;
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
            threat_weights,
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
        Self::validate_threat_header(&flags)?;
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
        let threat_weights = Self::read_threat_weights(&mut tail, hs, &flags)?;
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
            threat_weights,
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
        threat_weights: ThreatWeights,
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
        let screlu_i32_output_safe = flags.screlu && Self::screlu_i32_output_safe(&outw);
        let screlu_i32_accumulator_safe =
            flags.screlu && Self::screlu_i32_accumulator_safe(hs, &outw);

        Ok(NNUENet {
            hidden_size: hs,
            input_weights,
            input_row_map,
            input_biases,
            threat_weights,
            num_threat_features: flags.num_threat_features,
            output_weights: outw,
            output_bias: outb,
            use_screlu: flags.screlu,
            screlu_i32_output_safe,
            screlu_i32_accumulator_safe,
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
            crelu_hidden: flags.hl_crelu,
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
            hl_crelu: false,
            nkb: 16,
            layout: KbLayout::Uniform,
            ft: 0,
            has_threats: false,
            num_threat_features: 0,
            xray_threats: false,
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
                flags.hl_crelu = ext && (f & 32 != 0);
                flags.has_threats = f & 64 != 0;

                flags.ft = read_u16(r)? as usize;
                flags.l1s = read_u16(r)? as usize;
                flags.l2s = read_u16(r)? as usize;

                if flags.has_threats {
                    flags.num_threat_features = read_u32(r)? as usize;
                }

                if ext {
                    flags.nkb = read_u8(r)? as usize;
                    flags.layout = KbLayout::from_id(read_u8(r)?).ok_or("bad layout")?;
                } else if cons_inline {
                    flags.layout = KbLayout::Consensus;
                }

                if ver >= 10 {
                    let training_flags = read_u8(r)?;
                    flags.xray_threats = flags.has_threats && training_flags & 1 != 0;
                    if training_flags & 2 != 0 {
                        return Err("unsupported NNUE output-bucket training layout".to_string());
                    }
                    if training_flags & !3 != 0 {
                        return Err(format!(
                            "unsupported NNUE v{} training flags 0x{:02x}",
                            ver, training_flags
                        ));
                    }
                } else if flags.has_threats {
                    flags.xray_threats = true;
                }
            }
            _ => return Err(format!("unsupported v{}", ver)),
        }
        Ok(flags)
    }

    fn validate_threat_header(flags: &VersionFlags) -> Result<(), String> {
        if !flags.has_threats {
            return Ok(());
        }
        if flags.xray_threats {
            return Err(
                "unsupported NNUE threat features: x-ray-trained threat nets are not supported"
                    .into(),
            );
        }
        if !flags.pairwise || flags.l1s == 0 {
            return Err(
                "unsupported NNUE threat features: only pairwise hidden-layer threat nets are supported"
                    .into(),
            );
        }
        let expected = threat_feature_count();
        if flags.num_threat_features != expected {
            return Err(format!(
                "unsupported NNUE threat features: expected {} features, file has {}",
                expected, flags.num_threat_features
            ));
        }
        Ok(())
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

    fn read_threat_weights(
        r: &mut impl IoRead,
        hs: usize,
        flags: &VersionFlags,
    ) -> Result<ThreatWeights, String> {
        if !flags.has_threats {
            return Ok(Vec::new());
        }
        Self::validate_threat_header(flags)?;

        let count = flags
            .num_threat_features
            .checked_mul(hs)
            .ok_or("NNUE threat matrix size overflow")?;
        let mut bytes = vec![0u8; count];
        r.read_exact(&mut bytes)
            .map_err(|e| format!("read NNUE threat weights: {}", e))?;
        Ok(bytes.into_iter().map(|byte| byte as i8).collect())
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
            "info string Loaded NNUE v{} {} {} (FT={} L1={} L2={} threats={})",
            ver, name, act, hs, flags.l1s, flags.l2s, flags.num_threat_features
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
}
