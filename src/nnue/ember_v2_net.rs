pub(crate) const EMBER_V2_VERSION: u32 = 0x6A448AFA;

pub(crate) const LEB128_MAGIC_LEN: usize = 17;

const LEB128_MAGIC: [u8; 17] = *b"COMPRESSED_LEB128";

const STACK_BYTES: usize = 35208;
const STACK_HASH: u32 = 0x6333_7116;
const PSQT_BUCKETS: usize = 8;
const FT_HEADER_HASH: u32 = 0x6165_ddc9;
const ARCH_HASH: u32 = 0x0256_acdf;
const HIDDEN_SIZE: usize = 1024;
const PSQ_DIMS: usize = 22_528;
const THREAT_DIMS: usize = 60_720;
const NUM_STACKS: usize = 8;

#[derive(Clone, Copy)]
pub struct TensorDesc {
    pub leb: bool,
    pub offset: usize,
    pub byte_count: usize,
}

pub struct EmberV2Info {
    pub version: u32,
    pub hash: u32,
    pub description: Vec<u8>,
    pub body_start: usize,
    pub tensors: Vec<TensorDesc>,
}

#[allow(dead_code)]
pub struct EmberV2Stack {
    pub fc0_bias: Vec<i32>,
    pub fc0_weights: Vec<i8>,
    pub fc1_bias: Vec<i32>,
    pub fc1_weights: Vec<i8>,
    pub fc2_bias: i32,
    pub fc2_weights: Vec<i8>,
}

#[allow(dead_code)]
pub struct EmberV2Data {
    pub hidden_size: usize,
    pub psq_dims: usize,
    pub threat_dims: usize,
    pub num_stacks: usize,
    pub ft_bias: Vec<i16>,
    pub threat_weights: Vec<i8>,
    pub threat_psqt: Vec<i32>,
    pub psq_weights: Vec<i16>,
    pub psqt: Vec<i32>,
    pub stacks: Vec<EmberV2Stack>,
    pub overview: String,
}

fn le_u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

fn magic_matches_at(data: &[u8], at: usize) -> bool {
    let Some(end) = at.checked_add(LEB128_MAGIC_LEN) else {
        return false;
    };
    data.get(at..end) == Some(LEB128_MAGIC.as_slice())
}

fn find_magic_at_or_after(data: &[u8], start: usize) -> Option<usize> {
    if start + LEB128_MAGIC_LEN > data.len() {
        return None;
    }
    let limit = data.len() - LEB128_MAGIC_LEN;
    (start..=limit).find(|&at| magic_matches_at(data, at))
}

fn scan_tensors(data: &[u8], start: usize) -> Vec<TensorDesc> {
    let mut out = Vec::new();
    let mut pos = start;
    while pos < data.len() {
        if magic_matches_at(data, pos) {
            let count_field = pos + LEB128_MAGIC_LEN;
            if count_field + 4 > data.len() {
                break;
            }
            let bc = le_u32_at(data, count_field) as usize;
            let data_pos = count_field + 4;
            if data_pos + bc > data.len() {
                break;
            }
            out.push(TensorDesc {
                leb: true,
                offset: data_pos,
                byte_count: bc,
            });
            pos = data_pos + bc;
        } else {
            let next = find_magic_at_or_after(data, pos);
            let end = next.unwrap_or(data.len());
            out.push(TensorDesc {
                leb: false,
                offset: pos,
                byte_count: end - pos,
            });
            pos = end;
        }
    }
    out
}

impl EmberV2Info {
    pub fn is_format(data: &[u8]) -> bool {
        data.len() >= 4 && le_u32_at(data, 0) == EMBER_V2_VERSION
    }

    pub fn try_parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 {
            return Err("net file too short for a header".into());
        }
        if !Self::is_format(data) {
            return Err("not an v2-net container".into());
        }
        let version = le_u32_at(data, 0);
        if version != EMBER_V2_VERSION {
            return Err(format!("unsupported net version {:#x}", version));
        }
        let hash = le_u32_at(data, 4);
        let desc_len = le_u32_at(data, 8) as usize;
        let desc_end = 12usize + desc_len;
        if desc_end > data.len() {
            return Err("net description overruns container".into());
        }
        let description = data[12..desc_end].to_vec();
        let tensors = scan_tensors(data, desc_end);

        Ok(EmberV2Info {
            version,
            hash,
            description,
            body_start: desc_end,
            tensors,
        })
    }

    #[allow(dead_code)]
    pub fn summary(&self) -> String {
        let mut s = format!(
            "ext v{:#x} arch {:#x} desc {} body {} tensors {}",
            self.version,
            self.hash,
            self.description.len(),
            self.body_start,
            self.tensors.len(),
        );
        for t in &self.tensors {
            let kind = if t.leb { "leb" } else { "raw" };
            s.push_str(&format!(" {}{}@{}", kind, t.byte_count, t.offset));
        }
        s
    }

    pub fn decode(&self, data: &[u8]) -> Result<EmberV2Data, String> {
        if self.hash != ARCH_HASH {
            return Err(format!(
                "unsupported architecture hash {:#x} (expected {:#x})",
                self.hash, ARCH_HASH
            ));
        }
        let body = self.body_start;
        if body + LEB128_MAGIC_LEN + 4 > data.len() {
            return Err("net body too short".into());
        }
        let ft_hash = le_u32_at(data, body);
        if ft_hash != FT_HEADER_HASH {
            return Err(format!("unexpected ft header {:#x}", ft_hash));
        }
        let mut pos = body + 4;
        let (ft_bias, next) = decode_leb_values(data, pos)?;
        pos = next;
        let hidden_size = ft_bias.len();
        if hidden_size != HIDDEN_SIZE {
            return Err(format!(
                "unsupported hidden size {} (expected {})",
                hidden_size, HIDDEN_SIZE
            ));
        }
        let bin = find_magic_from(data, pos).ok_or("net has no further magic")?;
        let threat_r = bin - pos;
        if !threat_r.is_multiple_of(hidden_size) {
            return Err("threat weights length invalid".into());
        }
        let threat_dims = threat_r / hidden_size;
        if threat_dims != THREAT_DIMS {
            return Err(format!(
                "unsupported threat dimensions {} (expected {})",
                threat_dims, THREAT_DIMS
            ));
        }
        let threat_weights: Vec<i8> = data[pos..bin].iter().map(|&b| b as i8).collect();
        pos = bin;
        let (threat_psqt, after) = decode_leb_values(data, pos)?;
        pos = after;
        if threat_psqt.len() % PSQT_BUCKETS != 0 {
            return Err("threat psqt length invalid".into());
        }
        if threat_psqt.len() != threat_dims * PSQT_BUCKETS {
            return Err("threat psqt dimensions do not match threat weights".into());
        }
        let (psq_weights, after) = decode_leb_values(data, pos)?;
        pos = after;
        if hidden_size == 0 || psq_weights.len() % hidden_size != 0 {
            return Err("psq weights length invalid".into());
        }
        let psq_dims = psq_weights.len() / hidden_size;
        if psq_dims != PSQ_DIMS {
            return Err(format!(
                "unsupported PSQ dimensions {} (expected {})",
                psq_dims, PSQ_DIMS
            ));
        }
        let (psqt, after) = decode_leb_values(data, pos)?;
        pos = after;
        if psq_dims == 0 || psqt.len() != psq_dims * PSQT_BUCKETS {
            return Err("psqt length invalid".into());
        }
        let tail = &data[pos..];
        if !tail.len().is_multiple_of(STACK_BYTES) {
            return Err("tail length not a multiple of stack bytes".into());
        }
        let num_stacks = tail.len() / STACK_BYTES;
        if num_stacks != NUM_STACKS {
            return Err(format!(
                "unsupported stack count {} (expected {})",
                num_stacks, NUM_STACKS
            ));
        }
        let mut stacks = Vec::with_capacity(num_stacks);
        for si in 0..num_stacks {
            let seg = &tail[si * STACK_BYTES..(si + 1) * STACK_BYTES];
            let h = u32::from_le_bytes([seg[0], seg[1], seg[2], seg[3]]);
            if h != STACK_HASH {
                return Err(format!("stack {} unexpected header {:#x}", si, h));
            }
            stacks.push(parse_stack(seg)?);
        }
        let overview = format!(
            "hidden {} psq {} threat {} psqt {} stacks {}",
            hidden_size,
            psq_dims,
            threat_dims,
            psqt.len() / psq_dims,
            num_stacks,
        );
        let ft_bias = checked_i16_values(ft_bias, "feature-transformer biases")?;
        let psq_weights = checked_i16_values(psq_weights, "PSQ weights")?;
        Ok(EmberV2Data {
            hidden_size,
            psq_dims,
            threat_dims,
            num_stacks,
            ft_bias,
            threat_weights,
            threat_psqt,
            psq_weights,
            psqt,
            stacks,
            overview,
        })
    }
}

fn checked_i16_values(values: Vec<i32>, name: &str) -> Result<Vec<i16>, String> {
    values
        .into_iter()
        .map(|value| {
            i16::try_from(value)
                .map_err(|_| format!("{} value {} is outside the i16 range", name, value))
        })
        .collect()
}

fn parse_stack(seg: &[u8]) -> Result<EmberV2Stack, String> {
    let mut cur = 4;
    let read_i32s = |cur: &mut usize, n: usize| -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if *cur + 4 > seg.len() {
                return Err("stack i32 out of range".into());
            }
            let b = &seg[*cur..*cur + 4];
            out.push(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            *cur += 4;
        }
        Ok(out)
    };
    let read_i8s = |cur: &mut usize, n: usize| -> Result<Vec<i8>, String> {
        let end = (*cur).checked_add(n).ok_or("stack len overflow")?;
        if end > seg.len() {
            return Err("stack i8 out of range".into());
        }
        let out: Vec<i8> = seg[*cur..end].iter().map(|&b| b as i8).collect();
        *cur = end;
        Ok(out)
    };
    let fc0_bias = read_i32s(&mut cur, 32)?;
    let fc0_weights = read_i8s(&mut cur, 32 * 1024)?;
    let fc1_bias = read_i32s(&mut cur, 32)?;
    let fc1_weights = read_i8s(&mut cur, 32 * 64)?;
    let fc2_bias = read_i32s(&mut cur, 1)?;
    let fc2_weights = read_i8s(&mut cur, 128)?;
    Ok(EmberV2Stack {
        fc0_bias,
        fc0_weights,
        fc1_bias,
        fc1_weights,
        fc2_bias: fc2_bias[0],
        fc2_weights,
    })
}

fn find_magic_from(data: &[u8], start: usize) -> Option<usize> {
    let mut at = start;
    while at + LEB128_MAGIC_LEN <= data.len() {
        if magic_matches_at(data, at) {
            return Some(at);
        }
        at += 1;
    }
    None
}

fn decode_leb_values(data: &[u8], start: usize) -> Result<(Vec<i32>, usize), String> {
    if !magic_matches_at(data, start) {
        return Err("expected magic".into());
    }
    let cf = start + LEB128_MAGIC_LEN;
    if cf + 4 > data.len() {
        return Err("count outside".into());
    }
    let bc = le_u32_at(data, cf) as usize;
    let dp = cf + 4;
    if dp + bc > data.len() {
        return Err("leb block outside".into());
    }
    let vals = decode_leb_block(data, dp, bc)?
        .into_iter()
        .map(|v| v as i32)
        .collect();
    Ok((vals, dp + bc))
}

pub(crate) fn load_ember_v2(data: &[u8]) -> Result<EmberV2Data, String> {
    let info = EmberV2Info::try_parse(data)?;
    info.decode(data)
}

#[allow(dead_code)]
pub(crate) fn decode_leb_block(
    data: &[u8],
    pos: usize,
    byte_count: usize,
) -> Result<Vec<i64>, String> {
    let end = pos + byte_count;
    let mut out = Vec::new();
    let mut p = pos;
    while p < end {
        let (v, next) = decode_signed_leb(data, p)?;
        out.push(v);
        p = next;
    }
    if p != end {
        return Err(format!(
            "LEB block does not end at boundary ({} vs {})",
            p, end
        ));
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn decode_signed_leb(data: &[u8], mut pos: usize) -> Result<(i64, usize), String> {
    let mut result = 0i64;
    let mut shift = 0u32;
    loop {
        let Some(&byte) = data.get(pos) else {
            return Err("leb out of range".into());
        };
        pos += 1;
        result |= i64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            if byte & 0x40 != 0 && shift < 32 {
                result |= !((1i64 << (shift + 7)) - 1);
            }
            return Ok((result, pos));
        }
        shift += 7;
        if shift > 63 {
            return Err("leb overflow".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EmberV2Info, TensorDesc, ARCH_HASH, LEB128_MAGIC, LEB128_MAGIC_LEN};

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn build(hash: u32, desc: &str, tensors: &[(bool, usize)]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, 0x6A448AFA);
        push_u32(&mut out, hash);
        push_u32(&mut out, desc.len() as u32);
        out.extend_from_slice(desc.as_bytes());
        for (leb, len) in tensors {
            if *leb {
                out.extend_from_slice(&LEB128_MAGIC);
                push_u32(&mut out, *len as u32);
                out.resize(out.len() + *len, 0);
            } else {
                out.resize(out.len() + *len, 0xAB);
            }
        }
        out
    }

    #[test]
    fn recognizes_signature() {
        assert!(EmberV2Info::is_format(&[0xfa, 0x8a, 0x44, 0x6a]));
        assert!(!EmberV2Info::is_format(&[0x45, 0x43, 0x4e, 0x31]));
        assert!(!EmberV2Info::is_format(&[0x45, 0x55, 0x4e, 0x4e]));
    }

    #[test]
    fn parses_header_and_tensor_sequence() {
        let bytes = build(
            0x1234_5678,
            "trained",
            &[
                (true, 10),
                (false, 100),
                (true, 20),
                (true, 30),
                (false, 50),
            ],
        );
        let info = EmberV2Info::try_parse(&bytes).unwrap();
        assert_eq!(info.version, 0x6a448afa);
        assert_eq!(info.hash, 0x1234_5678);
        assert_eq!(info.description.len(), 7);
        assert_eq!(info.tensors.len(), 5);
        assert!(info.tensors[0].leb);
        assert_eq!(info.tensors[0].byte_count, 10);
        assert!(!info.tensors[1].leb);
        assert_eq!(info.tensors[1].byte_count, 100);
        assert!(info.tensors[2].leb);
        assert_eq!(info.tensors[2].byte_count, 20);
        assert!(info.tensors[3].leb);
        assert_eq!(info.tensors[3].byte_count, 30);
        assert!(!info.tensors[4].leb);
        assert_eq!(info.tensors[4].byte_count, 50);
    }

    #[test]
    fn truncated_description_is_rejected() {
        let bytes = build(0, "x", &[]);
        assert!(EmberV2Info::try_parse(&bytes[..8]).is_err());
    }

    #[test]
    fn short_container_body_is_rejected_without_panicking() {
        for trailing_bytes in 1..LEB128_MAGIC_LEN {
            let mut bytes = build(ARCH_HASH, "short body", &[]);
            bytes.resize(bytes.len() + trailing_bytes, 0);

            let info = EmberV2Info::try_parse(&bytes).expect("header should still parse");
            assert!(info.decode(&bytes).is_err());
        }
    }

    #[test]
    fn decode_rejects_an_unknown_architecture_before_tensor_loading() {
        let bytes = build(0x1234_5678, "unknown", &[]);
        let info = EmberV2Info::try_parse(&bytes).expect("container header should parse");
        let error = match info.decode(&bytes) {
            Ok(_) => panic!("an unknown feature architecture must not be guessed"),
            Err(error) => error,
        };
        assert!(error.contains("unsupported architecture hash"));
    }

    #[test]
    fn tensor_desc_is_copyable() {
        let t = TensorDesc {
            leb: true,
            offset: 1,
            byte_count: 2,
        };
        let c = t;
        assert_eq!(c.offset, 1);
    }

    #[test]
    fn decodes_leb_block_until_boundary() {
        let buf = vec![0xACu8, 0x02, 0x7F, 0x7B];
        let vals = super::decode_leb_block(&buf, 0, buf.len()).expect("decode should succeed");
        assert_eq!(vals, vec![300, -1, -5]);
    }

    #[test]
    fn decode_leb_block_rejects_truncation() {
        let buf = vec![0xACu8, 0x02, 0x7F, 0x7B, 0x80];
        let result = super::decode_leb_block(&buf, 0, buf.len());
        assert!(result.is_err());
    }
}
