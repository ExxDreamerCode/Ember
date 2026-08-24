pub(crate) const OTHER_NNUE_VERSION: u32 = 0x6A448AFA;

pub(crate) const LEB128_MAGIC_LEN: usize = 17;

const LEB128_MAGIC: [u8; 17] = [
    b'C', b'O', b'M', b'P', b'R', b'E', b'S', b'S', b'E', b'D', b'_', b'L', b'E', b'B',
    b'1', b'2', b'8',
];

#[derive(Clone, Copy)]
pub struct TensorDesc {
    pub leb: bool,
    pub offset: usize,
    pub byte_count: usize,
}

pub struct OtherNetInfo {
    pub version: u32,
    pub hash: u32,
    pub description: Vec<u8>,
    pub body_start: usize,
    pub tensors: Vec<TensorDesc>,
}

fn le_u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

fn magic_matches_at(data: &[u8], at: usize) -> bool {
    (0..LEB128_MAGIC_LEN).all(|i| data[at + i] == LEB128_MAGIC[i])
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
            out.push(TensorDesc { leb: true, offset: data_pos, byte_count: bc });
            pos = data_pos + bc;
        } else {
            let next = find_magic_at_or_after(data, pos);
            let end = next.unwrap_or(data.len());
            out.push(TensorDesc { leb: false, offset: pos, byte_count: end - pos });
            pos = end;
        }
    }
    out
}

impl OtherNetInfo {
    pub fn is_format(data: &[u8]) -> bool {
        data.len() >= 4 && le_u32_at(data, 0) == OTHER_NNUE_VERSION
    }

    pub fn try_parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 {
            return Err("net file too short for a header".into());
        }
        if !Self::is_format(data) {
            return Err("not an external-net container".into());
        }
        let version = le_u32_at(data, 0);
        if version != OTHER_NNUE_VERSION {
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

        Ok(OtherNetInfo {
            version,
            hash,
            description,
            body_start: desc_end,
            tensors,
        })
    }

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
}

#[allow(dead_code)]
pub(crate) fn decode_leb_block(data: &[u8], pos: usize, byte_count: usize) -> Result<Vec<i64>, String> {
    let end = pos + byte_count;
    let mut out = Vec::new();
    let mut p = pos;
    while p < end {
        let (v, next) = decode_signed_leb(data, p)?;
        out.push(v);
        p = next;
    }
    if p != end {
        return Err(format!("LEB block does not end at boundary ({} vs {})", p, end));
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
    use super::{LEB128_MAGIC, OtherNetInfo, TensorDesc};

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
        assert!(OtherNetInfo::is_format(&[0xfa, 0x8a, 0x44, 0x6a]));
        assert!(!OtherNetInfo::is_format(&[0x45, 0x43, 0x4e, 0x31]));
        assert!(!OtherNetInfo::is_format(&[0x45, 0x55, 0x4e, 0x4e]));
    }

    #[test]
    fn parses_header_and_tensor_sequence() {
        let bytes = build(
            0x1234_5678,
            "trained",
            &[(true, 10), (false, 100), (true, 20), (true, 30), (false, 50)],
        );
        let info = OtherNetInfo::try_parse(&bytes).unwrap();
        assert_eq!(info.version, 0x6a448afa);
        assert_eq!(info.hash, 0x1234_5678);
        assert_eq!(info.description.len(), 7);
        assert_eq!(info.tensors.len(), 5);
        assert_eq!(info.tensors[0].leb, true);
        assert_eq!(info.tensors[0].byte_count, 10);
        assert_eq!(info.tensors[1].leb, false);
        assert_eq!(info.tensors[1].byte_count, 100);
        assert_eq!(info.tensors[2].leb, true);
        assert_eq!(info.tensors[2].byte_count, 20);
        assert_eq!(info.tensors[3].leb, true);
        assert_eq!(info.tensors[3].byte_count, 30);
        assert_eq!(info.tensors[4].leb, false);
        assert_eq!(info.tensors[4].byte_count, 50);
    }

    #[test]
    fn truncated_description_is_rejected() {
        let bytes = build(0, "x", &[]);
        assert!(OtherNetInfo::try_parse(&bytes[..8]).is_err());
    }

    #[test]
    fn parses_real_downloaded_net_if_present() {
        let path = "nn-0ee0657fb25e.nnue";
        if let Ok(data) = std::fs::read(path) {
            let info = OtherNetInfo::try_parse(&data).expect("real net should parse");
            assert_eq!(info.tensors.len(), 7, "{}", info.summary());
            let kinds: Vec<bool> = info.tensors.iter().map(|t| t.leb).collect();
            assert_eq!(kinds, vec![false, true, false, true, true, true, false], "{}", info.summary());
        }
    }

    #[test]
    fn tensor_desc_is_copyable() {
        let t = TensorDesc { leb: true, offset: 1, byte_count: 2 };
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

    #[test]
    fn real_net_dimensions_are_consistent() {
        let path = "nn-0ee0657fb25e.nnue";
        if let Ok(data) = std::fs::read(path) {
            let info = OtherNetInfo::try_parse(&data).expect("real net should parse");
            assert_eq!(info.tensors.len(), 7);
            let biases = super::decode_leb_block(&data, info.tensors[1].offset, info.tensors[1].byte_count)
                .expect("biases should decode");
            assert_eq!(biases.len(), 1024);
            assert_eq!(info.tensors[2].byte_count, 60720 * 1024);
            let threat_psqt = super::decode_leb_block(&data, info.tensors[3].offset, info.tensors[3].byte_count)
                .expect("threat psqt should decode");
            assert_eq!(threat_psqt.len(), 60720 * 8);
            let psq_w_len = info.tensors[4].byte_count;
            let psqt_len = info.tensors[5].byte_count;
            assert!(psq_w_len > 22_000_000, "psqW block too small: {}", psq_w_len);
            assert!(psqt_len > 100_000, "psqt block too small: {}", psqt_len);
        }
    }
}