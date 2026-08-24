pub(crate) const OTHER_NNUE_VERSION: u32 = 0x6A448AFA;

pub(crate) const LEB128_MAGIC_LEN: usize = 17;

const LEB128_MAGIC: [u8; 17] = [
    b'C', b'O', b'M', b'P', b'R', b'E', b'S', b'S', b'E', b'D', b'_', b'L', b'E', b'B',
    b'1', b'2', b'8',
];

#[derive(Clone, Copy)]
pub(super) struct BlockRef {
    pub offset: usize,
    pub byte_count: usize,
}

pub struct OtherNetInfo {
    pub version: u32,
    pub hash: u32,
    pub description: Vec<u8>,
    pub body_start: usize,
    pub blocks: Vec<BlockRef>,
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

fn scan_leb_blocks(data: &[u8], start: usize) -> Vec<BlockRef> {
    let mut blocks = Vec::new();
    let mut from = start;
    loop {
        let Some(idx) = find_magic_at_or_after(data, from) else {
            break;
        };
        let count_field = idx + LEB128_MAGIC_LEN;
        if count_field + 4 > data.len() {
            break;
        }
        let byte_count = le_u32_at(data, count_field) as usize;
        let data_pos = count_field + 4;
        if data_pos + byte_count > data.len() {
            break;
        }
        blocks.push(BlockRef {
            offset: data_pos,
            byte_count,
        });
        from = data_pos + byte_count;
    }
    blocks
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
        let blocks = scan_leb_blocks(data, desc_end);

        Ok(OtherNetInfo {
            version,
            hash,
            description,
            body_start: desc_end,
            blocks,
        })
    }

    pub fn summary(&self) -> String {
        let mut s = format!(
            "ext v{:#x} arch {:#x} desc {} body {} blocks {}",
            self.version,
            self.hash,
            self.description.len(),
            self.body_start,
            self.blocks.len(),
        );
        for b in &self.blocks {
            s.push_str(&format!(" [{}@{}]", b.offset, b.byte_count));
        }
        s
    }
}

#[cfg(not(test))]
#[allow(dead_code)] // подключается на этапе декодирования тензоров
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
    use super::{BlockRef, LEB128_MAGIC, OtherNetInfo, decode_signed_leb};

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn build(hash: u32, desc: &str, block_bytes: &[usize]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, 0x6A448AFA);
        push_u32(&mut out, hash);
        push_u32(&mut out, desc.len() as u32);
        out.extend_from_slice(desc.as_bytes());
        for byte_count in block_bytes {
            out.extend_from_slice(&LEB128_MAGIC);
            push_u32(&mut out, *byte_count as u32);
            out.resize(out.len() + *byte_count, 0);
        }
        out
    }

    #[test]
    fn recognizes_container_signature() {
        let sig: &[u8] = &[0xfa, 0x8a, 0x44, 0x6a];
        assert!(OtherNetInfo::is_format(sig));
        let ember_compact: &[u8] = &[0x45, 0x43, 0x4e, 0x31];
        let ember_dense: &[u8] = &[0x45, 0x55, 0x4e, 0x4e];
        assert!(!OtherNetInfo::is_format(ember_compact));
        assert!(!OtherNetInfo::is_format(ember_dense));
    }

    #[test]
    fn parses_header_and_blocks() {
        let bytes = build(0x1234_5678, "trained", &[71, 41]);
        let info = OtherNetInfo::try_parse(&bytes).unwrap();
        assert_eq!(info.version, 0x6a448afa);
        assert_eq!(info.hash, 0x1234_5678);
        assert_eq!(info.description.len(), 7);
        assert_eq!(info.blocks.len(), 2);
        assert_eq!(info.blocks[0].byte_count, 71);
        assert_eq!(info.blocks[1].byte_count, 41);
        assert!(info.blocks[0].offset < info.blocks[1].offset);
    }

    #[test]
    fn truncated_description_is_rejected() {
        let bytes = build(0, "x", &[]);
        let result = OtherNetInfo::try_parse(&bytes[..8]);
        assert!(result.is_err());
    }

    #[test]
    fn leb_decodes_signed_values() {
        let mut buf = vec![0xACu8, 0x02]; // 300
        buf.push(0x7F); // -1
        buf.push(0x7B); // -5
        let (v, pos) = decode_signed_leb(&buf, 0).unwrap();
        assert_eq!(v, 300);
        let (v2, pos2) = decode_signed_leb(&buf, pos).unwrap();
        assert_eq!(v2, -1);
        let (v3, pos3) = decode_signed_leb(&buf, pos2).unwrap();
        assert_eq!(v3, -5);
        assert_eq!(pos3, buf.len());
    }

    #[test]
    fn block_ref_is_copyable() {
        let b = BlockRef { offset: 1, byte_count: 2 };
        let c = b;
        assert_eq!(c.offset, 1);
    }

    #[test]
    fn parses_real_downloaded_net_if_present() {
        let path = "nn-0ee0657fb25e.nnue";
        if let Ok(data) = std::fs::read(path) {
            let info = OtherNetInfo::try_parse(&data).expect("real net should parse");
            assert_eq!(info.blocks.len(), 4, "{}", info.summary());
            let expected = [100usize, 62_178_672, 62_973_334, 89_816_236];
            for (i, blk) in info.blocks.iter().enumerate() {
                assert!(
                    blk.offset > expected[i] && blk.offset <= expected[i] + 21,
                    "block {} offset {}, expected near {}",
                    i,
                    blk.offset,
                    expected[i]
                );
            }
        }
    }
}