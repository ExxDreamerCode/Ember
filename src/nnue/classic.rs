use crate::board::BoardState;
use crate::types::{BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK};

pub(crate) const CLASSIC_KIND_VERSION: u32 = 0x7AF3_2F16;
const HALF_DIM: usize = 256;
const TRANSFORMED_DIM: usize = HALF_DIM * 2;
const PER_KING_PLANE: usize = 10 * 64 + 1;
const FT_DIM: usize = 64 * PER_KING_PLANE;
const L1_OUT: usize = 32;
const L2_OUT: usize = 32;
const SCALE: i32 = 16;
const FT_HEADER_HASH: u32 = 0x5D69_D7B8;

#[inline(always)]
fn orient(perspective: usize, sf_sq: usize) -> usize {
    sf_sq ^ (perspective * 63)
}

#[inline(always)]
fn ember_to_sf(square: usize) -> usize {
    square ^ 56
}

#[inline(always)]
fn ps_base(piece_type: u8, enemy: bool) -> usize {
    let base = match piece_type {
        PAWN => 0,
        KNIGHT => 2,
        BISHOP => 4,
        ROOK => 6,
        QUEEN => 8,
        _ => unreachable!("pieces without kings only"),
    };
    base * 64 + 1 + usize::from(enemy) * 64
}
#[derive(Clone)]
pub struct ClassicHalfKpNet {
    pub arch_hash: u32,
    pub feature_hash: u32,
    pub net_hash: u32,
    pub description: String,
    ft_bias: Vec<i16>,
    ft_weights: Vec<i16>,
    l1_bias: Vec<i32>,
    l1_weights: Vec<i8>,
    l2_bias: Vec<i32>,
    l2_weights: Vec<i8>,
    out_bias: i32,
    out_weights: Vec<i8>,
}

fn u32_at(data: &[u8], at: usize) -> Result<u32, String> {
    let seg = data
        .get(at..at + 4)
        .ok_or_else(|| "read out of range".to_string())?;
    Ok(u32::from_le_bytes([seg[0], seg[1], seg[2], seg[3]]))
}

fn read_i16_vec(data: &[u8], n: usize, pos: &mut usize, what: &str) -> Result<Vec<i16>, String> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let seg = data
            .get(*pos..*pos + 2)
            .ok_or_else(|| format!("{what} out of range"))?;
        out.push(i16::from_le_bytes([seg[0], seg[1]]));
        *pos += 2;
    }
    Ok(out)
}

fn read_i32_vec(data: &[u8], n: usize, pos: &mut usize, what: &str) -> Result<Vec<i32>, String> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let seg = data
            .get(*pos..*pos + 4)
            .ok_or_else(|| format!("{what} out of range"))?;
        out.push(i32::from_le_bytes([seg[0], seg[1], seg[2], seg[3]]));
        *pos += 4;
    }
    Ok(out)
}

fn read_i8_vec(data: &[u8], n: usize, pos: &mut usize, what: &str) -> Result<Vec<i8>, String> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let b = data
            .get(*pos)
            .copied()
            .ok_or_else(|| format!("{what} out of range"))?;
        out.push(b as i8);
        *pos += 1;
    }
    Ok(out)
}

impl ClassicHalfKpNet {
    pub fn is_format(data: &[u8]) -> bool {
        data.len() >= 4
            && u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == CLASSIC_KIND_VERSION
    }

    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if !Self::is_format(data) {
            return Err("not a legacy HalfKP network".into());
        }
        let arch_hash = u32_at(data, 4)?;
        let desc_len = u32_at(data, 8)? as usize;
        let desc_end = desc_len
            .checked_add(12)
            .ok_or("description length overflow")?;
        if desc_end > data.len() {
            return Err("net description overruns the file".into());
        }
        let description = String::from_utf8_lossy(&data[12..desc_end]).into_owned();

        let mut pos = desc_end;
        let feature_hash = u32_at(data, pos)?;
        pos += 4;
        if feature_hash != FT_HEADER_HASH {
            return Err(format!(
                "unsupported feature hash {feature_hash:#x} (expected HalfKP(Friend) {FT_HEADER_HASH:#x})"
            ));
        }

        let ft_bias = read_i16_vec(data, HALF_DIM, &mut pos, "ft_bias")?;
        let ft_weights = read_i16_vec(data, FT_DIM * HALF_DIM, &mut pos, "ft_weights")?;

        let net_hash = u32_at(data, pos)?;
        pos += 4;

        let l1_bias = read_i32_vec(data, L1_OUT, &mut pos, "l1_bias")?;
        let l1_weights = read_i8_vec(data, L1_OUT * TRANSFORMED_DIM, &mut pos, "l1_weights")?;
        let l2_bias = read_i32_vec(data, L2_OUT, &mut pos, "l2_bias")?;
        let l2_weights = read_i8_vec(data, L2_OUT * L2_OUT, &mut pos, "l2_weights")?;
        let out_bias = {
            let seg = data
                .get(pos..pos + 4)
                .ok_or_else(|| "out_bias out of range".to_string())?;
            pos += 4;
            i32::from_le_bytes([seg[0], seg[1], seg[2], seg[3]])
        };
        let out_weights = read_i8_vec(data, L2_OUT, &mut pos, "out_weights")?;

        if pos != data.len() {
            return Err(format!(
                "legacy HalfKP size mismatch (consumed {pos} of {} bytes)",
                data.len()
            ));
        }

        Ok(ClassicHalfKpNet {
            arch_hash,
            feature_hash,
            net_hash,
            description,
            ft_bias,
            ft_weights,
            l1_bias,
            l1_weights,
            l2_bias,
            l2_weights,
            out_bias,
            out_weights,
        })
    }

    pub fn overview(&self) -> String {
        format!(
            "HalfKP(Friend) 41024->256, hidden 512->32->32->1 (features={:#x} net={:#x})",
            self.feature_hash, self.net_hash
        )
    }

    pub fn evaluate_stm(&self, st: &BoardState) -> i32 {
        let stm = if st.w { 0usize } else { 1usize };
        let mut acc = [[0i16; HALF_DIM]; 2];
        for persp in 0..2usize {
            let color = persp;
            let king_sq = ember_to_sf(st.king_sq(color == 0));
            let ksq = orient(persp, king_sq);
            for lane in 0..HALF_DIM {
                acc[persp][lane] = self.ft_bias[lane];
            }
            for pi in 0..12usize {
                let piece_color = pi / 6;
                let piece_type = pi % 6;
                if piece_type == KING as usize {
                    continue;
                }
                let enemy = piece_color != persp;
                let base = ps_base(piece_type as u8, enemy);
                let mut bb = st.bb[pi];
                while bb != 0 {
                    let s = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let idx = orient(persp, ember_to_sf(s)) + base + PER_KING_PLANE * ksq;
                    let row_start = idx * HALF_DIM;
                    for lane in 0..HALF_DIM {
                        acc[persp][lane] = (acc[persp][lane] as i32
                            + self.ft_weights[row_start + lane] as i32)
                            as i16;
                    }
                }
            }
        }

        let mut input = [0i32; TRANSFORMED_DIM];
        for (half, persp) in [stm, 1 - stm].iter().copied().enumerate() {
            let base = half * HALF_DIM;
            for j in 0..HALF_DIM {
                input[base + j] = acc[persp][j].clamp(0, 127) as i32;
            }
        }

        let mut h1 = [0i32; L1_OUT];
        for o in 0..L1_OUT {
            let mut s = self.l1_bias[o];
            for i in 0..TRANSFORMED_DIM {
                s += self.l1_weights[o * TRANSFORMED_DIM + i] as i32 * input[i];
            }
            h1[o] = (s >> 6).clamp(0, 127);
        }

        let mut h2 = [0i32; L2_OUT];
        for o in 0..L2_OUT {
            let mut s = self.l2_bias[o];
            for i in 0..L2_OUT {
                s += self.l2_weights[o * L2_OUT + i] as i32 * h1[i];
            }
            h2[o] = (s >> 6).clamp(0, 127);
        }

        let mut out = self.out_bias;
        for i in 0..L2_OUT {
            out += self.out_weights[i] as i32 * h2[i];
        }
        out / SCALE
    }
}

#[cfg(test)]
pub(crate) fn synthetic_test_net_bytes(out_bias: i32) -> Vec<u8> {
    let mut out = Vec::new();
    let push_u32 = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());

    push_u32(&mut out, CLASSIC_KIND_VERSION);
    push_u32(&mut out, 0x1234_5678);
    let desc = "Features=HalfKP(Friend)[41024->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))";
    push_u32(&mut out, desc.len() as u32);
    out.extend_from_slice(desc.as_bytes());

    push_u32(&mut out, FT_HEADER_HASH);
    for _ in 0..HALF_DIM {
        out.extend_from_slice(&0i16.to_le_bytes());
    }
    for _ in 0..FT_DIM * HALF_DIM {
        out.extend_from_slice(&0i16.to_le_bytes());
    }
    push_u32(&mut out, 0xBAD_F00D);
    for _ in 0..L1_OUT {
        push_u32(&mut out, 0);
    }
    out.extend(
        std::iter::repeat(0i8)
            .map(|x| x as u8)
            .take(L1_OUT * TRANSFORMED_DIM),
    );
    for _ in 0..L2_OUT {
        push_u32(&mut out, 0);
    }
    out.extend(
        std::iter::repeat(0i8)
            .map(|x| x as u8)
            .take(L2_OUT * L2_OUT),
    );
    push_u32(&mut out, out_bias as u32);
    out.extend(std::iter::repeat(0i8).map(|x| x as u8).take(L2_OUT));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_net() -> Vec<u8> {
        synthetic_test_net_bytes(0)
    }

    #[test]
    fn detects_legacy_format() {
        let good = [0x16u8, 0x2f, 0xf3, 0x7a, 0xaa, 0xbb, 0xcc, 0xdd];
        assert!(ClassicHalfKpNet::is_format(&good));
        assert!(!ClassicHalfKpNet::is_format(&[0x45, 0x4e, 0x4e, 0x55]));
        assert!(!ClassicHalfKpNet::is_format(&[]));
    }

    #[test]
    fn parses_synthetic_and_evaluates_zero() {
        let bytes = synthetic_net();
        let net = ClassicHalfKpNet::parse(&bytes).expect("synthetic net should parse");
        assert_eq!(net.arch_hash, 0x1234_5678);
        assert_eq!(net.feature_hash, FT_HEADER_HASH);
        assert!(net.description.contains("HalfKP"));
        let st = crate::board::BoardState::empty();
        assert_eq!(net.evaluate_stm(&st), 0);
    }

    #[test]
    fn rejects_unknown_magic() {
        let mut bytes = synthetic_net();
        bytes[0] = 0x45;
        bytes[1] = 0x4e;
        bytes[2] = 0x4e;
        bytes[3] = 0x55;
        assert!(ClassicHalfKpNet::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated_file() {
        let bytes = synthetic_net();
        let cut = &bytes[..bytes.len() - 7];
        assert!(ClassicHalfKpNet::parse(cut).is_err());
    }
}
