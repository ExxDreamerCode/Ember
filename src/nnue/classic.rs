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
const EMPTY_SQ: u8 = crate::board::EMPTY_SQ;

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

#[inline(always)]
fn king_square(state: &BoardState, perspective: usize) -> usize {
    ember_to_sf(state.king_sq(perspective == 0))
}

#[inline(always)]
fn feature_index(
    perspective: usize,
    oriented_king: usize,
    square: usize,
    piece: u8,
) -> Option<usize> {
    let piece_type = piece % 6;
    if piece_type == KING as u8 {
        return None;
    }
    let piece_color = piece / 6;
    let enemy = usize::from(piece_color) != perspective;
    let base = ps_base(piece_type, enemy);
    let sf_sq = ember_to_sf(square);
    Some(orient(perspective, sf_sq) + base + PER_KING_PLANE * oriented_king)
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

#[derive(Clone)]
pub struct ClassicHalfKpAccumulator {
    accumulation: [[i16; HALF_DIM]; 2],
}

impl ClassicHalfKpAccumulator {
    pub fn new() -> Self {
        Self {
            accumulation: [[0; HALF_DIM]; 2],
        }
    }

    #[inline(always)]
    fn add_row(&mut self, net: &ClassicHalfKpNet, perspective: usize, index: usize) {
        let row_start = index * HALF_DIM;
        let row = &net.ft_weights[row_start..row_start + HALF_DIM];
        for (dst, weight) in self.accumulation[perspective].iter_mut().zip(row) {
            *dst = dst.wrapping_add(*weight);
        }
    }

    #[inline(always)]
    fn remove_row(&mut self, net: &ClassicHalfKpNet, perspective: usize, index: usize) {
        let row_start = index * HALF_DIM;
        let row = &net.ft_weights[row_start..row_start + HALF_DIM];
        for (dst, weight) in self.accumulation[perspective].iter_mut().zip(row) {
            *dst = dst.wrapping_sub(*weight);
        }
    }

    pub fn refresh(&mut self, net: &ClassicHalfKpNet, state: &BoardState) {
        for perspective in 0..2usize {
            self.refresh_perspective(net, state, perspective);
        }
    }

    fn refresh_perspective(
        &mut self,
        net: &ClassicHalfKpNet,
        state: &BoardState,
        perspective: usize,
    ) {
        for (dst, bias) in self.accumulation[perspective]
            .iter_mut()
            .zip(net.ft_bias.iter())
        {
            *dst = *bias;
        }

        let oriented_king = orient(perspective, king_square(state, perspective));
        for (piece, &bb) in state.bb.iter().enumerate() {
            let mut bb = bb;
            while bb != 0 {
                let square = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                if let Some(index) = feature_index(perspective, oriented_king, square, piece as u8)
                {
                    self.add_row(net, perspective, index);
                }
            }
        }
    }

    pub fn update_from_parent(
        &mut self,
        parent: &Self,
        net: &ClassicHalfKpNet,
        before: &BoardState,
        after: &BoardState,
    ) {
        self.clone_from(parent);
        for perspective in 0..2usize {
            if king_square(before, perspective) != king_square(after, perspective) {
                self.refresh_perspective(net, after, perspective);
                continue;
            }

            let oriented_king = orient(perspective, king_square(after, perspective));
            for square in 0..64usize {
                let before_piece = before.mailbox[square];
                let after_piece = after.mailbox[square];
                if before_piece == after_piece {
                    continue;
                }
                if before_piece != EMPTY_SQ {
                    if let Some(index) =
                        feature_index(perspective, oriented_king, square, before_piece)
                    {
                        self.remove_row(net, perspective, index);
                    }
                }
                if after_piece != EMPTY_SQ {
                    if let Some(index) =
                        feature_index(perspective, oriented_king, square, after_piece)
                    {
                        self.add_row(net, perspective, index);
                    }
                }
            }
        }
    }
}

impl Default for ClassicHalfKpAccumulator {
    fn default() -> Self {
        Self::new()
    }
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
        let mut acc = ClassicHalfKpAccumulator::new();
        acc.refresh(self, st);
        self.evaluate_stm_acc(&acc, st)
    }

    pub fn evaluate_stm_acc(&self, acc: &ClassicHalfKpAccumulator, st: &BoardState) -> i32 {
        let stm = usize::from(!st.w);

        let mut input = [0i32; TRANSFORMED_DIM];
        for (half, &perspective) in [stm, 1 - stm].iter().enumerate() {
            let base = half * HALF_DIM;
            for (offset, value) in acc.accumulation[perspective].iter().enumerate() {
                input[base + offset] = i32::from(*value).clamp(0, 127);
            }
        }

        let mut h1 = [0i32; L1_OUT];
        for (o, h1_value) in h1.iter_mut().enumerate() {
            let weights = &self.l1_weights[o * TRANSFORMED_DIM..(o + 1) * TRANSFORMED_DIM];
            let mut s = self.l1_bias[o];
            for (weight, value) in weights.iter().zip(input.iter()) {
                s += i32::from(*weight) * value;
            }
            *h1_value = (s >> 6).clamp(0, 127);
        }

        let mut h2 = [0i32; L2_OUT];
        for (o, h2_value) in h2.iter_mut().enumerate() {
            let weights = &self.l2_weights[o * L2_OUT..(o + 1) * L2_OUT];
            let mut s = self.l2_bias[o];
            for (weight, value) in weights.iter().zip(h1.iter()) {
                s += i32::from(*weight) * value;
            }
            *h2_value = (s >> 6).clamp(0, 127);
        }

        let mut out = self.out_bias;
        for (weight, value) in self.out_weights.iter().zip(h2.iter()) {
            out += i32::from(*weight) * value;
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
    use crate::movegen::{apply_move, generate_moves};

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

    #[test]
    fn incremental_accumulator_matches_full_refresh() {
        let mut bytes = synthetic_test_net_bytes(7);
        let mut rng_state = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state
        };
        let ft_bias_start = 12
            + 4
            + {
                let desc = "Features=HalfKP(Friend)[41024->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))";
                desc.len()
            }
            + 4;
        let ft_weights_start = ft_bias_start + HALF_DIM * 2;
        let ft_weights_end = ft_weights_start + FT_DIM * HALF_DIM * 2;
        for chunk in bytes[ft_weights_start..ft_weights_end].chunks_exact_mut(2) {
            let value = (next() % 61 - 30) as i16;
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        let net = ClassicHalfKpNet::parse(&bytes).expect("synthetic net should parse");

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ];

        for fen in fens {
            let mut engine = crate::Engine::new();
            engine.set_fen(fen);
            let before = engine.st;
            let mut parent = ClassicHalfKpAccumulator::new();
            parent.refresh(&net, &before);
            for mv in generate_moves(&before, before.w, &before.cr, before.ep) {
                let mut after = before;
                apply_move(
                    &mut after,
                    crate::board::move_sr(mv),
                    crate::board::move_sc(mv),
                    crate::board::move_er(mv),
                    crate::board::move_ec(mv),
                    crate::board::move_promotion(mv),
                );
                let mut incremental = ClassicHalfKpAccumulator::new();
                incremental.update_from_parent(&parent, &net, &before, &after);
                let mut refreshed = ClassicHalfKpAccumulator::new();
                refreshed.refresh(&net, &after);
                assert_eq!(
                    incremental.accumulation, refreshed.accumulation,
                    "incremental accumulator mismatch after move {mv} from {fen}"
                );
            }
        }
    }
}
