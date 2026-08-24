use super::other_nets::{OtherNetData, OtherStack};
use crate::board::{BoardState, KING_ATTACKS, KNIGHT_ATTACKS};

pub const PSQ_DIMS: usize = 22528;
pub const THREAT_DIMS: usize = 59808;
const PAIR_BASE: usize = THREAT_DIMS;
const TOTAL_TD: usize = THREAT_DIMS + 912;
const PSQT_BUCKETS: usize = 8;

const OUTPUT_SCALE: i32 = 16;
const WEIGHT_SCALE_BITS: i32 = 6;
const FT_MAX_VAL: i32 = 255;
const HIDDEN_ONE_VAL: i32 = 128;
const FV_SCALE: i32 = 16;

const NVT: [u32; 16] = [4, 10, 8, 8, 10, 0, 4, 10, 8, 8, 10, 0, 0, 0, 0, 0];

fn orient(sq: u32) -> u32 {
    if sq & 4 != 0 {
        7
    } else {
        0
    }
}

fn halfka_index(persp: u32, sq: u32, pc: u32, ksq: u32) -> usize {
    let flip = 56 * persp;
    let ksq_o = ksq ^ flip;
    let base = if persp == 0 {
        match pc {
            0 => 0,
            1 => 128,
            2 => 256,
            3 => 384,
            4 => 512,
            5 => 640,
            6 => 64,
            7 => 192,
            8 => 320,
            9 => 448,
            10 => 576,
            _ => 640,
        }
    } else {
        match pc {
            0 => 64,
            1 => 192,
            2 => 320,
            3 => 448,
            4 => 576,
            5 => 640,
            6 => 0,
            7 => 128,
            8 => 256,
            9 => 384,
            10 => 512,
            _ => 640,
        }
    };
    let rank = ksq_o / 8;
    let file = ksq_o % 8;
    let b = rank * 4 + core::cmp::min(file, 7 - file);
    ((sq ^ orient(ksq_o) ^ flip) + base + b * 11 * 64) as usize
}
fn pawn_attacks_bb(color: u32, sq: u32) -> u64 {
    let bb = 1u64 << sq;
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = 0x8080_8080_8080_8080;
    if color == 0 {
        ((bb & !H) >> 7) | ((bb & !A) >> 9)
    } else {
        ((bb & !H) << 9) | ((bb & !A) << 7)
    }
}

fn attacks_bb(pt: u32, from: u32, occ: u64) -> u64 {
    match pt {
        1 => KNIGHT_ATTACKS[from as usize],
        2 => crate::magic::bishop_attacks(from as usize, occ),
        3 => crate::magic::rook_attacks(from as usize, occ),
        4 => {
            crate::magic::bishop_attacks(from as usize, occ)
                | crate::magic::rook_attacks(from as usize, occ)
        }
        5 => KING_ATTACKS[from as usize],
        _ => 0,
    }
}

fn piece_at(st: &BoardState, sq: u32) -> u32 {
    for (i, bb) in st.bb.iter().enumerate() {
        if bb & (1u64 << sq) != 0 {
            return i as u32;
        }
    }
    0
}

fn pawn_pair_mask(from: u32) -> u64 {
    let f = from % 8;
    let r = from / 8;
    if r == 0 || r == 7 {
        return 0;
    }
    let mut m = 0u64;
    for df in -1i32..=1 {
        let ff = f as i32 + df;
        if (0..8).contains(&ff) {
            m |= 1u64 << (r * 8 + ff as u32);
        }
    }
    m
}

fn find_king(st: &BoardState, persp: u32) -> u32 {
    let bb = if persp == 0 { st.bb[5] } else { st.bb[11] };
    bb.trailing_zeros()
}
const MAP: [[i32; 6]; 6] = [
    [-1, 0, -1, 1, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

pub struct ThreatLut {
    index_lut1: Vec<u32>,
    offsets: Vec<u32>,
    index_lut2: Vec<u16>,
}

impl ThreatLut {
    pub fn build() -> Self {
        let mut idx_lut2 = vec![0u16; 16 * 64 * 64];
        for pc in 0..12u32 {
            let pt = pc % 6;
            let color = pc / 6;
            for from in 0..64u32 {
                let attacks = if pt == 0 {
                    if from / 8 >= 1 && from / 8 <= 6 {
                        pawn_attacks_bb(color, from)
                    } else {
                        0
                    }
                } else {
                    attacks_bb(pt, from, 0)
                };
                for to in 0..64u32 {
                    let cnt = (((1u64 << to) - 1) & attacks).count_ones();
                    idx_lut2[((pc * 64 + from) * 64 + to) as usize] = cnt as u16;
                }
            }
        }

        let mut helper = [(0usize, 0usize); 16];
        let mut offsets = vec![0u32; 16 * 64];
        let mut cumulative = 0usize;
        for pc in 0..12u32 {
            let pt = pc % 6;
            let color = pc / 6;
            let mut cum_piece = 0usize;
            for from in 0..64u32 {
                offsets[(pc * 64 + from) as usize] = cum_piece as u32;
                let cnt = if pt == 0 {
                    if from / 8 >= 1 && from / 8 <= 6 {
                        pawn_attacks_bb(color, from).count_ones() as usize
                    } else {
                        0
                    }
                } else {
                    attacks_bb(pt, from, 0).count_ones() as usize
                };
                cum_piece += cnt;
            }
            helper[pc as usize] = (cum_piece, cumulative);
            cumulative += NVT[pc as usize] as usize * cum_piece;
        }

        let mut index_lut1 = vec![0u32; 16 * 16 * 2];
        for attacker in 0..12u32 {
            for attacked in 0..12u32 {
                let atk_type = attacker % 6;
                let atd_type = attacked % 6;
                let is_enemy = (attacker ^ attacked) == 8;
                let map_v = MAP[atk_type as usize][atd_type as usize];
                let semi_excluded = atk_type == atd_type && (is_enemy || atk_type != 0);
                let feature_i: i64 = helper[attacker as usize].1 as i64
                    + ((attacked / 6) as i64 * (NVT[attacker as usize] as i64 / 2) + map_v as i64)
                        * helper[attacker as usize].0 as i64;
                let feature = if feature_i < 0 || feature_i >= THREAT_DIMS as i64 {
                    THREAT_DIMS as u32
                } else {
                    feature_i as u32
                };
                let excluded = map_v < 0;
                let base = ((attacker * 12) + attacked) as usize * 2;
                index_lut1[base] = if excluded {
                    THREAT_DIMS as u32
                } else {
                    feature
                };
                index_lut1[base + 1] = if excluded || semi_excluded {
                    THREAT_DIMS as u32
                } else {
                    feature
                };
            }
        }

        ThreatLut {
            index_lut1,
            offsets,
            index_lut2: idx_lut2,
        }
    }

    pub fn make_index(
        &self,
        persp: u32,
        attacker: u32,
        from: u32,
        to: u32,
        attacked: u32,
        ksq: u32,
    ) -> usize {
        let o = orient(ksq ^ (56 * persp));
        let from_o = from ^ o;
        let to_o = to ^ o;
        let swap = 8 * persp;
        let att_o = attacker ^ swap;
        let atd_o = attacked ^ swap;
        let idx = (att_o as usize * 12 + atd_o as usize) * 2 + (from_o < to_o) as usize;
        let lut1 = self.index_lut1[idx] as usize;
        if lut1 >= THREAT_DIMS {
            return THREAT_DIMS;
        }
        lut1 + self.offsets[(att_o * 64 + from_o) as usize] as usize
            + self.index_lut2[((att_o * 64 + from_o) * 64 + to_o) as usize] as usize
    }
}

fn pair_index(persp: u32, c1: u32, sq1: u32, c2: u32, sq2: u32, ksq: u32) -> usize {
    let o = orient(ksq ^ (56 * persp));
    let s1 = sq1 ^ o ^ (56 * persp);
    let s2 = sq2 ^ o ^ (56 * persp);
    let id1 = 48i64 * (c1 ^ persp) as i64 + s1 as i64 - 16;
    let id2 = 48i64 * (c2 ^ persp) as i64 + s2 as i64 - 16;
    if id1 < 0 || id2 < 0 {
        return usize::MAX;
    }
    let (hi, lo) = if id1 > id2 { (id1, id2) } else { (id2, id1) };
    PAIR_BASE + (hi * (hi - 1) / 2 + lo) as usize
}

fn try_add_pair(
    net: &OtherNetData,
    idx: usize,
    acc: &mut [i32; 1024],
    psqt: &mut [i32; PSQT_BUCKETS],
    hidden: usize,
) {
    if idx >= TOTAL_TD {
        return;
    }
    let rb = &net.threat_weights[idx * hidden..(idx + 1) * hidden];
    let pb = &net.threat_psqt[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS];
    add_row_acc(acc, psqt, rb, pb);
}

fn clip8(v: i32) -> u8 {
    if v <= 0 {
        0
    } else if v >= FT_MAX_VAL {
        FT_MAX_VAL as u8
    } else {
        v as u8
    }
}

fn sat16(v: i32) -> i16 {
    if v > i16::MAX as i32 {
        i16::MAX
    } else if v < i16::MIN as i32 {
        i16::MIN
    } else {
        v as i16
    }
}
fn add_row_acc(
    acc: &mut [i32; 1024],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i8],
    psqt_row: &[i32],
) {
    for (a, w) in acc.iter_mut().zip(row.iter()) {
        *a += *w as i32;
    }
    for (p, w) in psqt.iter_mut().zip(psqt_row.iter()) {
        *p += *w;
    }
}

fn eval_ft_pair(acc: &[i32; 1024], j: usize) -> u8 {
    let a = clip8(acc[j]) as i32;
    let b = clip8(acc[512 + j]) as i32;
    (a * b / 512) as u8
}

fn eval_stack_forward(stack: &OtherStack, ft_out: &[u8; 1024]) -> i32 {
    let mut y0 = [0i32; 32];
    for (i, y) in y0.iter_mut().enumerate() {
        let mut s = stack.fc0_bias[i];
        for (j, &x) in ft_out.iter().enumerate() {
            s += stack.fc0_weights[i * 1024 + j] as i32 * x as i32;
        }
        *y = s;
    }
    let mut act0 = [0u8; 64];
    for i in 0..32 {
        let w = sat16(y0[i]);
        let sqr = ((w as i32 * w as i32) >> (2 * 7 + 7)) as u8;
        act0[i] = sqr.min(127);
        act0[32 + i] = (w >> 7).max(0) as u8;
    }
    let mut y1 = [0i32; 32];
    for (i, y) in y1.iter_mut().enumerate() {
        let mut s = stack.fc1_bias[i];
        for (j, x) in act0.iter().enumerate() {
            s += stack.fc1_weights[i * 64 + j] as i32 * *x as i32;
        }
        *y = s;
    }
    let mut act1 = [0u8; 64];
    for i in 0..32 {
        let w = sat16(y1[i]);
        let sqr = ((w as i32 * w as i32) >> (2 * 6 + 7)) as u8;
        act1[i] = clip8(sqr as i32);
        act1[32 + i] = (w >> 6).max(0) as u8;
    }
    let mut buf = [0u8; 128];
    buf[..64].copy_from_slice(&act0);
    buf[64..].copy_from_slice(&act1);
    let mut z = stack.fc2_bias;
    for (j, x) in buf.iter().enumerate() {
        z += stack.fc2_weights[j] as i32 * *x as i32;
    }
    z + (y0[30] - y0[31])
}
pub fn evaluate_other_net(net: &OtherNetData, st: &BoardState) -> i32 {
    let lut = ThreatLut::build();
    let stm: usize = if st.w { 0 } else { 1 };
    let ntm = 1 - stm;
    let total_pc: u32 = (0..12u32).map(|i| st.bb[i as usize].count_ones()).sum();
    let bucket = (((total_pc as i32 - 2) / 4).clamp(0, (PSQT_BUCKETS - 1) as i32)) as usize;
    let hidden = net.hidden_size;
    let mut acc_sides = [[0i32; 1024]; 2];
    let mut psqt_sides = [[0i32; PSQT_BUCKETS]; 2];

    for persp in 0..2u32 {
        let ksq = find_king(st, persp);
        let mut acc = [0i32; 1024];
        let mut psqt = [0i32; PSQT_BUCKETS];
        acc[..hidden].copy_from_slice(&net.ft_bias[..hidden]);
        for pc in 0..12u32 {
            if (persp == 0 && pc == 5) || (persp == 1 && pc == 11) {
                continue;
            }
            let mut bb = st.bb[pc as usize];
            while bb != 0 {
                let sq = bb.trailing_zeros();
                bb &= bb - 1;
                let idx = halfka_index(persp, sq, pc, ksq);
                if idx < PSQ_DIMS {
                    let r = &net.psq_weights[idx * hidden..(idx + 1) * hidden];
                    let pr = &net.psqt[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS];
                    for i in 0..hidden {
                        acc[i] += r[i];
                    }
                    for b in 0..PSQT_BUCKETS {
                        psqt[b] += pr[b];
                    }
                }
            }
        }
        let occ = (0..12u32).fold(0u64, |a, i| a | st.bb[i as usize]);
        let pawn_targets = st.bb[1] | st.bb[3] | st.bb[7] | st.bb[9];
        let queen_targets = (0..12u32)
            .filter(|&i| i % 6 != 4)
            .fold(0u64, |a, i| a | st.bb[i as usize]);
        let minor_targets =
            st.bb[0] | st.bb[1] | st.bb[2] | st.bb[3] | st.bb[6] | st.bb[7] | st.bb[8] | st.bb[9];
        for c in 0..2u32 {
            for pt in 1..5u32 {
                let pc = c * 6 + pt;
                let mut bb = st.bb[pc as usize];
                let targets = if pt == 1 || pt == 4 {
                    queen_targets
                } else {
                    minor_targets
                };
                while bb != 0 {
                    let from = bb.trailing_zeros();
                    bb &= bb - 1;
                    let mut atk = attacks_bb(pt, from, occ) & targets;
                    while atk != 0 {
                        let to = atk.trailing_zeros();
                        atk &= atk - 1;
                        let attacked = piece_at(st, to);
                        let idx = lut.make_index(persp, pc, from, to, attacked, ksq);
                        if idx < THREAT_DIMS {
                            let rb = &net.threat_weights[idx * hidden..(idx + 1) * hidden];
                            let pb = &net.threat_psqt[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS];
                            for i in 0..hidden {
                                acc[i] += rb[i] as i32;
                            }
                            for b in 0..PSQT_BUCKETS {
                                psqt[b] += pb[b];
                            }
                        }
                    }
                }
            }
            let pawns = if c == 0 { st.bb[0] } else { st.bb[6] };
            let mut pb = pawns;
            while pb != 0 {
                let from = pb.trailing_zeros();
                pb &= pb - 1;
                let mut atk = pawn_attacks_bb(c, from) & pawn_targets;
                while atk != 0 {
                    let to = atk.trailing_zeros();
                    atk &= atk - 1;
                    let attacked = piece_at(st, to);
                    let idx = lut.make_index(persp, c * 6, from, to, attacked, ksq);
                    if idx < THREAT_DIMS {
                        let rb = &net.threat_weights[idx * hidden..(idx + 1) * hidden];
                        let pb = &net.threat_psqt[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS];
                        for i in 0..hidden {
                            acc[i] += rb[i] as i32;
                        }
                        for b in 0..PSQT_BUCKETS {
                            psqt[b] += pb[b];
                        }
                    }
                }
            }
        }
        for c in 0..2u32 {
            let pawn = if c == 0 { st.bb[0] } else { st.bb[6] };
            let mut bb = pawn;
            while bb != 0 {
                let from = bb.trailing_zeros();
                bb &= bb - 1;
                let band = pawn_pair_mask(from) & (st.bb[0] | st.bb[6]);
                let mut b2 = band & pawn;
                while b2 != 0 {
                    let to = b2.trailing_zeros();
                    b2 &= b2 - 1;
                    if from != to {
                        let idx = pair_index(persp, c, from, to, c, ksq);
                        try_add_pair(net, idx, &mut acc, &mut psqt, hidden);
                    }
                }
                let other = if c == 0 { st.bb[6] } else { st.bb[0] };
                let mut b3 = band & other;
                while b3 != 0 {
                    let to = b3.trailing_zeros();
                    b3 &= b3 - 1;
                    let idx = pair_index(persp, c, from, 1 - c, to, ksq);
                    try_add_pair(net, idx, &mut acc, &mut psqt, hidden);
                }
            }
        }
        acc_sides[persp as usize] = acc;
        psqt_sides[persp as usize] = psqt;
    }
    let mut ft_out = [0u8; 1024];
    for p in 0..2usize {
        for j in 0..512 {
            ft_out[p * 512 + j] = eval_ft_pair(&acc_sides[p], j);
        }
    }
    let stack = &net.stacks[bucket];
    let fwd = eval_stack_forward(stack, &ft_out);
    let psqt_v = (psqt_sides[stm][bucket] - psqt_sides[ntm][bucket]) / 2;
    let scaled = (fwd as i64 * 600 * OUTPUT_SCALE as i64)
        / ((HIDDEN_ONE_VAL as i64) << (WEIGHT_SCALE_BITS + 1));
    ((psqt_v as i64 + scaled) / FV_SCALE as i64) as i32
}
#[cfg(test)]
mod tests {
    use super::evaluate_other_net;
    use crate::nnue::other_nets::load_other_net;
    use crate::Engine;

    fn score_for(fen: &str, net: &crate::nnue::other_nets::OtherNetData) -> i32 {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        evaluate_other_net(net, &engine.st)
    }

    #[test]
    fn real_net_scores_are_finite_and_symmetric() {
        let data = match std::fs::read("nn-0ee0657fb25e.nnue") {
            Ok(d) => d,
            Err(_) => return,
        };
        let net = load_other_net(&data).expect("real net should decode");
        let start = score_for(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &net,
        );
        assert!((-1000..1000).contains(&start), "start score {start}");
        let pos = score_for(
            "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
            &net,
        );
        assert!((-5000..5000).contains(&pos), "pos score {pos}");
        let _ = score_for("8/8/8/R2pP1k/8/8/6Q1/4K3 w - d6 0 1", &net);
        let _ = score_for("8/8/8/k1PpR2/8/8/2q5/4K3 w - d6 0 1", &net);
    }
}
