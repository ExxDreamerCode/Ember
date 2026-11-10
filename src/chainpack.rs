use crate::board::{bit, piece_on, piece_type, BoardState, EMPTY_SQ};

pub const STEM_SIZE: usize = 32;
pub const CHUNK_MAGIC: &[u8; 4] = b"BINP";

const SCORE_VLE_BLOCK_SIZE: usize = 4;

const NIBBLE_EP_PAWN: u8 = 12;
const NIBBLE_WHITE_CASTLING_ROOK: u8 = 13;
const NIBBLE_BLACK_CASTLING_ROOK: u8 = 14;
const NIBBLE_BLACK_KING_TO_MOVE: u8 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedMoveKind {
    Normal,
    Castle,
    EnPassant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedMove {
    pub from: usize,
    pub to: usize,

    pub promotion: u8,
    pub kind: PackedMoveKind,
    pub is_null: bool,
}

impl PackedMove {
    pub const NULL: PackedMove = PackedMove {
        from: 0,
        to: 0,
        promotion: 0,
        kind: PackedMoveKind::Normal,
        is_null: true,
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainPos {
    pub bb: [u64; 12],
    pub white_stm: bool,

    pub ep: Option<usize>,

    pub castling: [bool; 4],
    pub rule50: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StemEntry {
    pub pos: ChainPos,
    pub mv: PackedMove,

    pub score: i16,
    pub ply: u16,

    pub result: i16,
    pub rule50: u16,
}

#[inline]
pub fn std_square(ember_sq: usize) -> usize {
    (7 - ember_sq / 8) * 8 + ember_sq % 8
}

#[inline]
pub fn ember_square(std_sq: usize) -> usize {
    sq(7 - std_sq / 8, std_sq % 8)
}

fn sq(r: usize, c: usize) -> usize {
    r * 8 + c
}

pub fn std_bitboards(st: &BoardState) -> [u64; 12] {
    let mut out = [0u64; 12];
    for (i, bb) in st.bb.iter().enumerate() {
        out[i] = bb.swap_bytes();
    }
    out
}

#[inline]
pub fn signed_to_unsigned(value: i16) -> u16 {
    let bits = value as u16;
    if bits & 0x8000 != 0 {
        (bits ^ 0x7FFF).rotate_left(1)
    } else {
        bits.rotate_left(1)
    }
}

#[inline]
pub fn unsigned_to_signed(value: u16) -> i16 {
    let mut v = value.rotate_right(1);
    if v & 0x8000 != 0 {
        v ^= 0x7FFF;
    }
    v as i16
}

#[inline]
fn used_bits_safe(n: u64) -> usize {
    if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros() as usize
    }
}

#[inline]
fn nth_set_bit_index(v: u64, n: u64) -> usize {
    let mut value = v;
    let mut count = n;
    while count > 0 {
        value &= value - 1;
        count -= 1;
    }
    value.trailing_zeros() as usize
}

#[inline]
fn before_mask(s: usize) -> u64 {
    (1u64 << s) - 1
}

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;

fn shift(state: u64, df: i32, dr: i32) -> u64 {
    let mut out = state;
    for _ in 0..df.max(0) {
        out = (out & !FILE_H) << 1;
    }
    for _ in 0..(-df).max(0) {
        out = (out & !FILE_A) >> 1;
    }
    if dr > 0 {
        out << (8 * dr)
    } else if dr < 0 {
        out >> (8 * -dr)
    } else {
        out
    }
}

fn king_attacks(s: usize) -> u64 {
    let b = bit(s);
    let mut out = shift(b, 1, 0) | shift(b, -1, 0);
    out |= shift(out | b, 0, 1) | shift(out | b, 0, -1);
    out
}

fn knight_attacks(s: usize) -> u64 {
    let b = bit(s);
    let mut out = shift(b, 1, 2) | shift(b, -1, 2) | shift(b, 2, 1) | shift(b, -2, 1);
    out |= shift(b, 1, -2) | shift(b, -1, -2) | shift(b, 2, -1) | shift(b, -2, -1);
    out
}

fn pawn_attacks(white: bool, s: usize) -> u64 {
    let b = bit(s);
    if white {
        shift(b, 1, 1) | shift(b, -1, 1)
    } else {
        shift(b, 1, -1) | shift(b, -1, -1)
    }
}

fn sliding_attacks(s: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
    let file = (s % 8) as i32;
    let rank = (s / 8) as i32;
    let mut out = 0u64;
    for &(df, dr) in dirs {
        let mut f = file + df;
        let mut r = rank + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let sq_i = (r * 8 + f) as usize;
            out |= bit(sq_i);
            if occ & bit(sq_i) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    out
}

fn rook_attacks(s: usize, occ: u64) -> u64 {
    sliding_attacks(s, occ, &[(1, 0), (-1, 0), (0, 1), (0, -1)])
}

fn bishop_attacks(s: usize, occ: u64) -> u64 {
    sliding_attacks(s, occ, &[(1, 1), (-1, 1), (1, -1), (-1, -1)])
}

#[derive(Debug, Default)]
struct BitWriter {
    movetext: Vec<u8>,
    bits_left: usize,
}

impl BitWriter {
    fn add_bits(&mut self, bits: u8, count: usize) {
        if count == 0 {
            return;
        }

        if self.bits_left == 0 {
            self.movetext.push(bits << (8 - count));
            self.bits_left = 8;
        } else if count <= self.bits_left {
            let last = self.movetext.len() - 1;
            self.movetext[last] |= bits << (self.bits_left - count);
        } else {
            let spill = count - self.bits_left;
            let last = self.movetext.len() - 1;
            self.movetext[last] |= bits >> spill;
            self.movetext.push(bits << (8 - spill));
            self.bits_left += 8;
        }

        self.bits_left -= count;
    }

    fn add_vle(&mut self, mut v: u16, block_size: usize) {
        let mask = (1u16 << block_size) - 1;
        loop {
            let block = ((v & mask) as u8) | (((v > mask) as u8) << block_size);
            self.add_bits(block, block_size + 1);
            v >>= block_size;
            if v == 0 {
                break;
            }
        }
    }
}

#[derive(Debug)]
struct BitReader<'a> {
    movetext: &'a [u8],
    bits_left: usize,
    offset: usize,
}

impl<'a> BitReader<'a> {
    fn new(movetext: &'a [u8]) -> Self {
        Self {
            movetext,
            bits_left: 8,
            offset: 0,
        }
    }

    fn extract_bits(&mut self, count: usize) -> Option<u8> {
        if count == 0 {
            return Some(0);
        }

        if self.bits_left == 0 {
            self.offset += 1;
            self.bits_left = 8;
        }
        if self.offset >= self.movetext.len() {
            return None;
        }

        let byte = self.movetext[self.offset] << (8 - self.bits_left);
        let mut bits = byte >> (8 - count);

        if count > self.bits_left {
            let spill = count - self.bits_left;
            if self.offset + 1 >= self.movetext.len() {
                return None;
            }
            bits |= self.movetext[self.offset + 1] >> (8 - spill);
            self.bits_left += 8;
            self.offset += 1;
        }

        self.bits_left -= count;
        Some(bits)
    }

    fn extract_vle(&mut self, block_size: usize) -> Option<u16> {
        let mask = (1u16 << block_size) - 1;
        let mut v = 0u16;
        let mut shift_by = 0;
        loop {
            let block = self.extract_bits(block_size + 1)? as u16;
            v |= (block & mask) << shift_by;
            if (block >> block_size) == 0 {
                break;
            }
            shift_by += block_size;
        }
        Some(v)
    }

    fn num_read_bytes(&self) -> usize {
        self.offset + (self.bits_left != 8) as usize
    }
}

impl ChainPos {
    pub fn from_ember(st: &BoardState) -> Self {
        Self {
            bb: std_bitboards(st),
            white_stm: st.w,
            ep: st.ep.map(std_square),
            castling: st.cr,
            rule50: st.halfmove_clock as u16,
        }
    }

    fn our_pieces(&self) -> u64 {
        let (lo, hi) = if self.white_stm { (0, 6) } else { (6, 12) };
        let mut out = 0u64;
        for bb in &self.bb[lo..hi] {
            out |= *bb;
        }
        out
    }

    fn their_pieces(&self) -> u64 {
        let (lo, hi) = if self.white_stm { (6, 12) } else { (0, 6) };
        let mut out = 0u64;
        for bb in &self.bb[lo..hi] {
            out |= *bb;
        }
        out
    }

    fn piece_at(&self, s: usize) -> u8 {
        piece_on(&self.bb, s)
    }

    pub fn classify_move(&self, from: usize, to: usize, promotion: u8) -> PackedMove {
        let pi = self.piece_at(from);
        debug_assert!(pi != EMPTY_SQ, "classify_move on empty square {from}");
        let pt = piece_type(pi);
        let white_mover = pi < 6;
        let king_att = king_attacks(from);

        if pt == 5
            && (king_att & bit(to) == 0
                || (self.bb[if white_mover { 3 } else { 9 }] & bit(to) != 0))
        {
            let from_file = from % 8;
            let to_file = to % 8;
            let rank_base = from / 8 * 8;
            let rook_sq = if to_file == 6 && from_file <= 4 {
                rank_base + 7
            } else if to_file == 2 && from_file >= 4 {
                rank_base
            } else {
                to
            };
            return PackedMove {
                from,
                to: rook_sq,
                promotion: 0,
                kind: PackedMoveKind::Castle,
                is_null: false,
            };
        }

        let kind = if pt == 0 && from % 8 != to % 8 && self.ep == Some(to) {
            PackedMoveKind::EnPassant
        } else {
            PackedMoveKind::Normal
        };

        PackedMove {
            from,
            to,
            promotion,
            kind,
            is_null: false,
        }
    }

    pub fn packed_move_from_uci(&self, uci: &str) -> Option<PackedMove> {
        let b = uci.as_bytes();
        if uci.len() < 4 || uci == "0000" {
            return Some(PackedMove::NULL);
        }
        let from = (b[1] - b'1') as usize * 8 + (b[0] - b'a') as usize;
        let to = (b[3] - b'1') as usize * 8 + (b[2] - b'a') as usize;
        let promotion = if uci.len() >= 5 {
            match b[4] {
                b'q' | b'Q' => 4,
                b'r' | b'R' => 3,
                b'b' | b'B' => 2,
                b'n' | b'N' => 1,
                _ => 0,
            }
        } else {
            0
        };
        Some(self.classify_move(from, to, promotion))
    }

    pub fn do_move(&mut self, mv: &PackedMove) {
        if mv.is_null {
            self.ep = None;
            self.rule50 = self.rule50.saturating_add(1);
            self.white_stm = !self.white_stm;
            return;
        }

        let from = mv.from;
        let to = mv.to;
        let pi = self.piece_at(from);
        debug_assert!(pi != EMPTY_SQ, "do_move on empty square {from}");
        let pt = piece_type(pi);
        let white_mover = pi < 6;

        let mut is_capture = false;

        match mv.kind {
            PackedMoveKind::Castle => {
                let short = to % 8 == 7;
                let king_to = from / 8 * 8 + if short { 6 } else { 2 };
                let rook_to = from / 8 * 8 + if short { 5 } else { 3 };
                let rook_pi = self.piece_at(to);
                debug_assert!(rook_pi != EMPTY_SQ && piece_type(rook_pi) == 3);
                self.bb[rook_pi as usize] &= !bit(to);
                self.bb[rook_pi as usize] |= bit(rook_to);
                self.bb[pi as usize] &= !bit(from);
                self.bb[pi as usize] |= bit(king_to);
            }
            PackedMoveKind::EnPassant => {
                let captured_sq = to ^ 8;
                let captured = self.piece_at(captured_sq);
                debug_assert!(captured != EMPTY_SQ && piece_type(captured) == 0);
                self.bb[captured as usize] &= !bit(captured_sq);
                self.bb[pi as usize] &= !bit(from);
                self.bb[pi as usize] |= bit(to);
                is_capture = true;
            }
            PackedMoveKind::Normal => {
                let captured = self.piece_at(to);
                if captured != EMPTY_SQ {
                    debug_assert!(
                        (captured < 6) != white_mover,
                        "normal move captured own piece"
                    );
                    self.bb[captured as usize] &= !bit(to);
                    is_capture = true;
                }
                self.bb[pi as usize] &= !bit(from);
                if mv.promotion != 0 {
                    let promo_pi = (if white_mover { 0 } else { 6 }) + (mv.promotion - 1);
                    self.bb[promo_pi as usize] |= bit(to);
                } else {
                    self.bb[pi as usize] |= bit(to);
                }
            }
        }

        if pt == 0 || is_capture {
            self.rule50 = 0;
        } else {
            self.rule50 = self.rule50.saturating_add(1);
        }

        if pt == 5 {
            let (short_idx, long_idx) = if white_mover { (0, 1) } else { (2, 3) };
            self.castling[short_idx] = false;
            self.castling[long_idx] = false;
        }
        for (corner, idx) in [(7, 0usize), (0, 1), (63, 2), (56, 3)] {
            if from == corner || to == corner {
                self.castling[idx] = false;
            }
        }

        self.ep = if pt == 0 && to.abs_diff(from) == 16 {
            Some((from + to) / 2)
        } else {
            None
        };

        self.white_stm = !self.white_stm;
    }

    pub fn from_stem(stem: &[u8; STEM_SIZE]) -> Option<Self> {
        let mut occupied = 0u64;
        for (i, b) in stem[0..8].iter().enumerate() {
            occupied |= (*b as u64) << (8 * (7 - i));
        }

        let nibble_at = |idx: usize| -> u8 {
            let byte = stem[8 + idx / 2];
            if idx.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            }
        };

        let mut black_to_move = false;
        let mut count = 0usize;
        let mut occ = occupied;
        while occ != 0 {
            occ &= occ - 1;
            if nibble_at(count) == NIBBLE_BLACK_KING_TO_MOVE {
                black_to_move = true;
            }
            count += 1;
        }

        let mut pos = ChainPos {
            bb: [0u64; 12],
            white_stm: !black_to_move,
            ep: None,
            castling: [false; 4],
            rule50: u16::from_be_bytes([stem[30], stem[31]]),
        };

        let mut idx = 0usize;
        occ = occupied;
        while occ != 0 {
            let s = occ.trailing_zeros() as usize;
            occ &= occ - 1;
            let nibble = nibble_at(idx);
            idx += 1;
            match nibble {
                0..=11 => {
                    let pi = (nibble / 2) + (nibble % 2) * 6;
                    pos.bb[pi as usize] |= bit(s);
                }
                NIBBLE_EP_PAWN => {
                    let (pi, target) = if black_to_move {
                        (0usize, s.checked_sub(8))
                    } else {
                        (6usize, s.checked_add(8))
                    };
                    pos.bb[pi] |= bit(s);
                    if let Some(target) = target {
                        if target < 64 {
                            pos.ep = Some(target);
                        }
                    }
                }
                NIBBLE_WHITE_CASTLING_ROOK => {
                    pos.bb[3] |= bit(s);
                    match s {
                        7 => pos.castling[0] = true,
                        0 => pos.castling[1] = true,
                        _ => {}
                    }
                }
                NIBBLE_BLACK_CASTLING_ROOK => {
                    pos.bb[9] |= bit(s);
                    match s {
                        63 => pos.castling[2] = true,
                        56 => pos.castling[3] = true,
                        _ => {}
                    }
                }
                NIBBLE_BLACK_KING_TO_MOVE => {
                    pos.bb[11] |= bit(s);
                }
                _ => return None,
            }
        }

        Some(pos)
    }

    fn encode_move(&self, mv: &PackedMove) -> (u8, usize, u32, usize) {
        let our = self.our_pieces();
        let their = self.their_pieces();
        let occupied = our | their;

        let num_pieces = our.count_ones() as u64;
        let piece_id = (our & before_mask(mv.from)).count_ones() as u8;

        let pi = self.piece_at(mv.from);
        let pt = piece_type(pi);
        let white = self.white_stm;

        let (move_id, num_moves): (u32, u64) = match pt {
            0 => {
                let mut targets = their;
                if let Some(ep) = self.ep {
                    targets |= bit(ep);
                }
                let mut destinations = pawn_attacks(white, mv.from) & targets;

                let forward: i32 = if white { 8 } else { -8 };
                let sq_forward = mv.from as i32 + forward;
                if (0..64).contains(&sq_forward) && occupied & bit(sq_forward as usize) == 0 {
                    destinations |= bit(sq_forward as usize);
                    let start_rank = if white { 1 } else { 6 };
                    if mv.from / 8 == start_rank {
                        let sq_forward2 = sq_forward + forward;
                        if (0..64).contains(&sq_forward2)
                            && occupied & bit(sq_forward2 as usize) == 0
                        {
                            destinations |= bit(sq_forward2 as usize);
                        }
                    }
                }

                let mut id = (destinations & before_mask(mv.to)).count_ones();
                let mut num = destinations.count_ones() as u64;
                let promotion_rank = if white { 6 } else { 1 };
                if mv.from / 8 == promotion_rank {
                    id = id * 4 + u32::from(mv.promotion - 1);
                    num *= 4;
                }
                (id, num)
            }

            5 => {
                let attacks = king_attacks(mv.from) & !our;
                let attacks_size = attacks.count_ones();
                let (short_idx, long_idx) = if white { (0, 1) } else { (2, 3) };
                let num_castlings =
                    self.castling[short_idx] as u64 + self.castling[long_idx] as u64;
                let num = attacks_size as u64 + num_castlings;

                if mv.kind == PackedMoveKind::Castle {
                    let long_rights = self.castling[long_idx];
                    let short_castle = mv.to % 8 == 7;
                    let mut id = attacks_size - 1;
                    if long_rights {
                        id += 1;
                    }
                    if short_castle {
                        id += 1;
                    }
                    (id, num)
                } else {
                    let id = (attacks & before_mask(mv.to)).count_ones();
                    (id, num)
                }
            }
            1 => {
                let attacks = knight_attacks(mv.from) & !our;
                let id = (attacks & before_mask(mv.to)).count_ones();
                (id, attacks.count_ones() as u64)
            }
            2 => {
                let attacks = bishop_attacks(mv.from, occupied) & !our;
                let id = (attacks & before_mask(mv.to)).count_ones();
                (id, attacks.count_ones() as u64)
            }
            3 => {
                let attacks = rook_attacks(mv.from, occupied) & !our;
                let id = (attacks & before_mask(mv.to)).count_ones();
                (id, attacks.count_ones() as u64)
            }
            _ => {
                let attacks =
                    (rook_attacks(mv.from, occupied) | bishop_attacks(mv.from, occupied)) & !our;
                let id = (attacks & before_mask(mv.to)).count_ones();
                (id, attacks.count_ones() as u64)
            }
        };

        (
            piece_id,
            used_bits_safe(num_pieces),
            move_id,
            used_bits_safe(num_moves),
        )
    }

    fn decode_move(&self, reader: &mut BitReader) -> Option<PackedMove> {
        let our = self.our_pieces();

        let piece_id = reader.extract_bits(used_bits_safe(our.count_ones() as u64))?;
        let from = nth_set_bit_index(our, piece_id as u64);

        let pi = self.piece_at(from);
        if pi == EMPTY_SQ {
            return None;
        }
        let pt = piece_type(pi);
        let white = self.white_stm;

        let mv = match pt {
            0 => {
                let mut targets = self.their_pieces();
                if let Some(ep) = self.ep {
                    targets |= bit(ep);
                }
                let mut destinations = pawn_attacks(white, from) & targets;

                let forward: i32 = if white { 8 } else { -8 };
                let sq_forward = from as i32 + forward;
                let occupied = our | self.their_pieces();
                if (0..64).contains(&sq_forward) && occupied & bit(sq_forward as usize) == 0 {
                    destinations |= bit(sq_forward as usize);
                    let start_rank = if white { 1 } else { 6 };
                    if from / 8 == start_rank {
                        let sq_forward2 = sq_forward + forward;
                        if (0..64).contains(&sq_forward2)
                            && occupied & bit(sq_forward2 as usize) == 0
                        {
                            destinations |= bit(sq_forward2 as usize);
                        }
                    }
                }

                let promotion_rank = if white { 6 } else { 1 };
                if from / 8 == promotion_rank {
                    let id = reader
                        .extract_bits(used_bits_safe(destinations.count_ones() as u64 * 4))?
                        as usize;
                    let promotion = (id % 4 + 1) as u8;
                    let to = nth_set_bit_index(destinations, (id / 4) as u64);
                    PackedMove {
                        from,
                        to,
                        promotion,
                        kind: PackedMoveKind::Normal,
                        is_null: false,
                    }
                } else {
                    let id = reader
                        .extract_bits(used_bits_safe(destinations.count_ones() as u64))?
                        as usize;
                    let to = nth_set_bit_index(destinations, id as u64);
                    let kind = if self.ep == Some(to) && from % 8 != to % 8 {
                        PackedMoveKind::EnPassant
                    } else {
                        PackedMoveKind::Normal
                    };
                    PackedMove {
                        from,
                        to,
                        promotion: 0,
                        kind,
                        is_null: false,
                    }
                }
            }
            5 => {
                let attacks = king_attacks(from) & !our;
                let attacks_size = attacks.count_ones();
                let (short_idx, long_idx) = if white { (0, 1) } else { (2, 3) };
                let num_castlings =
                    self.castling[short_idx] as u64 + self.castling[long_idx] as u64;
                let id = reader.extract_bits(used_bits_safe(attacks_size as u64 + num_castlings))?
                    as usize;

                if id as u32 >= attacks_size {
                    let idx = id as u32 - attacks_size;
                    let short = !(idx == 0 && self.castling[long_idx]);
                    let rank_base = from / 8 * 8;
                    PackedMove {
                        from,
                        to: rank_base + if short { 7 } else { 0 },
                        promotion: 0,
                        kind: PackedMoveKind::Castle,
                        is_null: false,
                    }
                } else {
                    let to = nth_set_bit_index(attacks, id as u64);
                    PackedMove {
                        from,
                        to,
                        promotion: 0,
                        kind: PackedMoveKind::Normal,
                        is_null: false,
                    }
                }
            }
            1 => {
                let attacks = knight_attacks(from) & !our;
                let id = reader.extract_bits(used_bits_safe(attacks.count_ones() as u64))? as usize;
                let to = nth_set_bit_index(attacks, id as u64);
                PackedMove {
                    from,
                    to,
                    promotion: 0,
                    kind: PackedMoveKind::Normal,
                    is_null: false,
                }
            }
            2 => {
                let attacks = bishop_attacks(from, our | self.their_pieces()) & !our;
                let id = reader.extract_bits(used_bits_safe(attacks.count_ones() as u64))? as usize;
                let to = nth_set_bit_index(attacks, id as u64);
                PackedMove {
                    from,
                    to,
                    promotion: 0,
                    kind: PackedMoveKind::Normal,
                    is_null: false,
                }
            }
            3 => {
                let attacks = rook_attacks(from, our | self.their_pieces()) & !our;
                let id = reader.extract_bits(used_bits_safe(attacks.count_ones() as u64))? as usize;
                let to = nth_set_bit_index(attacks, id as u64);
                PackedMove {
                    from,
                    to,
                    promotion: 0,
                    kind: PackedMoveKind::Normal,
                    is_null: false,
                }
            }
            _ => {
                let occ = our | self.their_pieces();
                let attacks = (rook_attacks(from, occ) | bishop_attacks(from, occ)) & !our;
                let id = reader.extract_bits(used_bits_safe(attacks.count_ones() as u64))? as usize;
                let to = nth_set_bit_index(attacks, id as u64);
                PackedMove {
                    from,
                    to,
                    promotion: 0,
                    kind: PackedMoveKind::Normal,
                    is_null: false,
                }
            }
        };

        Some(mv)
    }
}

fn pack_position(pos: &ChainPos, out: &mut [u8; STEM_SIZE]) {
    let mut occupied = 0u64;
    for bb in &pos.bb {
        occupied |= *bb;
    }
    out[0..8].copy_from_slice(&occupied.to_be_bytes());

    let mut packed_state = [0u8; 16];
    let mut nibble_idx = 0usize;
    let mut occ = occupied;
    while occ != 0 {
        let s = occ.trailing_zeros() as usize;
        occ &= occ - 1;

        let pi = piece_on(&pos.bb, s);
        let pt = piece_type(pi);
        let mut nibble = (pi % 6) * 2 + (pi / 6);

        if pi == 11 && !pos.white_stm {
            nibble = NIBBLE_BLACK_KING_TO_MOVE;
        } else if pt == 0 && pos.ep_is_ep_pawn(s) {
            nibble = NIBBLE_EP_PAWN;
        } else if pt == 3 {
            if pos.castling[0] && s == 7 || pos.castling[1] && s == 0 {
                nibble = NIBBLE_WHITE_CASTLING_ROOK;
            } else if pos.castling[2] && s == 63 || pos.castling[3] && s == 56 {
                nibble = NIBBLE_BLACK_CASTLING_ROOK;
            }
        }

        if nibble_idx.is_multiple_of(2) {
            packed_state[nibble_idx / 2] = nibble;
        } else {
            packed_state[nibble_idx / 2] |= nibble << 4;
        }
        nibble_idx += 1;
    }

    out[8..24].copy_from_slice(&packed_state);
}

impl ChainPos {
    fn ep_is_ep_pawn(&self, s: usize) -> bool {
        let Some(ep) = self.ep else {
            return false;
        };

        if self.white_stm {
            ep >= 8 && piece_type(self.piece_at(ep - 8)) == 0 && ep - 8 == s
        } else {
            ep + 8 < 64 && piece_type(self.piece_at(ep + 8)) == 0 && ep + 8 == s
        }
    }
}

pub fn pack_stem(pos: &ChainPos, mv: &PackedMove, score: i16, ply: u16, result: i16) -> [u8; 32] {
    let mut out = [0u8; STEM_SIZE];
    pack_position(pos, &mut out);

    let kind_bits: u16 = if mv.promotion != 0 {
        1
    } else {
        match mv.kind {
            PackedMoveKind::Normal => 0,
            PackedMoveKind::Castle => 2,
            PackedMoveKind::EnPassant => 3,
        }
    };
    let packed_move = if mv.is_null {
        0u16
    } else {
        (kind_bits << 14)
            | ((mv.from as u16) << 8)
            | ((mv.to as u16) << 2)
            | (if mv.promotion != 0 {
                (mv.promotion - 1) as u16
            } else {
                0
            })
    };
    out[24..26].copy_from_slice(&packed_move.to_be_bytes());

    out[26..28].copy_from_slice(&signed_to_unsigned(score).to_be_bytes());

    let pr = ((signed_to_unsigned(result) & 0b11) << 14) | (ply & 0x3FFF);
    out[28..30].copy_from_slice(&pr.to_be_bytes());

    out[30..32].copy_from_slice(&pos.rule50.to_be_bytes());
    out
}

pub fn unpack_stem(stem: &[u8; STEM_SIZE]) -> Option<StemEntry> {
    let pos = ChainPos::from_stem(stem)?;

    let packed_move = u16::from_be_bytes([stem[24], stem[25]]);
    let mv = if packed_move == 0 {
        PackedMove::NULL
    } else {
        let kind_bits = packed_move >> 14;
        let kind = match kind_bits {
            0 | 1 => PackedMoveKind::Normal,
            2 => PackedMoveKind::Castle,
            3 => PackedMoveKind::EnPassant,
            _ => return None,
        };
        let from = ((packed_move >> 8) & 0x3F) as usize;
        let to = ((packed_move >> 2) & 0x3F) as usize;
        let promotion = if kind_bits == 1 {
            ((packed_move & 0b11) + 1) as u8
        } else {
            0
        };
        PackedMove {
            from,
            to,
            promotion,
            kind,
            is_null: false,
        }
    };

    let score = unsigned_to_signed(u16::from_be_bytes([stem[26], stem[27]]));
    let pr = u16::from_be_bytes([stem[28], stem[29]]);
    let ply = pr & 0x3FFF;
    let result = unsigned_to_signed(pr >> 14);
    let rule50 = u16::from_be_bytes([stem[30], stem[31]]);

    Some(StemEntry {
        pos,
        mv,
        score,
        ply,
        result,
        rule50,
    })
}

#[derive(Debug)]
pub struct MoveScoreListWriter {
    num_plies: u16,
    writer: BitWriter,
    last_score: i16,
}

impl MoveScoreListWriter {
    pub fn new(head_score: i16) -> Self {
        Self {
            num_plies: 0,
            writer: BitWriter::default(),
            last_score: head_score.wrapping_neg(),
        }
    }

    pub fn add_move(&mut self, pos: &ChainPos, mv: &PackedMove, score: i16) {
        debug_assert!(!mv.is_null, "null moves cannot be chained");

        let (piece_id, piece_bits, move_id, move_bits) = pos.encode_move(mv);
        self.writer.add_bits(piece_id, piece_bits);
        self.writer.add_bits(move_id as u8, move_bits);

        let delta = score.wrapping_sub(self.last_score);
        self.writer
            .add_vle(signed_to_unsigned(delta), SCORE_VLE_BLOCK_SIZE);
        self.last_score = score.wrapping_neg();

        self.num_plies += 1;
    }

    pub fn num_plies(&self) -> u16 {
        self.num_plies
    }

    pub fn movetext(&self) -> &[u8] {
        &self.writer.movetext
    }

    pub fn into_movetext(self) -> Vec<u8> {
        self.writer.movetext
    }
}

#[derive(Debug)]
pub struct MoveScoreListReader<'a> {
    head: StemEntry,
    reader: BitReader<'a>,
    last_score: i16,
    num_plies: u16,
    num_read_plies: u16,
}

impl<'a> MoveScoreListReader<'a> {
    pub fn new(head: &StemEntry, movetext: &'a [u8], num_plies: u16) -> Self {
        Self {
            head: head.clone(),
            reader: BitReader::new(movetext),
            last_score: head.score.wrapping_neg(),
            num_plies,
            num_read_plies: 0,
        }
    }

    pub fn has_next(&self) -> bool {
        self.num_read_plies < self.num_plies
    }

    pub fn next_entry(&mut self) -> Option<StemEntry> {
        self.head.pos.do_move(&self.head.mv);
        let mv = self.head.pos.decode_move(&mut self.reader)?;
        let delta = unsigned_to_signed(self.reader.extract_vle(SCORE_VLE_BLOCK_SIZE)?);
        let score = self.last_score.wrapping_add(delta);
        self.last_score = score.wrapping_neg();

        self.head.mv = mv;
        self.head.score = score;
        self.head.ply += 1;
        self.head.result = -self.head.result;
        self.head.rule50 = self.head.pos.rule50;
        self.num_read_plies += 1;

        Some(self.head.clone())
    }

    pub fn num_read_bytes(&self) -> usize {
        self.reader.num_read_bytes()
    }
}

pub fn unpack_chain(
    stem: &[u8; STEM_SIZE],
    num_plies: u16,
    movetext: &[u8],
) -> Option<Vec<StemEntry>> {
    let head = unpack_stem(stem)?;
    let mut entries = vec![head.clone()];
    let mut reader = MoveScoreListReader::new(&head, movetext, num_plies);
    while reader.has_next() {
        entries.push(reader.next_entry()?);
    }
    Some(entries)
}

pub fn scan_single_chain(payload: &[u8]) -> Option<(u64, usize)> {
    if payload.len() < STEM_SIZE + 2 {
        return None;
    }
    let mut stem = [0u8; STEM_SIZE];
    stem.copy_from_slice(&payload[..STEM_SIZE]);
    let num_plies = u16::from_be_bytes([payload[STEM_SIZE], payload[STEM_SIZE + 1]]);

    let head = unpack_stem(&stem)?;
    let mut reader = MoveScoreListReader::new(&head, &payload[STEM_SIZE + 2..], num_plies);
    while reader.has_next() {
        reader.next_entry()?;
    }
    Some((
        1 + num_plies as u64,
        STEM_SIZE + 2 + reader.num_read_bytes(),
    ))
}

pub fn scan_chunk(payload: &[u8]) -> Option<(u64, usize)> {
    let mut offset = 0usize;
    let mut entries = 0u64;
    while offset < payload.len() {
        let (chain_entries, consumed) = scan_single_chain(&payload[offset..])?;
        entries += chain_entries;
        offset += consumed;
    }
    Some((entries, offset))
}
