use pyrrhic_rs::{Color, DtzProbeValue, EngineAdapter, Piece, TableBases, WdlProbeResult};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use crate::board::{
    move_from, move_promotion, move_to, BoardState, Move, BB, BK, BN, BP, BQ, BR, KING_ATTACKS,
    KNIGHT_ATTACKS, MAX_PLY, TB_WIN_SCORE, WB, WK, WN, WP, WQ, WR,
};
use crate::magic::{bishop_attacks, rook_attacks};

fn exact_search_score(wdl: WdlProbeResult, ply: usize) -> i32 {
    let ply = ply.min(MAX_PLY) as i32;
    match wdl {
        WdlProbeResult::Win => TB_WIN_SCORE - ply,
        WdlProbeResult::Loss => -TB_WIN_SCORE + ply,
        WdlProbeResult::Draw | WdlProbeResult::CursedWin | WdlProbeResult::BlessedLoss => 0,
    }
}

fn wdl_from_dtz(dtz: i32, halfmoves: u8) -> Option<WdlProbeResult> {
    if dtz == 0 {
        return Some(WdlProbeResult::Draw);
    }
    let distance = dtz.unsigned_abs();
    let remaining = u32::from(halfmoves) + distance;
    // The fork returns signed DTZ without a precise/rounded flag. Withhold
    // exact bounds for totals 99..=101, including some known outcomes, rather
    // than resolve the fifty-move boundary from an unqualified distance.
    // This is deliberately more conservative than the old precision-aware
    // backend. Root move selection still uses the fork's own selector.
    if (99..=101).contains(&remaining) {
        return None;
    }
    Some(match (dtz > 0, distance <= 100 && remaining < 100) {
        (true, true) => WdlProbeResult::Win,
        (false, true) => WdlProbeResult::Loss,
        (true, false) => WdlProbeResult::CursedWin,
        (false, false) => WdlProbeResult::BlessedLoss,
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
struct MaterialSideKey {
    total: u8,
    king: u8,
    queen: u8,
    rook: u8,
    bishop: u8,
    knight: u8,
    pawn: u8,
}

impl MaterialSideKey {
    fn from_part(part: &str) -> Option<Self> {
        let mut side = Self::empty();
        for ch in part.bytes() {
            match ch {
                b'K' => side.king += 1,
                b'Q' => side.queen += 1,
                b'R' => side.rook += 1,
                b'B' => side.bishop += 1,
                b'N' => side.knight += 1,
                b'P' => side.pawn += 1,
                _ => return None,
            }
            side.total += 1;
        }
        Some(side)
    }

    fn from_board(st: &BoardState, white: bool) -> Self {
        let offset = if white { 0 } else { 6 };
        Self {
            total: (0..6).map(|i| st.bb[offset + i].count_ones() as u8).sum(),
            pawn: st.bb[offset].count_ones() as u8,
            knight: st.bb[offset + 1].count_ones() as u8,
            bishop: st.bb[offset + 2].count_ones() as u8,
            rook: st.bb[offset + 3].count_ones() as u8,
            queen: st.bb[offset + 4].count_ones() as u8,
            king: st.bb[offset + 5].count_ones() as u8,
        }
    }

    const fn empty() -> Self {
        Self {
            total: 0,
            king: 0,
            queen: 0,
            rook: 0,
            bishop: 0,
            knight: 0,
            pawn: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MaterialKey {
    stronger: MaterialSideKey,
    weaker: MaterialSideKey,
}

impl MaterialKey {
    fn new(white: MaterialSideKey, black: MaterialSideKey) -> Self {
        if white < black {
            Self {
                stronger: black,
                weaker: white,
            }
        } else {
            Self {
                stronger: white,
                weaker: black,
            }
        }
    }

    fn from_board(st: &BoardState) -> Self {
        Self::new(
            MaterialSideKey::from_board(st, true),
            MaterialSideKey::from_board(st, false),
        )
    }

    fn from_stem(stem: &str) -> Option<Self> {
        let (white, black) = stem.split_once('v')?;
        Some(Self::new(
            MaterialSideKey::from_part(white)?,
            MaterialSideKey::from_part(black)?,
        ))
    }
}

#[derive(Debug, Default)]
struct SyzygyCapabilities {
    max_pieces: u32,
    wdl_materials: HashSet<MaterialKey>,
    dtz_materials: HashSet<MaterialKey>,
}

impl SyzygyCapabilities {
    fn from_tables(tables: &TableBases<EmberAdapter>) -> Self {
        let mut wdl_materials = HashSet::new();
        let mut dtz_materials = HashSet::new();
        for (name, has_dtz) in tables.materials() {
            let Some(material) = MaterialKey::from_stem(&name) else {
                continue;
            };
            wdl_materials.insert(material);
            if has_dtz {
                dtz_materials.insert(material);
            }
        }
        Self {
            max_pieces: tables.max_pieces(),
            wdl_materials,
            dtz_materials,
        }
    }
}

#[derive(Clone)]
struct EmberAdapter;

impl EngineAdapter for EmberAdapter {
    fn pawn_attacks(color: Color, square: u64) -> u64 {
        let bit = 1u64 << ((square as usize) ^ 56);
        let attacks = if color == Color::White {
            (bit & !0x8080_8080_8080_8080) >> 7 | (bit & !0x0101_0101_0101_0101) >> 9
        } else {
            (bit & !0x0101_0101_0101_0101) << 7 | (bit & !0x8080_8080_8080_8080) << 9
        };
        attacks.swap_bytes()
    }

    fn knight_attacks(square: u64) -> u64 {
        KNIGHT_ATTACKS[(square as usize) ^ 56].swap_bytes()
    }

    fn bishop_attacks(square: u64, occupied: u64) -> u64 {
        bishop_attacks((square as usize) ^ 56, occupied.swap_bytes()).swap_bytes()
    }

    fn rook_attacks(square: u64, occupied: u64) -> u64 {
        rook_attacks((square as usize) ^ 56, occupied.swap_bytes()).swap_bytes()
    }

    fn queen_attacks(square: u64, occupied: u64) -> u64 {
        Self::bishop_attacks(square, occupied) | Self::rook_attacks(square, occupied)
    }

    fn king_attacks(square: u64) -> u64 {
        KING_ATTACKS[(square as usize) ^ 56].swap_bytes()
    }
}

#[derive(Clone, Copy)]
struct ProbePosition {
    white: u64,
    black: u64,
    kings: u64,
    queens: u64,
    rooks: u64,
    bishops: u64,
    knights: u64,
    pawns: u64,
    ep: u32,
    turn: bool,
    halfmoves: u8,
}

impl ProbePosition {
    fn from_board(st: &BoardState) -> Self {
        let flip = |bb: u64| bb.swap_bytes();
        Self {
            white: flip(st.bb[WP] | st.bb[WN] | st.bb[WB] | st.bb[WR] | st.bb[WQ] | st.bb[WK]),
            black: flip(st.bb[BP] | st.bb[BN] | st.bb[BB] | st.bb[BR] | st.bb[BQ] | st.bb[BK]),
            kings: flip(st.bb[WK] | st.bb[BK]),
            queens: flip(st.bb[WQ] | st.bb[BQ]),
            rooks: flip(st.bb[WR] | st.bb[BR]),
            bishops: flip(st.bb[WB] | st.bb[BB]),
            knights: flip(st.bb[WN] | st.bb[BN]),
            pawns: flip(st.bb[WP] | st.bb[BP]),
            ep: st.ep.map_or(0, |sq| (sq ^ 56) as u32),
            turn: st.w,
            halfmoves: st.halfmove_clock,
        }
    }

    fn probe_wdl(self, tb: &TableBases<EmberAdapter>) -> Option<WdlProbeResult> {
        tb.probe_wdl(
            self.white,
            self.black,
            self.kings,
            self.queens,
            self.rooks,
            self.bishops,
            self.knights,
            self.pawns,
            self.ep,
            self.turn,
        )
        .ok()
    }

    fn probe_dtz(self, tb: &TableBases<EmberAdapter>) -> Option<i32> {
        tb.probe_dtz(
            self.white,
            self.black,
            self.kings,
            self.queens,
            self.rooks,
            self.bishops,
            self.knights,
            self.pawns,
            self.ep,
            self.turn,
        )
        .ok()
    }

    fn probe_root(self, tb: &TableBases<EmberAdapter>) -> Option<pyrrhic_rs::DtzProbeResult> {
        tb.probe_root(
            self.white,
            self.black,
            self.kings,
            self.queens,
            self.rooks,
            self.bishops,
            self.knights,
            self.pawns,
            self.halfmoves.into(),
            self.ep,
            self.turn,
        )
        .ok()
    }
}

#[derive(Clone)]
pub struct SyzygyTables {
    generation: Option<Arc<SyzygyGeneration>>,
}

struct SyzygyGeneration {
    tables: TableBases<EmberAdapter>,
    capabilities: SyzygyCapabilities,
}

impl Default for SyzygyTables {
    fn default() -> Self {
        Self::new()
    }
}

impl SyzygyTables {
    pub fn new() -> Self {
        SyzygyTables { generation: None }
    }

    pub fn load(&mut self, path: &str) -> Result<(), String> {
        if path.is_empty() || path.to_lowercase() == "<empty>" {
            self.generation = None;
            return Ok(());
        }
        let directory = Path::new(path);
        if !directory.is_dir() {
            return Err(format!(
                "Syzygy directory not found: {}",
                directory.display()
            ));
        }
        let tables =
            TableBases::new(path).map_err(|e| format!("Failed to load Syzygy tables: {e:?}"))?;
        let capabilities = SyzygyCapabilities::from_tables(&tables);
        self.generation = Some(Arc::new(SyzygyGeneration {
            tables,
            capabilities,
        }));
        Ok(())
    }

    pub fn max_pieces(&self) -> u32 {
        self.generation
            .as_ref()
            .map_or(0, |generation| generation.capabilities.max_pieces)
    }

    pub fn is_loaded(&self) -> bool {
        self.generation.is_some()
    }

    pub fn piece_count(st: &BoardState) -> u32 {
        (0..12).map(|i| st.bb[i].count_ones()).sum()
    }

    pub fn pieces_ok(st: &BoardState) -> bool {
        Self::piece_count(st) >= 2
            && st.bb[crate::board::WK].count_ones() == 1
            && st.bb[crate::board::BK].count_ones() == 1
            && st.cr.iter().all(|&right| !right)
    }

    pub fn can_probe_wdl(&self, st: &BoardState) -> bool {
        let Some(generation) = &self.generation else {
            return false;
        };
        if !Self::pieces_ok(st) {
            return false;
        }
        let piece_count = Self::piece_count(st);
        if piece_count == 2 {
            return true;
        }
        piece_count <= self.max_pieces()
            && generation
                .capabilities
                .wdl_materials
                .contains(&MaterialKey::from_board(st))
    }

    pub fn can_probe_dtz(&self, st: &BoardState) -> bool {
        let Some(generation) = &self.generation else {
            return false;
        };
        if !Self::pieces_ok(st) {
            return false;
        }
        let piece_count = Self::piece_count(st);
        if piece_count == 2 {
            return true;
        }
        piece_count <= self.max_pieces() && {
            let material = MaterialKey::from_board(st);
            generation.capabilities.wdl_materials.contains(&material)
                && generation.capabilities.dtz_materials.contains(&material)
        }
    }

    pub fn probe_wdl(&self, st: &BoardState) -> Option<WdlProbeResult> {
        if !self.can_probe_wdl(st) {
            return None;
        }
        let tables = &self.generation.as_ref()?.tables;
        ProbePosition::from_board(st).probe_wdl(tables)
    }

    pub fn probe_wdl_50(&self, st: &BoardState) -> Option<WdlProbeResult> {
        if !self.can_probe_dtz(st) {
            return None;
        }
        let tables = &self.generation.as_ref()?.tables;
        let position = ProbePosition::from_board(st);
        wdl_from_dtz(position.probe_dtz(tables)?, position.halfmoves)
    }

    pub fn probe_dtz(&self, st: &BoardState) -> Option<i32> {
        if !self.can_probe_dtz(st) {
            return None;
        }
        let tables = &self.generation.as_ref()?.tables;
        ProbePosition::from_board(st).probe_dtz(tables)
    }

    /// Use the library's root selector and verify the selected move against
    /// Ember's legal move list before returning it to the UCI driver.
    pub fn probe_root_move(&self, st: &BoardState, legal_moves: &[Move]) -> Option<Move> {
        if legal_moves.is_empty() || !self.can_probe_dtz(st) {
            return None;
        }
        let tables = &self.generation.as_ref()?.tables;
        let root = ProbePosition::from_board(st).probe_root(tables)?;
        let DtzProbeValue::DtzResult(best) = root.root else {
            return None;
        };
        let promotion = match best.promotion {
            Piece::Queen => b'Q',
            Piece::Rook => b'R',
            Piece::Bishop => b'B',
            Piece::Knight => b'N',
            Piece::Pawn | Piece::King => 0,
        };
        legal_moves.iter().copied().find(|mv| {
            move_from(*mv) == (usize::from(best.from_square) ^ 56)
                && move_to(*mv) == (usize::from(best.to_square) ^ 56)
                && move_promotion(*mv) == promotion
        })
    }

    /// Exact, 50-move-aware score for interior search. Rounded boundary
    /// results are deliberately left to normal search instead of being used
    /// as false alpha-beta bounds.
    pub fn probe_search_score(&self, st: &BoardState, ply: usize) -> Option<i32> {
        Some(exact_search_score(self.probe_wdl_50(st)?, ply))
    }

    pub fn probe_root_score(&self, st: &BoardState) -> Option<i32> {
        if let Some(wdl) = self.probe_wdl_50(st) {
            return Some(exact_search_score(wdl, 0));
        }
        self.probe_dtz(st).map(i32::signum)
    }
}

#[cfg(test)]
mod tests {
    use super::{exact_search_score, wdl_from_dtz, EmberAdapter, ProbePosition, SyzygyTables};
    use crate::engine::Engine;
    use pyrrhic_rs::{Color, EngineAdapter, WdlProbeResult};

    fn engine_from_fen(fen: &str) -> Engine {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        engine
    }

    #[test]
    fn retired_generation_drops_after_its_last_snapshot() {
        // The weak reference observes the private Arc lifetime. A public
        // probe test cannot distinguish a released mapping from a pinned one.
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ember-syzygy-drop-{suffix}"));
        std::fs::create_dir(&dir).unwrap();
        std::fs::File::create(dir.join("KQvK.rtbw"))
            .unwrap()
            .set_len(80)
            .unwrap();
        let mut published = SyzygyTables::new();
        published.load(dir.to_str().unwrap()).unwrap();
        let weak = std::sync::Arc::downgrade(published.generation.as_ref().unwrap());
        let worker = published.clone();
        published.load(dir.to_str().unwrap()).unwrap();
        assert!(weak.upgrade().is_some());
        drop(worker);
        assert!(weak.upgrade().is_none());
        drop(published);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn converted_position_preserves_halfmoves_and_square_orientation() {
        let engine = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 73 1");
        let position = ProbePosition::from_board(&engine.st);
        assert_eq!(position.halfmoves, 73);
        assert_eq!(position.queens & (1u64 << 1), 1u64 << 1);
        assert_eq!(position.kings & (1u64 << 63), 1u64 << 63);
    }

    #[test]
    fn converted_position_preserves_en_passant() {
        let engine = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let position = ProbePosition::from_board(&engine.st);
        assert_eq!(position.ep, 43); // D6 in A1-based bitboards.
    }

    #[test]
    fn interior_scores_avoid_rounded_fifty_move_boundary() {
        assert!(exact_search_score(WdlProbeResult::Win, 7) > 0);
        assert!(exact_search_score(WdlProbeResult::Loss, 7) < 0);
        assert_eq!(exact_search_score(WdlProbeResult::CursedWin, 7), 0);
        assert_eq!(exact_search_score(WdlProbeResult::BlessedLoss, 7), 0);
        assert_eq!(wdl_from_dtz(30, 68), Some(WdlProbeResult::Win));
        assert_eq!(wdl_from_dtz(30, 69), None);
        assert_eq!(wdl_from_dtz(30, 70), None);
        assert_eq!(wdl_from_dtz(30, 71), None);
        assert_eq!(wdl_from_dtz(30, 72), Some(WdlProbeResult::CursedWin));
    }

    #[test]
    fn adapter_attack_masks_use_a1_based_squares() {
        assert_eq!(
            EmberAdapter::pawn_attacks(Color::White, 12),
            (1 << 19) | (1 << 21)
        );
        assert_eq!(
            EmberAdapter::pawn_attacks(Color::Black, 52),
            (1 << 43) | (1 << 45)
        );
        assert_eq!(EmberAdapter::knight_attacks(1) & (1 << 18), 1 << 18);
    }
}
