use shakmaty::{
    uci::UciMove, Bitboard, Board, ByColor, ByRole, CastlingMode, Chess, Color as SColor,
    FromSetup, Setup, Square,
};
use shakmaty_syzygy::{AmbiguousWdl, Dtz, MaybeRounded, Tablebase, Wdl};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::board::{move_to_uci, BoardState, Move, MATE, MAX_PLY};

// Keep tablebase results below mate scores while making them decisive against
// any normal evaluation. The ply adjustment preserves the usual preference
// for reaching a winning tablebase sooner and delaying a losing one.
const TB_WIN_SCORE: i32 = MATE - MAX_PLY as i32;

fn exact_search_score(wdl: AmbiguousWdl, ply: usize) -> Option<i32> {
    let ply = ply.min(MAX_PLY) as i32;
    match wdl {
        AmbiguousWdl::Win => Some(TB_WIN_SCORE - ply),
        AmbiguousWdl::Loss => Some(-TB_WIN_SCORE + ply),
        AmbiguousWdl::Draw | AmbiguousWdl::CursedWin | AmbiguousWdl::BlessedLoss => Some(0),
        AmbiguousWdl::MaybeWin | AmbiguousWdl::MaybeLoss => None,
    }
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

#[derive(Clone, Debug, Default)]
struct SyzygyCapabilities {
    max_pieces: u32,
    wdl_materials: Arc<HashSet<MaterialKey>>,
    dtz_materials: Arc<HashSet<MaterialKey>>,
}

impl SyzygyCapabilities {
    fn from_directory(path: &Path, max_pieces: u32) -> Result<Self, String> {
        let mut wdl_materials = HashSet::new();
        let mut dtz_materials = HashSet::new();

        for entry in fs::read_dir(path)
            .map_err(|e| format!("Failed to list Syzygy directory {}: {}", path.display(), e))?
        {
            let entry = entry.map_err(|e| {
                format!(
                    "Failed to read Syzygy directory entry in {}: {}",
                    path.display(),
                    e
                )
            })?;
            let path = entry.path();
            let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let Some(material) = MaterialKey::from_stem(stem) else {
                continue;
            };

            match ext {
                "rtbw" => {
                    wdl_materials.insert(material);
                }
                "rtbz" => {
                    dtz_materials.insert(material);
                }
                _ => {}
            }
        }

        Ok(Self {
            max_pieces,
            wdl_materials: Arc::new(wdl_materials),
            dtz_materials: Arc::new(dtz_materials),
        })
    }

    fn empty() -> Self {
        Self::default()
    }
}

fn to_shakmaty_board(st: &BoardState) -> Board {
    let shift = |ember_sq: usize| -> u64 { 1u64 << to_shakmaty_square(ember_sq) as u32 };

    let mut white_bb = [0u64; 6];
    let mut black_bb = [0u64; 6];
    for sq in 0..64 {
        let b = shift(sq);
        for pi in 0..12 {
            if st.bb[pi] & (1u64 << sq) != 0 {
                if pi < 6 {
                    white_bb[pi] |= b;
                } else {
                    black_bb[pi - 6] |= b;
                }
                break;
            }
        }
    }
    let pawn = Bitboard(white_bb[0] | black_bb[0]);
    let knight = Bitboard(white_bb[1] | black_bb[1]);
    let bishop = Bitboard(white_bb[2] | black_bb[2]);
    let rook = Bitboard(white_bb[3] | black_bb[3]);
    let queen = Bitboard(white_bb[4] | black_bb[4]);
    let king = Bitboard(white_bb[5] | black_bb[5]);
    let white =
        Bitboard(white_bb[0] | white_bb[1] | white_bb[2] | white_bb[3] | white_bb[4] | white_bb[5]);
    let black =
        Bitboard(black_bb[0] | black_bb[1] | black_bb[2] | black_bb[3] | black_bb[4] | black_bb[5]);
    Board::try_from_bitboards(
        ByRole {
            pawn,
            knight,
            bishop,
            rook,
            queen,
            king,
        },
        ByColor { white, black },
    )
    .ok()
    .unwrap_or_else(Board::empty)
}

fn to_shakmaty_square(ember_sq: usize) -> Square {
    let col = ember_sq & 7;
    let ember_rank = ember_sq >> 3;
    let shak_rank = 7 - ember_rank;
    Square::new((shak_rank * 8 + col) as u32)
}

fn board_to_chess(st: &BoardState) -> Option<Chess> {
    let board = to_shakmaty_board(st);
    let color = if st.w { SColor::White } else { SColor::Black };
    let setup = Setup {
        board,
        turn: color,
        castling_rights: Bitboard(0),
        ep_square: st.ep.map(to_shakmaty_square),
        promoted: Bitboard(0),
        pockets: None,
        remaining_checks: None,
        halfmoves: st.halfmove_clock.into(),
        fullmoves: std::num::NonZeroU32::MIN,
    };
    Chess::from_setup(setup, CastlingMode::Standard).ok()
}

fn new_tablebase() -> Tablebase<Chess> {
    #[cfg(target_pointer_width = "64")]
    {
        // Syzygy tables are treated as immutable while the engine is running.
        // This is true for Nix-store tables and for normal tablebase installs.
        unsafe { Tablebase::with_mmap_filesystem() }
    }

    #[cfg(not(target_pointer_width = "64"))]
    {
        Tablebase::new()
    }
}

#[derive(Clone)]
pub struct SyzygyTables {
    pub tables: Option<Arc<Tablebase<Chess>>>,
    capabilities: SyzygyCapabilities,
}

impl Default for SyzygyTables {
    fn default() -> Self {
        Self::new()
    }
}

impl SyzygyTables {
    pub fn new() -> Self {
        SyzygyTables {
            tables: None,
            capabilities: SyzygyCapabilities::empty(),
        }
    }

    pub fn load(&mut self, path: &str) -> Result<(), String> {
        if path.is_empty() || path.to_lowercase() == "<empty>" {
            self.tables = None;
            self.capabilities = SyzygyCapabilities::empty();
            return Ok(());
        }
        let path = Path::new(path);
        if !path.exists() {
            return Err(format!("Syzygy directory not found: {}", path.display()));
        }
        let mut tb = new_tablebase();
        tb.add_directory(path)
            .map_err(|e| format!("Failed to load Syzygy tables: {}", e))?;
        let max_pieces = tb.max_pieces() as u32;
        if max_pieces == 0 {
            return Err(format!("No Syzygy tables found in {}", path.display()));
        }
        let capabilities = SyzygyCapabilities::from_directory(path, max_pieces)?;
        self.tables = Some(Arc::new(tb));
        self.capabilities = capabilities;
        Ok(())
    }

    pub fn max_pieces(&self) -> u32 {
        self.capabilities.max_pieces
    }

    pub fn is_loaded(&self) -> bool {
        self.tables.is_some()
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
        if !self.is_loaded() || !Self::pieces_ok(st) {
            return false;
        }
        let piece_count = Self::piece_count(st);
        if piece_count == 2 {
            return true;
        }
        piece_count <= self.max_pieces()
            && self
                .capabilities
                .wdl_materials
                .contains(&MaterialKey::from_board(st))
    }

    pub fn can_probe_dtz(&self, st: &BoardState) -> bool {
        if !self.is_loaded() || !Self::pieces_ok(st) {
            return false;
        }
        let piece_count = Self::piece_count(st);
        if piece_count == 2 {
            return true;
        }
        piece_count <= self.max_pieces() && {
            let material = MaterialKey::from_board(st);
            self.capabilities.wdl_materials.contains(&material)
                && self.capabilities.dtz_materials.contains(&material)
        }
    }

    pub fn probe_wdl(&self, st: &BoardState) -> Option<Wdl> {
        if !self.can_probe_wdl(st) {
            return None;
        }
        let tables = self.tables.as_ref()?;
        let chess = board_to_chess(st)?;
        tables.probe_wdl_after_zeroing(&chess).ok()
    }

    pub fn probe_wdl_50(&self, st: &BoardState) -> Option<AmbiguousWdl> {
        if !self.can_probe_dtz(st) {
            return None;
        }
        let tables = self.tables.as_ref()?;
        let chess = board_to_chess(st)?;
        tables.probe_wdl(&chess).ok()
    }

    pub fn probe_dtz(&self, st: &BoardState) -> Option<Dtz> {
        if !self.can_probe_dtz(st) {
            return None;
        }
        let tables = self.tables.as_ref()?;
        let chess = board_to_chess(st)?;
        match tables.probe_dtz(&chess).ok()? {
            MaybeRounded::Rounded(dtz) | MaybeRounded::Precise(dtz) => Some(dtz),
        }
    }

    /// Returns the library's canonical Syzygy root move. The dependency's
    /// selector handles captures, pawn moves, DTZ rounding, and the extra ply
    /// for non-zeroing moves; duplicating that recurrence here caused the
    /// previous zeroing and off-by-one bugs.
    pub fn probe_root_move(&self, st: &BoardState, legal_moves: &[Move]) -> Option<Move> {
        if legal_moves.is_empty() || !self.can_probe_dtz(st) {
            return None;
        }
        let tables = self.tables.as_ref()?;
        let chess = board_to_chess(st)?;
        let (best, _) = tables.best_move(&chess).ok()??;
        let best_uci = UciMove::from_standard(best).to_string();
        legal_moves
            .iter()
            .copied()
            .find(|mv| move_to_uci(st, *mv) == best_uci)
    }

    /// Exact, 50-move-aware score for interior search. Rounded boundary
    /// results are deliberately left to normal search instead of being used
    /// as false alpha-beta bounds.
    pub fn probe_search_score(&self, st: &BoardState, ply: usize) -> Option<i32> {
        exact_search_score(self.probe_wdl_50(st)?, ply)
    }

    pub fn probe_root_score(&self, st: &BoardState) -> Option<i32> {
        let wdl = self.probe_wdl_50(st)?;
        exact_search_score(wdl, 0).or_else(|| Some(wdl.signum()))
    }
}

#[cfg(test)]
mod tests {
    use super::{board_to_chess, exact_search_score, SyzygyTables};
    use crate::board::move_to_uci;
    use crate::engine::Engine;
    use crate::movegen::generate_moves;
    use shakmaty::{Position, Square};
    use shakmaty_syzygy::{AmbiguousWdl, Dtz, Wdl};
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn engine_from_fen(fen: &str) -> Engine {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        engine
    }

    fn temp_syzygy_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ember-syzygy-test-{unique}"));
        fs::create_dir(&dir).unwrap();
        dir
    }

    fn fake_table(dir: &Path, name: &str) {
        let file = File::create(dir.join(name)).unwrap();
        file.set_len(16).unwrap();
    }

    #[test]
    fn converted_position_preserves_the_halfmove_clock() {
        let engine = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 73 1");
        let chess = board_to_chess(&engine.st).expect("valid Syzygy position");

        assert_eq!(chess.halfmoves(), 73);
    }

    #[test]
    fn converted_position_preserves_en_passant() {
        let engine = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let chess = board_to_chess(&engine.st).expect("valid Syzygy position");

        assert_eq!(chess.maybe_ep_square(), Some(Square::D6));
    }

    #[test]
    fn dependency_dtz_recurrence_counts_the_root_ply() {
        assert_eq!((-Dtz(-2)).add_plies(1), Dtz(3));
        assert_eq!((-Dtz(2)).add_plies(1), Dtz(-3));
        assert_eq!(Dtz::before_zeroing(Wdl::Win), Dtz(1));
        assert_eq!(Dtz::before_zeroing(Wdl::CursedWin), Dtz(101));
        assert_eq!(Dtz::before_zeroing(Wdl::BlessedLoss), Dtz(-101));
    }

    #[test]
    fn interior_scores_are_exact_and_rule_50_aware() {
        assert!(exact_search_score(AmbiguousWdl::Win, 7).unwrap() > 0);
        assert!(exact_search_score(AmbiguousWdl::Loss, 7).unwrap() < 0);
        assert_eq!(exact_search_score(AmbiguousWdl::CursedWin, 7), Some(0));
        assert_eq!(exact_search_score(AmbiguousWdl::BlessedLoss, 7), Some(0));
        assert_eq!(exact_search_score(AmbiguousWdl::MaybeWin, 7), None);
        assert_eq!(exact_search_score(AmbiguousWdl::MaybeLoss, 7), None);
    }

    #[test]
    fn loaded_capabilities_filter_by_piece_count_and_material() {
        let dir = temp_syzygy_dir();
        fake_table(&dir, "KQvK.rtbw");
        fake_table(&dir, "KQvK.rtbz");

        let mut syzygy = SyzygyTables::new();
        syzygy.load(dir.to_str().unwrap()).unwrap();

        let kqvk = engine_from_fen("7k/8/8/8/8/8/8/Q3K3 w - - 0 1");
        let kvkq = engine_from_fen("q6k/8/8/8/8/8/8/4K3 w - - 0 1");
        let krv_k = engine_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
        let kqvkr = engine_from_fen("r6k/8/8/8/8/8/8/Q3K3 w - - 0 1");

        assert_eq!(syzygy.max_pieces(), 3);
        assert!(syzygy.can_probe_wdl(&kqvk.st));
        assert!(syzygy.can_probe_dtz(&kqvk.st));
        assert!(syzygy.can_probe_wdl(&kvkq.st));
        assert!(!syzygy.can_probe_wdl(&krv_k.st));
        assert!(!syzygy.can_probe_wdl(&kqvkr.st));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fast_check_accepts_en_passant_but_rejects_castling() {
        let dir = temp_syzygy_dir();
        fake_table(&dir, "KRvKR.rtbw");

        let mut syzygy = SyzygyTables::new();
        syzygy.load(dir.to_str().unwrap()).unwrap();

        let no_rights = engine_from_fen("4k2r/8/8/8/8/8/8/R3K3 w - - 0 1");
        let castling = engine_from_fen("4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1");
        let ep = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");

        assert!(syzygy.can_probe_wdl(&no_rights.st));
        assert!(!syzygy.can_probe_wdl(&castling.st));
        assert!(SyzygyTables::pieces_ok(&ep.st));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn known_six_piece_root_moves_when_tables_are_available() {
        let Ok(path) = std::env::var("EMBER_TEST_SYZYGY_PATH") else {
            eprintln!("skipping real Syzygy regressions: EMBER_TEST_SYZYGY_PATH is unset");
            return;
        };
        let mut syzygy = SyzygyTables::new();
        syzygy.load(&path).expect("load regression Syzygy tables");
        if syzygy.max_pieces() < 6 {
            eprintln!("skipping six-piece regressions: tablebase has fewer than six pieces");
            return;
        }

        let cases: &[(&str, &[&str])] = &[
            ("8/1r6/7R/3k2p1/5pK1/8/8/8 w - - 0 43", &["h6a6"]),
            ("8/2b4k/p7/4p3/4K3/1N6/8/8 w - - 4 50", &["e4f5", "b3d2"]),
            ("8/6k1/8/r5PR/2K4P/8/8/8 b - - 10 65", &["a5a6"]),
            ("5R2/3k2r1/1K6/1P6/8/8/5p2/8 b - - 1 51", &["g7g2"]),
            ("4q3/6KP/2N2p2/4k3/8/8/8/8 b - - 14 61", &["e5d6", "e5d5"]),
            ("1R6/8/7k/8/6p1/1P6/6r1/1K6 b - - 2 62", &["g2f2"]),
            ("6R1/8/8/1P6/7k/6p1/4r3/2K5 b - - 0 66", &["g3g2"]),
            ("5k2/8/8/p6P/n2K4/8/5P2/8 w - - 2 47", &["f2f4"]),
            ("1R6/4P2k/3K4/8/8/6p1/8/4r3 w - - 0 95", &["e7e8q", "e7e8r"]),
        ];

        for &(fen, expected) in cases {
            let engine = engine_from_fen(fen);
            let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
            let best = syzygy
                .probe_root_move(&engine.st, &legal)
                .expect("probe canonical root move");
            let actual = move_to_uci(&engine.st, best);
            assert!(
                expected.contains(&actual.as_str()),
                "expected one of {expected:?}, got {actual} for {fen}"
            );
        }
    }

    #[test]
    fn zeroing_capture_regression_only_needs_four_piece_tables() {
        let Ok(path) = std::env::var("EMBER_TEST_SYZYGY_PATH") else {
            eprintln!("skipping real Syzygy regressions: EMBER_TEST_SYZYGY_PATH is unset");
            return;
        };
        let mut syzygy = SyzygyTables::new();
        syzygy.load(&path).expect("load regression Syzygy tables");
        let engine = engine_from_fen("5k2/R7/8/8/5K2/p7/8/8 w - - 0 62");
        if !syzygy.can_probe_dtz(&engine.st) {
            eprintln!("skipping zeroing regression: KRvKP tables are unavailable");
            return;
        }
        let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let best = syzygy
            .probe_root_move(&engine.st, &legal)
            .expect("probe zeroing capture regression");

        assert_eq!(move_to_uci(&engine.st, best), "a7a3");
    }
}
