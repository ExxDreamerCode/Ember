use shakmaty::{
    Bitboard, Board, ByColor, ByRole, CastlingMode, Chess, Color as SColor, FromSetup, Setup,
};
use shakmaty_syzygy::{Dtz, MaybeRounded, Tablebase, Wdl};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::board::BoardState;

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
    let shift = |ember_sq: usize| -> u64 {
        let col = ember_sq & 7;
        let ember_rank = ember_sq >> 3;
        let shak_rank = 7 - ember_rank;
        let shak_sq = shak_rank * 8 + col;
        1u64 << shak_sq
    };

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

fn board_to_chess(st: &BoardState) -> Option<Chess> {
    let board = to_shakmaty_board(st);
    let color = if st.w { SColor::White } else { SColor::Black };
    let setup = Setup {
        board,
        turn: color,
        castling_rights: Bitboard(0),
        ep_square: None,
        promoted: Bitboard(0),
        pockets: None,
        remaining_checks: None,
        halfmoves: st.halfmove_clock,
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
            && st.ep.is_none()
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
        self.is_loaded() && Self::pieces_ok(st) && Self::piece_count(st) <= self.max_pieces() && {
            let material = MaterialKey::from_board(st);
            self.capabilities.wdl_materials.contains(&material)
                && self.capabilities.dtz_materials.contains(&material)
        }
    }

    pub fn can_probe_dtz_after_one_move(&self, st: &BoardState) -> bool {
        self.is_loaded()
            && Self::piece_count(st) >= 2
            && Self::piece_count(st) <= self.max_pieces().saturating_add(1)
    }

    pub fn probe_wdl(&self, st: &BoardState) -> Option<Wdl> {
        if !self.can_probe_wdl(st) {
            return None;
        }
        let tables = self.tables.as_ref()?;
        let chess = board_to_chess(st)?;
        tables.probe_wdl_after_zeroing(&chess).ok()
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

    pub fn wdl_to_score(wdl: Wdl) -> Option<i32> {
        match wdl {
            Wdl::Win => Some(100_000),
            Wdl::Loss => Some(-100_000),
            Wdl::Draw => Some(0),
            Wdl::CursedWin => Some(99_999),
            Wdl::BlessedLoss => Some(-99_999),
        }
    }

    pub fn probe_wdl_value(&self, st: &BoardState) -> Option<i32> {
        let wdl = self.probe_wdl(st)?;
        match wdl {
            Wdl::Win => Some(100_000),
            Wdl::Loss => Some(-100_000),
            Wdl::Draw => Some(0),
            Wdl::CursedWin => Some(100_000),
            Wdl::BlessedLoss => Some(-100_000),
        }
    }

    pub fn probe_decisive(&self, st: &BoardState) -> Option<(i32, bool)> {
        let wdl = self.probe_wdl(st)?;
        match wdl {
            Wdl::Win => Some((100_000, true)),
            Wdl::Loss => Some((-100_000, true)),
            Wdl::Draw => Some((0, true)),
            Wdl::CursedWin => Some((100_000, false)),
            Wdl::BlessedLoss => Some((-100_000, false)),
        }
    }

    pub fn probe_cutoff(&self, st: &BoardState, beta: i32, alpha: i32) -> Option<i32> {
        let wdl = self.probe_wdl(st)?;
        match wdl {
            Wdl::Win | Wdl::CursedWin => Some(beta),
            Wdl::Loss | Wdl::BlessedLoss => Some(alpha),
            Wdl::Draw => Some(0),
        }
    }

    pub fn dtz_bonus(&self, st: &BoardState) -> Option<i32> {
        let dtz = self.probe_dtz(st)?;
        let val: i32 = dtz.0;
        if val > 0 {
            Some(-val)
        } else if val < 0 {
            Some(val)
        } else {
            Some(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SyzygyTables;
    use crate::engine::Engine;
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
    fn fast_check_rejects_state_not_representable_in_syzygy() {
        let dir = temp_syzygy_dir();
        fake_table(&dir, "KRvKR.rtbw");

        let mut syzygy = SyzygyTables::new();
        syzygy.load(dir.to_str().unwrap()).unwrap();

        let no_rights = engine_from_fen("4k2r/8/8/8/8/8/8/R3K3 w - - 0 1");
        let castling = engine_from_fen("4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1");
        let ep = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");

        assert!(syzygy.can_probe_wdl(&no_rights.st));
        assert!(!syzygy.can_probe_wdl(&castling.st));
        assert!(!SyzygyTables::pieces_ok(&ep.st));

        fs::remove_dir_all(dir).unwrap();
    }
}
