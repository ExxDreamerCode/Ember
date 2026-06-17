use shakmaty::{
    Board, CastlingMode, Chess, Color as SColor, FromSetup, Setup, Bitboard, ByColor, ByRole,
};
use shakmaty_syzygy::{Dtz, MaybeRounded, Tablebase, Wdl};
use std::path::Path;

use crate::board::BoardState;

const SYZYGY_MAX_PIECES: u32 = 6;

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
                    white_bb[pi as usize] |= b;
                } else {
                    black_bb[(pi - 6) as usize] |= b;
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
    let white = Bitboard(
        white_bb[0] | white_bb[1] | white_bb[2] | white_bb[3] | white_bb[4] | white_bb[5],
    );
    let black = Bitboard(
        black_bb[0] | black_bb[1] | black_bb[2] | black_bb[3] | black_bb[4] | black_bb[5],
    );
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
    if !SyzygyTables::pieces_ok(st) {
        return None;
    }
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
        halfmoves: 0,
        fullmoves: std::num::NonZeroU32::MIN,
    };
    Chess::from_setup(setup, CastlingMode::Standard).ok()
}

pub struct SyzygyTables {
    pub tables: Option<Tablebase<Chess>>,
}

impl SyzygyTables {
    pub fn new() -> Self {
        SyzygyTables { tables: None }
    }

    pub fn load(&mut self, path: &str) -> Result<(), String> {
        if path.is_empty() || path.to_lowercase() == "<empty>" {
            self.tables = None;
            return Ok(());
        }
        if !Path::new(path).exists() {
            return Err(format!("Syzygy directory not found: {}", path));
        }
        let mut tb = Tablebase::new();
        tb.add_directory(path)
            .map_err(|e| format!("Failed to load Syzygy tables: {}", e))?;
        self.tables = Some(tb);
        Ok(())
    }

    pub fn pieces_ok(st: &BoardState) -> bool {
        let cnt: u32 = (0..12).map(|i| st.bb[i].count_ones()).sum();
        cnt > 0 && cnt <= SYZYGY_MAX_PIECES
    }

    pub fn probe_wdl(&self, st: &BoardState) -> Option<Wdl> {
        let tables = self.tables.as_ref()?;
        let chess = board_to_chess(st)?;
        tables.probe_wdl_after_zeroing(&chess).ok()
    }

    pub fn probe_dtz(&self, st: &BoardState) -> Option<Dtz> {
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