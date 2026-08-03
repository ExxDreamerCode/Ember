pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;

pub const PAWN: u8 = 0;
pub const KNIGHT: u8 = 1;
pub const BISHOP: u8 = 2;
pub const ROOK: u8 = 3;
pub const QUEEN: u8 = 4;
pub const KING: u8 = 5;

pub const EMPTY_SQ: u8 = 255;
pub const MATE: i32 = 100_000;
pub const INF: i32 = 1_000_000;
pub const MAX_PLY: usize = 128;
pub const QS_DEPTH: i32 = 0;
pub const MAX_HALF_MOVE_CLOCK: u8 = 150;

pub const WP: usize = 0;
pub const WN: usize = 1;
pub const WB: usize = 2;
pub const WR: usize = 3;
pub const WQ: usize = 4;
pub const WK: usize = 5;
pub const BP: usize = 6;
pub const BN: usize = 7;
pub const BB: usize = 8;
pub const BR: usize = 9;
pub const BQ: usize = 10;
pub const BK: usize = 11;

pub type Move = u16;
pub const NO_MOVE: Move = 0;
