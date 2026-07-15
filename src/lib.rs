#![feature(portable_simd)]

pub mod backend;
pub mod board;
pub mod book;
pub mod engine;
pub mod evaluate;
pub mod magic;
pub mod movegen;
pub mod polyglot_randoms;
pub mod search;
#[cfg(feature = "decision-trace")]
pub mod trace;
pub mod tt;
pub mod zobrist;

pub mod opening_book;

pub mod bitboard;
pub mod nnue;
mod simd;
pub mod syzygy;
pub mod types;

pub use board::piece_type as ptype;
pub use book::OpeningBook;
pub use engine::Engine;
