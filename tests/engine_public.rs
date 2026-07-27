use std::time::{Duration, Instant};

use chess_rs_lib::{opening_book, Engine, OpeningBook};

fn play_uci(engine: &mut Engine, uci: &str) {
    let bytes = uci.as_bytes();
    assert!(bytes.len() >= 4, "invalid UCI move: {uci}");
    let promotion = bytes.get(4).map_or(0, |piece| piece.to_ascii_uppercase());
    assert!(
        engine.make_move_uci(
            8 - usize::from(bytes[1] - b'0'),
            usize::from(bytes[0] - b'a'),
            8 - usize::from(bytes[3] - b'0'),
            usize::from(bytes[2] - b'a'),
            promotion,
        ),
        "expected legal move {uci}"
    );
}

#[test]
fn embedded_book_ponder_fallback_uses_book_reply_without_tt() {
    let mut engine = Engine::new();
    engine.book =
        Some(OpeningBook::load_from_bytes(opening_book::BOOK_DATA, "<embedded>").unwrap());

    let ponder = engine
        .ponder_move_after("e2e4")
        .expect("embedded book should provide a black reply after 1.e4");

    assert!(
        ["c7c5", "e7e5", "e7e6", "c7c6", "d7d6"].contains(&ponder.as_str()),
        "unexpected embedded-book ponder reply after 1.e4: {ponder}"
    );
}

#[test]
fn ponder_book_reply_can_relax_normal_book_confidence() {
    let mut engine = Engine::new();
    engine.book =
        Some(OpeningBook::load_from_bytes(opening_book::BOOK_DATA, "<embedded>").unwrap());
    engine.book_min_move_weight = u16::MAX;

    let ponder = engine
        .ponder_move_after("e2e4")
        .expect("relaxed book fallback should still provide a ponder reply");

    assert_eq!(ponder, "c7c5");
}

#[test]
fn book_confidence_cutoff_rejects_weight_one_tail_move() {
    let mut engine = Engine::new();
    engine.book =
        Some(OpeningBook::load_from_bytes(opening_book::BOOK_DATA, "<embedded>").unwrap());
    for mv in [
        "e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "c5d4", "c3d4", "b8c6", "g1f3",
        "g8e7", "f1d3", "e7f5", "d3f5", "e6f5", "b1c3", "f8e7",
    ] {
        play_uci(&mut engine, mv);
    }

    let (_best_move, _score, nodes, _elapsed) =
        engine.find_best_move_with_time_limits(0.01, 0.01, 1);

    assert!(
        nodes > 0,
        "the weight-one 10.h4 book tail should be rejected so search starts"
    );
}

#[test]
fn random_book_move_returns_before_search_when_a_good_move_exists() {
    let mut engine = Engine::new();
    engine.book =
        Some(OpeningBook::load_from_bytes(opening_book::BOOK_DATA, "<embedded>").unwrap());
    engine.random_book_move = true;
    for mv in ["g1f3", "c7c5", "e2e4", "a7a6"] {
        play_uci(&mut engine, mv);
    }

    let (best_move, _score, nodes, _elapsed) = engine.find_best_move_with_time_limits(1.0, 1.0, 64);

    assert_eq!(best_move, "d2d4", "https://lichess.org/F1W14oiR");
    assert_eq!(
        nodes, 0,
        "random book selection must not start search when a confident move exists"
    );
}

#[test]
fn caller_supplied_start_time_is_used_for_clock_search() {
    let mut engine = Engine::new();
    engine.book = None;
    engine.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let expired_start = Instant::now() - Duration::from_millis(50);

    let (_, _, nodes, elapsed) =
        engine.find_best_move_with_time_limits_prepared_started_at(0.005, 0.010, 64, expired_start);

    assert_eq!(nodes, 0, "search ignored the already-expired clock");
    assert!(
        elapsed >= 0.050,
        "reported elapsed time must include the caller's start point: {elapsed}"
    );
}
