use chess_rs_lib::board::{move_ec, move_er, move_promotion, move_sc, move_sr};
use chess_rs_lib::evaluate::evaluate_nnue_acc;
use chess_rs_lib::movegen::{apply_move, generate_moves};
use chess_rs_lib::nnue::{NNUEAccumulator, NNUENet};
use chess_rs_lib::Engine;
use std::hint::black_box;
use std::time::Instant;

const NET: &[u8] = include_bytes!("../net.nnue");

const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QK2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 8",
    "2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14",
    "8/2p2pk1/1p4p1/p2Pp3/P1P1P1P1/1P3K2/8/8 w - - 0 40",
    "8/5pk1/6p1/3N4/3P4/5P2/6PK/8 w - - 0 45",
    "8/P4k2/8/8/8/8/8/6K1 w - - 0 1",
    "8/8/8/R2pP1k/8/8/6Q1/4K3 w - d6 0 1",
];

fn parse_arg(name: &str, default: usize) -> usize {
    let prefix = format!("{name}=");
    std::env::args()
        .skip(1)
        .find_map(|arg| {
            arg.strip_prefix(&prefix)
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(default)
}

fn states() -> Vec<chess_rs_lib::board::BoardState> {
    FENS.iter()
        .map(|fen| {
            let mut engine = Engine::new();
            engine.book = None;
            engine.set_fen(fen);
            engine.st
        })
        .collect()
}

fn bench_refresh(net: &NNUENet, states: &[chess_rs_lib::board::BoardState], loops: usize) -> i64 {
    let mut checksum = 0i64;
    let mut acc = NNUEAccumulator::new(net.hidden_size);
    for _ in 0..loops {
        for st in states {
            acc.refresh(black_box(net), black_box(st));
            checksum += evaluate_nnue_acc(net, black_box(&acc), st) as i64;
        }
    }
    checksum
}

fn bench_incremental(
    net: &NNUENet,
    states: &[chess_rs_lib::board::BoardState],
    loops: usize,
) -> (i64, usize) {
    let mut checksum = 0i64;
    let mut updates = 0usize;

    for _ in 0..loops {
        for st in states {
            let moves = generate_moves(st, st.w, &st.cr, st.ep);
            let mut base_acc = NNUEAccumulator::new(net.hidden_size);
            base_acc.refresh(net, st);

            for &mv in &moves {
                let mut next_acc = base_acc.clone();
                if !next_acc.update_move(
                    net,
                    st,
                    move_sr(mv),
                    move_sc(mv),
                    move_er(mv),
                    move_ec(mv),
                    move_promotion(mv),
                ) {
                    continue;
                }

                let mut next = *st;
                apply_move(
                    &mut next,
                    move_sr(mv),
                    move_sc(mv),
                    move_er(mv),
                    move_ec(mv),
                    move_promotion(mv),
                );
                checksum += evaluate_nnue_acc(net, black_box(&next_acc), black_box(&next)) as i64;
                updates += 1;
            }
        }
    }

    (checksum, updates)
}

fn main() {
    let refresh_loops = parse_arg("--refresh-loops", 2000);
    let update_loops = parse_arg("--update-loops", 200);
    let net =
        NNUENet::load_from_bytes(NET, "<microbench>").expect("embedded dense NNUE should load");
    let states = states();

    let refresh_calls = refresh_loops * states.len();
    let start = Instant::now();
    let refresh_checksum = bench_refresh(&net, &states, refresh_loops);
    let refresh_elapsed = start.elapsed();

    let start = Instant::now();
    let (update_checksum, update_calls) = bench_incremental(&net, &states, update_loops);
    let update_elapsed = start.elapsed();

    println!(
        "refresh calls={} elapsed_ns={} ns_per_call={:.2} checksum={}",
        refresh_calls,
        refresh_elapsed.as_nanos(),
        refresh_elapsed.as_nanos() as f64 / refresh_calls as f64,
        refresh_checksum
    );
    println!(
        "incremental calls={} elapsed_ns={} ns_per_call={:.2} checksum={}",
        update_calls,
        update_elapsed.as_nanos(),
        update_elapsed.as_nanos() as f64 / update_calls as f64,
        update_checksum
    );
}
