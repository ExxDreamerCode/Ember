use chess_rs_lib::backend::{
    compiled_search_backends, parse_search_backend_name, search_backend_available,
};
use chess_rs_lib::board::{piece_on, piece_type, EMPTY_SQ};
use chess_rs_lib::evaluate;
use chess_rs_lib::search::{active_search_backend, set_search_backend_override};
use chess_rs_lib::syzygy::SyzygyTables;
use chess_rs_lib::time_management::TimeManager;
use chess_rs_lib::zobrist::compute_hash;
use chess_rs_lib::{opening_book, Engine, OpeningBook};
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const MIN_HASH_MB: usize = 1;
const MAX_HASH_MB: usize = 4096;
const MIN_THREADS: usize = 1;
const MAX_THREADS: usize = 256;
const SHORT_SYNC_SEARCH_LIMIT_SECONDS: f64 = 0.050;
const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct SearchTask {
    id: u64,
    handle: thread::JoinHandle<()>,
    stopped: Arc<AtomicBool>,
    rx: mpsc::Receiver<(String, i32, u64, f64)>,
}

enum UciEvent {
    Command(String),
    SearchFinished(u64),
    InputClosed,
}

#[derive(Clone, Copy, Debug)]
struct SearchLimits {
    soft_seconds: f64,
    hard_seconds: f64,
    depth: i32,
    clock_managed: bool,
}

fn try_load_book(engine: &mut Engine, path: &std::path::Path) -> bool {
    let display = path.display();
    if path.exists() {
        if let Some(path_str) = path.to_str() {
            if let Err(e) = engine.load_book(path_str) {
                eprintln!("info string Failed to load book {}: {}", display, e);
                return false;
            }
            return true;
        }
    }
    false
}

fn maybe_load_nnue(path: &str) -> bool {
    match evaluate::init_nnue(path) {
        Ok(()) => {
            eprintln!("info string NNUE loaded: {}", path);
            true
        }
        Err(e) => {
            eprintln!("info string Failed to load NNUE ({}): {}", path, e);
            false
        }
    }
}

fn main() {
    let mut engine = Engine::new();
    let mut time_manager = TimeManager::default();
    let mut search_task: Option<SearchTask> = None;
    let mut next_search_id = 0u64;

    eprintln!("info string Loading embedded NNUE...");
    match evaluate::init_embedded_nnue() {
        Ok(()) => eprintln!("info string Embedded NNUE loaded"),
        Err(e) => eprintln!("info string Failed to load embedded NNUE: {}", e),
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            try_load_book(&mut engine, &exe_dir.join("book.bin"));
        }
    }
    let local_book = std::path::Path::new("book.bin");
    if engine.book.is_none() && local_book.exists() {
        try_load_book(&mut engine, local_book);
    }
    if engine.book.is_none() {
        eprintln!("info string Book file not found, using embedded book");
        match OpeningBook::load_from_bytes(opening_book::BOOK_DATA, "<embedded>") {
            Ok(book) => engine.book = Some(book),
            Err(e) => eprintln!("info string Failed to load embedded book: {}", e),
        }
    }

    let (event_tx, event_rx) = mpsc::channel::<UciEvent>();
    let stdin_tx = event_tx.clone();

    thread::Builder::new()
        .name("stdin".into())
        .spawn(move || {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                if stdin_tx.send(UciEvent::Command(trimmed)).is_err() {
                    break;
                }
            }
            let _ = stdin_tx.send(UciEvent::InputClosed);
        })
        .expect("failed to spawn stdin thread");

    while let Ok(event) = event_rx.recv() {
        let trimmed = match event {
            UciEvent::Command(command) => command,
            UciEvent::SearchFinished(id) => {
                if search_task.as_ref().is_some_and(|task| task.id == id) {
                    let task = search_task.take().unwrap();
                    task.handle.join().ok();
                    if let Ok(result) = task.rx.recv() {
                        print_bestmove(&engine, &result.0);
                    }
                }
                continue;
            }
            UciEvent::InputClosed => {
                if let Some(task) = search_task.take() {
                    task.stopped.store(true, Ordering::SeqCst);
                    task.handle.join().ok();
                }
                break;
            }
        };
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "uci" => {
                println!("id name Ember 1.1.2");
                println!("id author ExxDreamerCode");
                println!(
                    "option name Hash type spin default 256 min {} max {}",
                    MIN_HASH_MB, MAX_HASH_MB
                );
                println!(
                    "option name Threads type spin default 1 min {} max {}",
                    MIN_THREADS, MAX_THREADS
                );
                println!("option name Move Overhead type spin default 7 min 0 max 5000");
                println!("option name Book type string default <embedded>");
                println!("option name NNUE type string default <embedded>");
                print!("option name NNUEBackend type combo default auto var auto");
                for backend in compiled_search_backends() {
                    print!(" var {}", backend.name());
                }
                println!();
                println!("option name SyzygyPath type string default <empty>");
                println!("option name UCI_Chess960 type check default false");
                #[cfg(feature = "decision-trace")]
                println!("option name TraceFile type string default <empty>");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                reset_engine(&mut engine);
                time_manager.reset_for_new_game();
            }
            "setoption" if parts.len() >= 3 && parts[1].to_lowercase() == "name" => {
                let Some((name, val)) = parse_option_name_value(&parts) else {
                    continue;
                };

                match name.as_str() {
                    "book" => {
                        if val.is_empty() {
                            engine.book = None;
                            eprintln!("info string Book disabled");
                        } else if val.to_lowercase() == "<embedded>"
                            || val.to_lowercase() == "<default>"
                        {
                            match OpeningBook::load_from_bytes(
                                opening_book::BOOK_DATA,
                                "<embedded>",
                            ) {
                                Ok(book) => {
                                    engine.book = Some(book);
                                    eprintln!("info string Book switched to embedded");
                                }
                                Err(e) => {
                                    eprintln!("info string Failed to load embedded book: {}", e)
                                }
                            }
                        } else {
                            let path = std::path::Path::new(&val);
                            if !try_load_book(&mut engine, path) {
                                if let Ok(exe_path) = std::env::current_exe() {
                                    if let Some(exe_dir) = exe_path.parent() {
                                        try_load_book(&mut engine, &exe_dir.join(&val));
                                    }
                                }
                            }
                        }
                    }
                    "nnue" => {
                        if val.is_empty() {
                            match evaluate::reset_nnue() {
                                Ok(()) => eprintln!(
                                    "info string NNUE disabled (eval will fall back to classic)"
                                ),
                                Err(e) => eprintln!("info string Failed to disable NNUE: {}", e),
                            }
                        } else if val.to_lowercase() == "<embedded>"
                            || val.to_lowercase() == "<default>"
                        {
                            match evaluate::init_embedded_nnue() {
                                Ok(()) => eprintln!("info string NNUE switched to embedded"),
                                Err(e) => {
                                    eprintln!("info string Failed to load embedded NNUE: {}", e)
                                }
                            }
                        } else {
                            maybe_load_nnue(&val);
                        }
                    }
                    "nnuebackend" | "nnue backend" | "searchbackend" | "search backend" => {
                        set_nnue_backend(&val);
                    }
                    "syzygypath" => {
                        if val.is_empty() || val.to_lowercase() == "<empty>" {
                            engine.searcher.syzygy = SyzygyTables::new();
                            eprintln!("info string Syzygy tables disabled");
                        } else {
                            match engine.searcher.syzygy.load(&val) {
                                Ok(()) => eprintln!("info string Syzygy tables loaded: {}", val),
                                Err(e) => {
                                    eprintln!("info string Failed to load Syzygy tables: {}", e)
                                }
                            }
                        }
                    }
                    "uci_chess960" => {
                        let enable = val == "true";
                        set_chess960_mode(&mut engine, enable);
                        if enable {
                            eprintln!("info string Chess960 mode enabled");
                        } else {
                            eprintln!("info string Chess960 mode disabled");
                        }
                    }
                    "move overhead" => {
                        let parsed = val.parse::<f64>();
                        if !parsed.is_ok_and(|value| time_manager.set_move_overhead_ms(value)) {
                            eprintln!("info string Ignoring out-of-range Move Overhead: {}", val);
                        }
                    }
                    _ => {
                        parse_setoption(&mut engine, &name, &val);
                    }
                }
            }
            "eval" => {
                let score = evaluate::evaluate_nnue(&engine.st);
                let classic = chess_rs_lib::evaluate::evaluate(&engine.st);
                println!("info string NNUE eval: {} cp (from stm)", score);
                let stm_sign = if engine.st.w { 1 } else { -1 };
                println!(
                    "info string Classic eval: {} cp (from white), {} cp (from stm)",
                    classic,
                    classic * stm_sign
                );
            }
            "position" => {
                if let Some(task) = search_task.take() {
                    task.stopped.store(true, Ordering::SeqCst);
                    task.handle.join().ok();
                }
                parse_position(&mut engine, &parts);
            }
            "go" => {
                if search_task.is_some() {
                    continue;
                }

                let limits = parse_go_params(&parts, &engine, &mut time_manager);
                let search_start = Instant::now();
                if limits.clock_managed && limits.hard_seconds <= SHORT_SYNC_SEARCH_LIMIT_SECONDS {
                    let result = engine.find_best_move_with_time_limits_started_at(
                        limits.soft_seconds,
                        limits.hard_seconds,
                        limits.depth,
                        search_start,
                    );
                    print_bestmove(&engine, &result.0);
                    continue;
                }
                let search_id = next_search_id;
                next_search_id = next_search_id.wrapping_add(1);

                let st = engine.st;
                let shared_tt = Arc::clone(&engine.shared_tt);
                let stopped = Arc::new(AtomicBool::new(false));
                let num_threads = engine.num_threads;
                let book = engine.book.clone();

                let mut search_searcher = chess_rs_lib::search::Searcher::new(
                    Arc::clone(&shared_tt),
                    Arc::clone(&stopped),
                );
                engine.searcher.copy_root_context_to(&mut search_searcher);
                search_searcher.tt_mb = engine.searcher.tt_mb;
                #[cfg(feature = "decision-trace")]
                let trace_path = engine.trace.path().map(|p| p.display().to_string());

                let stopped_for_search = Arc::clone(&stopped);
                let search_finished_tx = event_tx.clone();
                let (tx, rx) = mpsc::channel();

                let handle = thread::Builder::new()
                    .name("search".into())
                    .stack_size(8 * 1024 * 1024)
                    .spawn(move || {
                        let mut search_engine = Engine::new_with(
                            st,
                            search_searcher,
                            shared_tt,
                            num_threads,
                            stopped_for_search,
                            book,
                        );
                        #[cfg(feature = "decision-trace")]
                        if let Some(tp) = trace_path {
                            search_engine.set_trace_file(&tp);
                        }
                        let result = if limits.clock_managed {
                            search_engine.find_best_move_with_time_limits_prepared_started_at(
                                limits.soft_seconds,
                                limits.hard_seconds,
                                limits.depth,
                                search_start,
                            )
                        } else {
                            search_engine.find_best_move_with_time_limits_prepared(
                                limits.soft_seconds,
                                limits.hard_seconds,
                                limits.depth,
                            )
                        };
                        tx.send(result).ok();
                        search_finished_tx
                            .send(UciEvent::SearchFinished(search_id))
                            .ok();
                    })
                    .expect("failed to spawn search thread");

                search_task = Some(SearchTask {
                    id: search_id,
                    handle,
                    stopped,
                    rx,
                });
            }
            "stop" => {
                if let Some(task) = search_task.take() {
                    task.stopped.store(true, Ordering::SeqCst);
                    task.handle.join().ok();
                    if let Ok((best_move, _, _, _)) = task.rx.recv() {
                        print_bestmove(&engine, &best_move);
                    }
                }
            }
            "quit" => {
                if let Some(task) = search_task.take() {
                    task.stopped.store(true, Ordering::SeqCst);
                    task.handle.join().ok();
                    if let Ok((best_move, _, _, _)) = task.rx.recv() {
                        print_bestmove(&engine, &best_move);
                    }
                }
                break;
            }
            _ => {}
        }
    }
}

fn print_bestmove(engine: &Engine, best_move: &str) {
    if best_move.len() >= 4 && best_move != "0000" {
        if best_move.len() == 4 {
            let b = best_move.as_bytes();
            let sc = (b[0] - b'a') as usize;
            let sr = 8 - (b[1] - b'0') as usize;
            let er = 8 - (b[3] - b'0') as usize;
            if sr < 8 && sc < 8 && er < 8 {
                let piece_idx = piece_on(&engine.st.bb, sr * 8 + sc);
                if piece_idx != EMPTY_SQ && piece_type(piece_idx) == 0 && (er == 0 || er == 7) {
                    println!("bestmove {}q", best_move);
                    return;
                }
            }
        }
        println!("bestmove {}", best_move);
    } else {
        println!("bestmove 0000");
    }
}

fn parse_option_name_value(parts: &[&str]) -> Option<(String, String)> {
    if parts.len() < 3 || !parts[1].eq_ignore_ascii_case("name") {
        return None;
    }
    let value_idx = parts
        .iter()
        .position(|part| part.eq_ignore_ascii_case("value"));
    let name_end = value_idx.unwrap_or(parts.len());
    if name_end <= 2 {
        return None;
    }
    let name = parts[2..name_end].join(" ").to_lowercase();
    let value = value_idx
        .map(|idx| parts.get(idx + 1..).unwrap_or(&[]).join(" "))
        .unwrap_or_default();
    Some((name, value))
}

fn set_nnue_backend(value: &str) {
    let normalized = chess_rs_lib::backend::normalize_backend_name(value);
    if normalized.is_empty() || normalized == "auto" || normalized == "default" {
        set_search_backend_override(None);
        eprintln!(
            "info string NNUE backend set to auto ({})",
            active_search_backend().name()
        );
        return;
    }

    let Some(backend) = parse_search_backend_name(value) else {
        eprintln!("info string Unknown NNUE backend: {}", value);
        return;
    };

    if !search_backend_available(backend) {
        eprintln!(
            "info string NNUE backend {} is not available on this CPU",
            backend.name()
        );
        return;
    }

    if set_search_backend_override(Some(backend)) {
        eprintln!("info string NNUE backend set to {}", backend.name());
    }
}

fn parse_setoption(engine: &mut Engine, name: &str, val: &str) {
    match name {
        "hash" => {
            if let Ok(mb) = val.parse::<usize>() {
                if (MIN_HASH_MB..=MAX_HASH_MB).contains(&mb) {
                    engine.searcher.resize_tt(mb);
                } else {
                    eprintln!("info string Ignoring out-of-range Hash value: {}", mb);
                }
            }
        }
        "threads" => {
            if let Ok(n) = val.parse::<usize>() {
                if (MIN_THREADS..=MAX_THREADS).contains(&n) {
                    engine.num_threads = n;
                    eprintln!("info string Set threads to {}", engine.num_threads);
                } else {
                    eprintln!("info string Ignoring out-of-range Threads value: {}", n);
                }
            }
        }
        #[cfg(feature = "decision-trace")]
        "tracefile" if !val.is_empty() => {
            engine.set_trace_file(val);
        }
        _ => {}
    }
}

fn refresh_root_hash(engine: &mut Engine) {
    engine.st.hash = compute_hash(&engine.st);
    if engine.searcher.rep_stack_len == 0 {
        engine.searcher.rep_stack.push(engine.st.hash);
        engine.searcher.rep_stack_len = 1;
    } else if let Some(slot) = engine
        .searcher
        .rep_stack
        .get_mut(engine.searcher.rep_stack_len - 1)
    {
        *slot = engine.st.hash;
    }
}

fn set_chess960_mode(engine: &mut Engine, enable: bool) {
    engine.st.chess960 = enable;
    refresh_root_hash(engine);
}

fn reset_engine(engine: &mut Engine) {
    let book = engine.book.take();
    let num_threads = engine.num_threads;
    let chess960 = engine.st.chess960;
    let syzygy = engine.searcher.syzygy.clone();
    #[cfg(feature = "decision-trace")]
    let trace = std::mem::take(&mut engine.trace);
    let tt_mb = engine.searcher.tt_mb;
    *engine = Engine::new();
    engine.book = book;
    engine.num_threads = num_threads;
    engine.searcher.syzygy = syzygy;
    set_chess960_mode(engine, chess960);
    #[cfg(feature = "decision-trace")]
    {
        engine.trace = trace;
    }
    engine.searcher.resize_tt(tt_mb);
}

fn parse_position(engine: &mut Engine, parts: &[&str]) {
    if parts.len() < 2 {
        return;
    }
    if parts[1] == "startpos" {
        engine.set_fen(STARTPOS_FEN);
        let mut i = 2;
        if i < parts.len() && parts[i] == "moves" {
            i += 1;
            apply_position_moves(engine, &parts[i..]);
        }
    } else if parts[1] == "fen" && parts.len() >= 8 {
        let fen = format!(
            "{} {} {} {} {} {}",
            parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
        );
        engine.set_fen(&fen);
        let mut idx = 8;
        if idx < parts.len() && parts[idx] == "moves" {
            idx += 1;
            apply_position_moves(engine, &parts[idx..]);
        }
    }
}

fn apply_position_moves(engine: &mut Engine, moves: &[&str]) {
    for mv_text in moves {
        let legal =
            parse_uci_move(mv_text).is_some_and(|m| engine.make_move_uci(m.0, m.1, m.2, m.3, m.4));
        if !legal {
            eprintln!(
                "info string Stopping position move list at illegal move: {}",
                mv_text
            );
            break;
        }
    }
}

fn parse_uci_move(mv: &str) -> Option<(usize, usize, usize, usize, u8)> {
    if mv.len() < 4 {
        return None;
    }
    let b = mv.as_bytes();
    let sc = (b[0] - b'a') as usize;
    let sr = 8 - (b[1] - b'0') as usize;
    let ec = (b[2] - b'a') as usize;
    let er = 8 - (b[3] - b'0') as usize;
    if sr >= 8 || sc >= 8 || er >= 8 || ec >= 8 {
        return None;
    }
    let promotion = if mv.len() >= 5 {
        match b[4] {
            b'q' | b'Q' => b'Q',
            b'r' | b'R' => b'R',
            b'b' | b'B' => b'B',
            b'n' | b'N' => b'N',
            _ => 0,
        }
    } else {
        0
    };
    Some((sr, sc, er, ec, promotion))
}

fn parse_go_params(
    parts: &[&str],
    engine: &Engine,
    time_manager: &mut TimeManager,
) -> SearchLimits {
    let mut wtime = 300000f64;
    let mut btime = 300000f64;
    let mut winc = 0f64;
    let mut binc = 0f64;
    let mut movetime = 0f64;
    let mut depth = 64i32;
    let mut movestogo = 0i32;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "wtime" if i + 1 < parts.len() => {
                wtime = parts[i + 1].parse().unwrap_or(300000.0);
                i += 1;
            }
            "btime" if i + 1 < parts.len() => {
                btime = parts[i + 1].parse().unwrap_or(300000.0);
                i += 1;
            }
            "winc" if i + 1 < parts.len() => {
                winc = parts[i + 1].parse().unwrap_or(0.0);
                i += 1;
            }
            "binc" if i + 1 < parts.len() => {
                binc = parts[i + 1].parse().unwrap_or(0.0);
                i += 1;
            }
            "movetime" if i + 1 < parts.len() => {
                movetime = parts[i + 1].parse().unwrap_or(0.0);
                i += 1;
            }
            "depth" if i + 1 < parts.len() => {
                depth = parts[i + 1].parse().unwrap_or(64);
                i += 1;
            }
            "movestogo" if i + 1 < parts.len() => {
                movestogo = parts[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "infinite" => {
                movetime = 1_000_000.0;
            }
            _ => {}
        }
        i += 1;
    }

    let time_ms = if engine.st.w { wtime } else { btime };
    let inc = if engine.st.w { winc } else { binc };
    let (soft_seconds, hard_seconds, clock_managed) = if movetime > 0.0 {
        let t = movetime / 1000.0;
        (t, t, true)
    } else if depth < 64 {
        (1_000_000_000.0, 1_000_000_000.0, false)
    } else {
        let budget = time_manager.clock_budget(time_ms, inc, movestogo, engine.st.mc);
        (budget.soft_seconds, budget.hard_seconds, true)
    };

    SearchLimits {
        soft_seconds,
        hard_seconds,
        depth,
        clock_managed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_rs_lib::board::board_to_fen;

    #[test]
    fn position_command_stops_after_illegal_move() {
        let mut engine = Engine::new();
        parse_position(
            &mut engine,
            &["position", "startpos", "moves", "e2e5", "g1f3"],
        );

        assert_eq!(
            board_to_fen(&engine.st),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "g1f3 must not be applied after the illegal e2e5 prefix"
        );
    }

    #[test]
    fn position_startpos_preserves_search_context_within_game() {
        let mut engine = Engine::new();
        let shared_tt = Arc::clone(&engine.shared_tt);
        engine.searcher.history[12][28] = 1_234;
        engine.searcher.corr_hist[321] = -87;

        parse_position(
            &mut engine,
            &["position", "startpos", "moves", "e2e4", "e7e5"],
        );

        assert!(
            Arc::ptr_eq(&shared_tt, &engine.shared_tt),
            "position must preserve the transposition table allocated by ucinewgame"
        );
        assert_eq!(engine.searcher.history[12][28], 1_234);
        assert_eq!(engine.searcher.corr_hist[321], -87);
        assert_eq!(
            board_to_fen(&engine.st),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        );
    }

    #[test]
    fn clock_search_reserves_time_to_finish_the_crossing_iteration() {
        let engine = Engine::new();
        let mut time_manager = TimeManager::default();
        let limits = parse_go_params(
            &[
                "go", "wtime", "8000", "btime", "8000", "winc", "80", "binc", "80",
            ],
            &engine,
            &mut time_manager,
        );

        assert!((0.15..0.25).contains(&limits.soft_seconds));
        assert!(
            limits.hard_seconds > limits.soft_seconds,
            "clock search needs iteration-overrun reserve: {limits:?}"
        );
    }

    #[test]
    fn fixed_movetime_remains_an_exact_hard_limit() {
        let engine = Engine::new();
        let mut time_manager = TimeManager::default();
        let limits = parse_go_params(&["go", "movetime", "500"], &engine, &mut time_manager);

        assert_eq!(limits.soft_seconds, 0.5);
        assert_eq!(limits.hard_seconds, 0.5);
    }

    #[test]
    fn setoption_rejects_out_of_range_resources() {
        let mut engine = Engine::new();
        let hash_mb = engine.searcher.tt_mb;
        let threads = engine.num_threads;

        let (hash_name, hash_value) =
            parse_option_name_value(&["setoption", "name", "Hash", "value", "0"]).unwrap();
        let (threads_name, threads_value) =
            parse_option_name_value(&["setoption", "name", "Threads", "value", "1000000"]).unwrap();
        parse_setoption(&mut engine, &hash_name, &hash_value);
        parse_setoption(&mut engine, &threads_name, &threads_value);

        assert_eq!(
            engine.searcher.tt_mb, hash_mb,
            "invalid Hash must not change the active hash setting"
        );
        assert_eq!(
            engine.num_threads, threads,
            "invalid Threads must not change the worker count"
        );
    }

    #[test]
    fn setoption_parses_multiword_backend_name() {
        let (name, value) =
            parse_option_name_value(&["setoption", "name", "NNUE", "Backend", "value", "scalar"])
                .unwrap();

        assert_eq!(name, "nnue backend");
        assert_eq!(value, "scalar");
    }

    #[test]
    fn reset_preserves_chess960_hash_alignment() {
        let mut engine = Engine::new();
        engine.st.chess960 = true;

        reset_engine(&mut engine);

        let recomputed = chess_rs_lib::zobrist::compute_hash(&engine.st);
        assert!(engine.st.chess960, "reset should preserve Chess960 mode");
        assert_eq!(
            engine.st.hash, recomputed,
            "reset must refresh the cached hash after preserving Chess960 mode"
        );
        assert_eq!(
            engine.searcher.rep_stack[engine.searcher.rep_stack_len - 1],
            recomputed,
            "root repetition hash must match the refreshed Chess960 hash"
        );
    }

    #[test]
    fn reset_preserves_loaded_syzygy_tables() {
        let mut engine = Engine::new();
        engine.searcher.syzygy.tables = Some(std::sync::Arc::new(shakmaty_syzygy::Tablebase::<
            shakmaty::Chess,
        >::new()));

        reset_engine(&mut engine);

        assert!(
            engine.searcher.syzygy.is_loaded(),
            "ucinewgame must preserve the configured SyzygyPath"
        );
    }
}
