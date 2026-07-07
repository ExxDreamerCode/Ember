use chess_rs_lib::board::{piece_on, piece_type, EMPTY_SQ};
use chess_rs_lib::evaluate;
use chess_rs_lib::syzygy::SyzygyTables;
use chess_rs_lib::{opening_book, Engine, OpeningBook};
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const MIN_HASH_MB: usize = 1;
const MAX_HASH_MB: usize = 4096;
const MIN_THREADS: usize = 1;
const MAX_THREADS: usize = 256;

struct SearchTask {
    handle: thread::JoinHandle<()>,
    stopped: Arc<AtomicBool>,
    rx: mpsc::Receiver<(String, i32, u64, f64)>,
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
    let mut search_task: Option<SearchTask> = None;

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

    let (cmd_tx, cmd_rx) = mpsc::channel::<String>();

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
                if cmd_tx.send(trimmed).is_err() {
                    break;
                }
            }
        })
        .expect("failed to spawn stdin thread");

    loop {
        let cmd = if let Some(ref task) = search_task {
            match cmd_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(cmd) => Some(cmd),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if let Ok(result) = task.rx.try_recv() {
                        let task = search_task.take().unwrap();
                        task.handle.join().ok();
                        print_bestmove(&engine, &result.0);
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match cmd_rx.recv() {
                Ok(cmd) => Some(cmd),
                Err(_) => break,
            }
        };

        let Some(trimmed) = cmd else {
            break;
        };
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "uci" => {
                println!("id name Ember 1.1.1");
                println!("id author ExxDreamerCode");
                println!(
                    "option name Hash type spin default 256 min {} max {}",
                    MIN_HASH_MB, MAX_HASH_MB
                );
                println!(
                    "option name Threads type spin default 1 min {} max {}",
                    MIN_THREADS, MAX_THREADS
                );
                println!("option name Book type string default <embedded>");
                println!("option name NNUE type string default <embedded>");
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
            }
            "setoption" if parts.len() >= 3 && parts[1].to_lowercase() == "name" => {
                let name = parts[2].to_lowercase();
                let val_start = parts
                    .iter()
                    .position(|part| part.eq_ignore_ascii_case("value"))
                    .map(|idx| idx + 1)
                    .unwrap_or(3);
                let val = parts[val_start..].join(" ");

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
                        engine.st.chess960 = enable;
                        if enable {
                            eprintln!("info string Chess960 mode enabled");
                        } else {
                            eprintln!("info string Chess960 mode disabled");
                        }
                    }
                    _ => {
                        parse_setoption(&mut engine, &parts);
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

                let (tl, depth) = parse_go_params(&parts, &engine);

                let st = engine.st;
                let shared_tt = Arc::clone(&engine.shared_tt);
                let stopped = Arc::new(AtomicBool::new(false));
                let num_threads = engine.num_threads;

                let mut search_searcher = chess_rs_lib::search::Searcher::new(
                    Arc::clone(&shared_tt),
                    Arc::clone(&stopped),
                );
                engine.searcher.copy_root_context_to(&mut search_searcher);
                search_searcher.tt_mb = engine.searcher.tt_mb;
                #[cfg(feature = "decision-trace")]
                let trace_path = engine.trace.path().map(|p| p.display().to_string());

                let stopped_for_search = Arc::clone(&stopped);
                let (tx, rx) = mpsc::channel();

                let handle = thread::Builder::new()
                    .name("search".into())
                    .spawn(move || {
                        let mut search_engine = Engine::new_with(
                            st,
                            search_searcher,
                            shared_tt,
                            num_threads,
                            stopped_for_search,
                        );
                        #[cfg(feature = "decision-trace")]
                        if let Some(tp) = trace_path {
                            search_engine.set_trace_file(&tp);
                        }
                        let result = search_engine.find_best_move(tl, depth);
                        tx.send(result).ok();
                    })
                    .expect("failed to spawn search thread");

                search_task = Some(SearchTask {
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

fn parse_setoption(engine: &mut Engine, parts: &[&str]) {
    if parts.len() >= 5 && parts[1].to_lowercase() == "name" && parts[3].to_lowercase() == "value" {
        let value_idx = parts
            .iter()
            .position(|part| part.eq_ignore_ascii_case("value"))
            .unwrap_or(3);
        let name = parts[2..value_idx].join(" ").to_lowercase();
        let val = parts.get(value_idx + 1..).unwrap_or(&[]).join(" ");
        match name.as_str() {
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
                engine.set_trace_file(&val);
            }
            _ => {}
        }
    }
}

fn reset_engine(engine: &mut Engine) {
    let book = engine.book.take();
    let num_threads = engine.num_threads;
    let chess960 = engine.st.chess960;
    #[cfg(feature = "decision-trace")]
    let trace = std::mem::take(&mut engine.trace);
    let tt_mb = engine.searcher.tt_mb;
    *engine = Engine::new();
    engine.book = book;
    engine.num_threads = num_threads;
    engine.st.chess960 = chess960;
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
        reset_engine(engine);
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

#[allow(clippy::manual_clamp)]
fn clamp_time_limit(tl: f64) -> f64 {
    tl.max(0.05).min(60.0)
}

fn parse_go_params(parts: &[&str], engine: &Engine) -> (f64, i32) {
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

    let tl = if movetime > 0.0 {
        movetime / 1000.0
    } else if depth < 64 {
        1_000_000_000.0
    } else {
        let t = if engine.st.w { wtime } else { btime };
        let inc = if engine.st.w { winc } else { binc };
        let moves_left = if movestogo > 0 {
            movestogo as f64
        } else {
            30.0
        };
        (t / (moves_left + 2.0) + inc * 0.8) / 1000.0
    };
    let tl = if depth < 64 { tl } else { clamp_time_limit(tl) };

    (tl, depth)
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
    fn setoption_rejects_out_of_range_resources() {
        let mut engine = Engine::new();
        let hash_mb = engine.searcher.tt_mb;
        let threads = engine.num_threads;

        parse_setoption(&mut engine, &["setoption", "name", "Hash", "value", "0"]);
        parse_setoption(
            &mut engine,
            &["setoption", "name", "Threads", "value", "1000000"],
        );

        assert_eq!(
            engine.searcher.tt_mb, hash_mb,
            "invalid Hash must not change the active hash setting"
        );
        assert_eq!(
            engine.num_threads, threads,
            "invalid Threads must not change the worker count"
        );
    }
}
