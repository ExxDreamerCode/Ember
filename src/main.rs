use chess_rs_lib::board::{piece_on, piece_type, EMPTY_SQ};
use chess_rs_lib::evaluate;
use chess_rs_lib::syzygy::SyzygyTables;
use chess_rs_lib::{opening_book, Engine, OpeningBook};
use std::io::{self, BufRead};

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
    let stdin = io::stdin();
    let mut engine = Engine::new();

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

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "uci" => {
                println!("id name Ember 1.1.1");
                println!("id author ExxDreamerCode");
                println!("option name Hash type spin default 128 min 1 max 4096");
                println!("option name Threads type spin default 1 min 1 max 256");
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
                parse_position(&mut engine, &parts);
            }
            "go" => {
                parse_go(&mut engine, &parts);
            }
            "quit" => break,
            "stop" => {}
            _ => {}
        }
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
                    engine.searcher.resize_tt(mb);
                }
            }
            "threads" => {
                if let Ok(n) = val.parse::<usize>() {
                    engine.num_threads = n.max(1);
                    eprintln!("info string Set threads to {}", engine.num_threads);
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
            while i < parts.len() {
                if let Some(m) = parse_uci_move(parts[i]) {
                    if !engine.make_move_uci(m.0, m.1, m.2, m.3, m.4) {
                        eprintln!(
                            "info string Ignoring illegal move in position command: {}",
                            parts[i]
                        );
                    }
                }
                i += 1;
            }
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
            while idx < parts.len() {
                if let Some(m) = parse_uci_move(parts[idx]) {
                    if !engine.make_move_uci(m.0, m.1, m.2, m.3, m.4) {
                        eprintln!(
                            "info string Ignoring illegal move in position command: {}",
                            parts[idx]
                        );
                    }
                }
                idx += 1;
            }
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

fn parse_go(engine: &mut Engine, parts: &[&str]) {
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

    let root_state = engine.st;
    let (best_move, _, nodes, elapsed) = engine.find_best_move(tl, depth);
    let _nps = if elapsed > 0.0 {
        (nodes as f64 / elapsed) as i64
    } else {
        0
    };

    if best_move.len() >= 4 && best_move != "0000" {
        if best_move.len() == 4 {
            let b = best_move.as_bytes();
            let sc = (b[0] - b'a') as usize;
            let sr = 8 - (b[1] - b'0') as usize;
            let er = 8 - (b[3] - b'0') as usize;
            if sr < 8 && sc < 8 && er < 8 {
                let piece_idx = piece_on(&root_state.bb, sr * 8 + sc);
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
