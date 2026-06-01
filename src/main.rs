use std::io::{self, BufRead};
use chess_rs_lib::{Engine, ptype};

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

fn main() {
    let stdin = io::stdin();
    let mut engine = Engine::new();

    let mut loaded = false;
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            loaded = try_load_book(&mut engine, &exe_dir.join("book.bin"));
        }
    }
    if !loaded {
        try_load_book(&mut engine, &std::path::Path::new("book.bin"));
    }
    
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "uci" => {
                println!("id name RustChess 2.0");
                println!("id author Rust");
                println!("option name Hash type spin default 128 min 1 max 4096");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("option name Book type string default <empty>");
                println!("uciok");
            }
            "isready" => { println!("readyok"); }
            "ucinewgame" => {
                let book = engine.book.take();
                engine = Engine::new();
                engine.book = book;
            }
            "setoption" => {
                if parts.len() >= 3 && parts[1].to_lowercase() == "name" {
                    if parts[2].to_lowercase() == "book" {
                        let val_start = if parts.len() >= 5 && parts[3].to_lowercase() == "value" { 4 } else { 3 };
                        let val = parts[val_start..].join(" ");
                        if val.is_empty() {
                            engine.book = None;
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
                    } else {
                        parse_setoption(&mut engine, &parts);
                    }
                }
            }
            "position" => { parse_position(&mut engine, &parts); }
            "go" => { parse_go(&mut engine, &parts); }
            "quit" => break,
            "stop" => {}
            _ => {}
        }
    }
}

fn parse_setoption(engine: &mut Engine, parts: &[&str]) {
    if parts.len() >= 5 && parts[1].to_lowercase() == "name" && parts[3].to_lowercase() == "value" {
        let name = parts[2].to_lowercase();
        let val = parts[4];
        match name.as_str() {
            "hash" => {
                if let Ok(mb) = val.parse::<usize>() {
                    engine.searcher.resize_tt(mb);
                }
            }
            _ => {}
        }
    }
}

fn parse_position(engine: &mut Engine, parts: &[&str]) {
    if parts.len() < 2 { return; }
    if parts[1] == "startpos" {
        let book = engine.book.take();
        *engine = Engine::new();
        engine.book = book;
        engine.searcher.resize_tt(engine.searcher.tt_mb);
        let mut i = 2;
        if i < parts.len() && parts[i] == "moves" {
            i += 1;
            while i < parts.len() {
                if let Some(m) = parse_uci_move(parts[i]) {
                    engine.make_move_uci(m.0, m.1, m.2, m.3, m.4);
                }
                i += 1;
            }
        }
    } else if parts[1] == "fen" && parts.len() >= 8 {
        let fen = format!("{} {} {} {} {} {}", parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]);
        engine.set_fen(&fen);
        let mut idx = 8;
        if idx < parts.len() && parts[idx] == "moves" {
            idx += 1;
            while idx < parts.len() {
                if let Some(m) = parse_uci_move(parts[idx]) {
                    engine.make_move_uci(m.0, m.1, m.2, m.3, m.4);
                }
                idx += 1;
            }
        }
    }
}

fn parse_uci_move(mv: &str) -> Option<(usize,usize,usize,usize,u8)> {
    if mv.len() < 4 { return None; }
    let b = mv.as_bytes();
    let sc = (b[0]-b'a') as usize; let sr = 8-(b[1]-b'0') as usize;
    let ec = (b[2]-b'a') as usize; let er = 8-(b[3]-b'0') as usize;
    if sr >= 8 || sc >= 8 || er >= 8 || ec >= 8 { return None; }
    let promotion = if mv.len() >= 5 {
        match b[4] {
            b'q'|b'Q' => b'Q', b'r'|b'R' => b'R', b'b'|b'B' => b'B', b'n'|b'N' => b'N', _ => 0
        }
    } else { 0 };
    Some((sr, sc, er, ec, promotion))
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
            "wtime"     => { if i+1 < parts.len() { wtime = parts[i+1].parse().unwrap_or(300000.0); i += 1; } }
            "btime"     => { if i+1 < parts.len() { btime = parts[i+1].parse().unwrap_or(300000.0); i += 1; } }
            "winc"      => { if i+1 < parts.len() { winc = parts[i+1].parse().unwrap_or(0.0); i += 1; } }
            "binc"      => { if i+1 < parts.len() { binc = parts[i+1].parse().unwrap_or(0.0); i += 1; } }
            "movetime"  => { if i+1 < parts.len() { movetime = parts[i+1].parse().unwrap_or(0.0); i += 1; } }
            "depth"     => { if i+1 < parts.len() { depth = parts[i+1].parse().unwrap_or(64); i += 1; } }
            "movestogo" => { if i+1 < parts.len() { movestogo = parts[i+1].parse().unwrap_or(0); i += 1; } }
            "infinite"  => { movetime = 1_000_000.0; }
            _ => {}
        }
        i += 1;
    }

    let tl = if movetime > 0.0 {
        movetime / 1000.0
    } else {
        let t = if engine.st.w { wtime } else { btime };
        let inc = if engine.st.w { winc } else { binc };
        let moves_left = if movestogo > 0 { movestogo as f64 } else { 30.0 };
        (t / (moves_left + 2.0) + inc * 0.8) / 1000.0
    };
    let tl = tl.max(0.05).min(60.0);

    let (best_move, _, nodes, elapsed) = engine.find_best_move(tl, depth);
    let nps = if elapsed > 0.0 { (nodes as f64 / elapsed) as i64 } else { 0 };

    if best_move.len() >= 4 {
        let b = best_move.as_bytes();
        let sc = (b[0]-b'a') as usize; let sr = 8-(b[1]-b'0') as usize;
        let ec = (b[2]-b'a') as usize; let er = 8-(b[3]-b'0') as usize;
        if sr < 8 && sc < 8 {
            let piece = engine.st.b[sr][sc];
            if ptype(piece) == b'p' && (er == 0 || er == 7) && best_move.len() == 4 {
                println!("bestmove {}q", best_move);
                return;
            }
        }
        println!("bestmove {}", best_move);
    } else {
        println!("bestmove 0000");
    }
}