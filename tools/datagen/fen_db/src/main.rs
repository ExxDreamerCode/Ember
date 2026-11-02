use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use chess::{Board, ChessMove, Color};

const BATCH_LINES: usize = 200_000;
const RUN_LINES: usize = 5_000_000;
const CHUNK_BYTES_DEFAULT: u64 = 2 * 1024 * 1024 * 1024;
const MAX_PLY_DEFAULT: u32 = 256;

fn log(msg: &str) {
    print!("[fen_db] {msg}\n");
    let _ = std::io::stdout().flush();
}

fn strip_clock_fields(fen: &str) -> String {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.len() >= 4 {
        format!("{} {} {} {}", parts[0], parts[1], parts[2], parts[3])
    } else {
        fen.to_string()
    }
}

fn result_to_stm(result_tag: &str, stm_white: bool) -> Option<i8> {
    match result_tag {
        "1-0" => Some(if stm_white { 1 } else { -1 }),
        "0-1" => Some(if stm_white { -1 } else { 1 }),
        "1/2-1/2" => Some(0),
        _ => None,
    }
}

fn open_pgn_reader(path: &Path) -> Box<dyn BufRead> {
    let file = File::open(path).expect("failed to open PGN file");
    if path.extension().and_then(|e| e.to_str()) == Some("zst") {
        let decoder = zstd::stream::read::Decoder::new(file).expect("failed to create zstd decoder");
        Box::new(BufReader::new(decoder))
    } else {
        Box::new(BufReader::new(file))
    }
}

fn remove_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_comment = false;
    for ch in s.chars() {
        if in_comment {
            if ch == '}' {
                in_comment = false;
            }
        } else if ch == '{' {
            in_comment = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn parse_pgn_moves(text: &str) -> Option<(&str, Vec<String>)> {
    let mut result_tag = "*";
    let mut in_headers = true;
    let mut move_tokens: Vec<String> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            in_headers = false;
            continue;
        }
        if in_headers {
            if line.starts_with('[') {
                if let Some(rest) = line.strip_prefix("[Result ") {
                    if let Some(val) = rest
                        .trim_end_matches(']')
                        .trim()
                        .strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                    {
                        result_tag = val;
                    }
                }
                continue;
            } else {
                in_headers = false;
            }
        }
        let cleaned = remove_comments(line);
        for token in cleaned.split_whitespace() {
            let t = token.trim_end_matches(|c| c == '!' || c == '?' || c == '+' || c == '#');
            let t = t.trim_end_matches("e.p.");
            if t.is_empty() || t == "1-0" || t == "0-1" || t == "1/2-1/2" || t == "*" {
                continue;
            }
            if t.chars().all(|c| c.is_ascii_digit() || c == '.') && t.contains('.') {
                continue;
            }
            move_tokens.push(t.to_string());
        }
    }

    if move_tokens.is_empty() {
        None
    } else {
        Some((result_tag, move_tokens))
    }
}

fn flush_batch(batch: &mut Vec<String>, batch_idx: &mut u64, out_dir: &Path) {
    if batch.is_empty() {
        return;
    }
    let chunk_path = out_dir.join(format!("chunk_{}_{:06}.fen", std::process::id(), *batch_idx));
    let f = File::create(&chunk_path).expect("failed to create chunk file");
    let mut w = BufWriter::new(f);
    for line in &mut *batch {
        writeln!(w, "{line}").unwrap();
    }
    w.flush().unwrap();
    batch.clear();
    *batch_idx += 1;
}

fn process_game(
    current_game: &str,
    batch: &mut Vec<String>,
    batch_idx: &mut u64,
    out_dir: &Path,
    strip_clocks: bool,
    max_ply: u32,
    total: &mut u64,
    skipped: &mut u64,
) {
    if let Some((result_tag, san_moves)) = parse_pgn_moves(current_game) {
        let mut board = Board::default();
        let mut ply: u32 = 0;

        for san in &san_moves {
            if ply >= max_ply {
                break;
            }
            match ChessMove::from_san(&board, san) {
                Ok(mv) => {
                    board = board.make_move_new(mv);
                    ply += 1;
                    let stm_white = board.side_to_move() == Color::White;
                    let result = result_to_stm(result_tag, stm_white);
                    if result.is_none() {
                        *skipped += 1;
                        continue;
                    }
                    let fen = board.to_string();
                    let key = if strip_clocks {
                        strip_clock_fields(&fen)
                    } else {
                        fen
                    };
                    batch.push(format!("{key}\t{ply}\t{}", result.unwrap()));
                    *total += 1;
                    if batch.len() >= BATCH_LINES {
                        flush_batch(batch, batch_idx, out_dir);
                    }
                }
                Err(_) => break,
            }
        }
    }
}

fn scan_pgn_file(path: &Path, out_dir: &Path, strip_clocks: bool, max_ply: u32) -> (u64, u64) {
    fs::create_dir_all(out_dir).unwrap();
    let mut total: u64 = 0;
    let mut skipped: u64 = 0;
    let reader = open_pgn_reader(path);

    let mut current_game: String = String::new();
    let mut batch: Vec<String> = Vec::with_capacity(BATCH_LINES);
    let mut batch_idx: u64 = 0;
    let mut seen_moves: bool = false;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current_game.is_empty() && seen_moves {
                process_game(
                    &current_game,
                    &mut batch,
                    &mut batch_idx,
                    out_dir,
                    strip_clocks,
                    max_ply,
                    &mut total,
                    &mut skipped,
                );
                current_game.clear();
                seen_moves = false;
            }
            continue;
        }
        current_game.push_str(&line);
        current_game.push('\n');
        if !trimmed.starts_with('[') {
            seen_moves = true;
        }
    }
    if !current_game.is_empty() {
        process_game(
            &current_game,
            &mut batch,
            &mut batch_idx,
            out_dir,
            strip_clocks,
            max_ply,
            &mut total,
            &mut skipped,
        );
    }
    flush_batch(&mut batch, &mut batch_idx, out_dir);
    (total, skipped)
}

fn collect_chunk_files(tmp_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for d in tmp_dirs {
        if let Ok(entries) = fs::read_dir(d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|e| e.to_str()) == Some("fen") {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    files
}

fn write_sorted_run(chunk: &mut Vec<String>, run: &Path) {
    chunk.sort();
    let f = File::create(run).expect("failed to create run file");
    let mut w = BufWriter::new(f);
    for line in &*chunk {
        writeln!(w, "{line}").unwrap();
    }
    w.flush().unwrap();
    chunk.clear();
}

fn line_key(line: &str, strip_clocks: bool) -> String {
    let fen = line.split('\t').next().unwrap_or(line);
    if strip_clocks {
        strip_clock_fields(fen)
    } else {
        fen.to_string()
    }
}

fn sort_by_key(lines: &mut [String], strip_clocks: bool) {
    lines.sort_by(|a, b| line_key(a, strip_clocks).cmp(&line_key(b, strip_clocks)));
}

fn sort_and_dedup(
    tmp_dirs: &[PathBuf],
    out_dir: &Path,
    strip_clocks: bool,
    chunk_gb: f64,
) {
    fs::create_dir_all(out_dir).unwrap();
    let chunk_files = collect_chunk_files(tmp_dirs);
    if chunk_files.is_empty() {
        eprintln!("No .fen chunk files found in tmp dirs");
        std::process::exit(1);
    }
    log(&format!("Found {} chunk files", chunk_files.len()));

    let mut run_files: Vec<PathBuf> = Vec::new();
    let mut run_idx_global: u64 = 0;
    for (file_idx, src) in chunk_files.iter().enumerate() {
        log(&format!(
            "Sorting runs from {} ({}/{})",
            src.display(),
            file_idx + 1,
            chunk_files.len()
        ));
        let mut chunk: Vec<String> = Vec::with_capacity(RUN_LINES);
        let mut run_idx: u64 = 0;
        let f = File::open(src).expect("failed to open chunk file");
        let r = BufReader::new(f);
        for line in r.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.is_empty() {
                continue;
            }
            chunk.push(line);
            if chunk.len() >= RUN_LINES {
                sort_by_key(&mut chunk, strip_clocks);
                let target_dir = &tmp_dirs[(run_idx_global % tmp_dirs.len() as u64) as usize];
                run_idx_global += 1;
                let run = target_dir.join(format!("run_{:06}.tmp", run_idx));
                write_sorted_run(&mut chunk, &run);
                run_files.push(run);
                run_idx += 1;
            }
        }
        if !chunk.is_empty() {
            sort_by_key(&mut chunk, strip_clocks);
            let target_dir = &tmp_dirs[(run_idx_global % tmp_dirs.len() as u64) as usize];
            run_idx_global += 1;
            let run = target_dir.join(format!("run_{:06}.tmp", run_idx));
            write_sorted_run(&mut chunk, &run);
            run_files.push(run);
        }
    }

    log(&format!(
        "Produced {} sorted runs across {} tmp dir(s)",
        run_files.len(),
        tmp_dirs.len()
    ));

    log("Merging sorted runs -> output chunks");
    let mut all_lines: Vec<String> = Vec::new();
    for r in &run_files {
        let f = File::open(r).expect("failed to open run file");
        let br = BufReader::new(f);
        for line in br.lines() {
            if let Ok(l) = line {
                if !l.is_empty() {
                    all_lines.push(l);
                }
            }
        }
        let _ = fs::remove_file(r);
    }

    sort_by_key(&mut all_lines, strip_clocks);

    let chunk_bytes = (chunk_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let mut part_idx: u64 = 0;
    let mut current_size: u64 = 0;
    let mut current: Option<BufWriter<File>> = None;
    let mut written: u64 = 0;
    let mut skipped_dup: u64 = 0;
    let mut seen_key: Option<String> = None;

    for line in all_lines {
        let key = line_key(&line, strip_clocks);
        if seen_key.as_ref() == Some(&key) {
            skipped_dup += 1;
            continue;
        }
        seen_key = Some(key);

        if current.is_none() || current_size >= chunk_bytes {
            if let Some(mut w) = current.take() {
                w.flush().unwrap();
            }
            let p = out_dir.join(format!("positions_{:05}.fen", part_idx));
            let f = File::create(&p).expect("failed to create output chunk");
            current = Some(BufWriter::new(f));
            part_idx += 1;
            current_size = 0;
            log(&format!("Writing chunk {}", part_idx - 1));
        }
        let w = current.as_mut().unwrap();
        writeln!(w, "{line}").unwrap();
        current_size += line.len() as u64 + 1;
        written += 1;
    }

    if let Some(mut w) = current.take() {
        w.flush().unwrap();
    }

    log(&format!(
        "SORT+UNIQ done: {} unique positions in {} chunk(s) under {} ({} dupes removed)",
        written,
        part_idx,
        out_dir.display(),
        skipped_dup
    ));
}

fn parse_args() -> (String, Vec<(String, String)>) {
    let mut cmd = String::new();
    let mut opts: Vec<(String, String)> = Vec::new();
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        if arg.starts_with("--") {
            let key = arg.clone();
            let val = iter.next().unwrap_or_default();
            opts.push((key, val));
        } else if cmd.is_empty() {
            cmd = arg;
        }
    }
    (cmd, opts)
}

fn get_opt(opts: &[(String, String)], key: &str) -> Option<String> {
    opts.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

fn main() {
    let (cmd, opts) = parse_args();

    match cmd.as_str() {
        "scan" => {
            let src = get_opt(&opts, "--src").expect("--src required");
            let tmp = get_opt(&opts, "--tmp").expect("--tmp required");
            let tmp2 = get_opt(&opts, "--tmp2");
            let workers: usize = get_opt(&opts, "--workers")
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).to_string())
                .parse()
                .expect("--workers must be a number");
            let strip_clocks = opts.iter().any(|(k, _)| k == "--strip-clocks");
            let max_ply: u32 = get_opt(&opts, "--max-ply")
                .unwrap_or_else(|| MAX_PLY_DEFAULT.to_string())
                .parse()
                .expect("--max-ply must be a number");

            let tmp_dirs: Vec<PathBuf> = match tmp2 {
                Some(t2) => vec![PathBuf::from(&tmp), PathBuf::from(&t2)],
                None => vec![PathBuf::from(&tmp)],
            };
            for d in &tmp_dirs {
                fs::create_dir_all(d).unwrap();
            }

            let src_path = Path::new(&src);
            let pgn_files: Vec<_> = fs::read_dir(src_path)
                .expect("failed to read src dir")
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let p = e.path();
                    matches!(
                        p.extension().and_then(|e2| e2.to_str()),
                        Some("pgn") | Some("epd") | Some("zst")
                    )
                })
                .map(|e| e.path())
                .collect();
            if pgn_files.is_empty() {
                eprintln!("No .pgn/.epd/.zst files found under {src}");
                std::process::exit(1);
            }

            log(&format!(
                "Found {} PGN/EPD/zst files; workers={}; tmp_dirs={:?}",
                pgn_files.len(),
                workers,
                tmp_dirs
                    .iter()
                    .map(|d| d.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
            ));

            let (tx, rx) = mpsc::channel();
            for (i, p) in pgn_files.iter().enumerate() {
                let tx = tx.clone();
                let out_dir = tmp_dirs[i % tmp_dirs.len()].clone();
                let p = p.clone();
                let strip_clocks = strip_clocks;
                let max_ply = max_ply;
                thread::spawn(move || {
                    let (total, skipped) = scan_pgn_file(&p, &out_dir, strip_clocks, max_ply);
                    tx.send((total, skipped)).unwrap();
                });
            }
            drop(tx);

            let mut total: u64 = 0;
            let mut skipped: u64 = 0;
            for _ in 0..pgn_files.len() {
                let (t, s) = rx.recv().unwrap();
                total += t;
                skipped += s;
                log(&format!("Scanned positions so far: {}", total));
            }
            log(&format!(
                "SCAN done: {} raw positions across {} tmp dir(s) ({} skipped without result)",
                total,
                tmp_dirs.len(),
                skipped
            ));
        }
        "sort" => {
            let tmp = get_opt(&opts, "--tmp").expect("--tmp required");
            let tmp2 = get_opt(&opts, "--tmp2");
            let out = get_opt(&opts, "--out").unwrap_or_else(|| "results/Datagen/fens".to_string());
            let chunk_gb: f64 = get_opt(&opts, "--chunk-gb")
                .unwrap_or_else(|| CHUNK_BYTES_DEFAULT.to_string())
                .parse()
                .expect("--chunk-gb must be a number");
            let strip_clocks = opts.iter().any(|(k, _)| k == "--strip-clocks");

            let tmp_dirs: Vec<PathBuf> = match tmp2 {
                Some(t2) => vec![PathBuf::from(&tmp), PathBuf::from(&t2)],
                None => vec![PathBuf::from(&tmp)],
            };
            sort_and_dedup(&tmp_dirs, Path::new(&out), strip_clocks, chunk_gb);
        }
        _ => {
            eprintln!(
                "Usage: fen_db <scan|sort> [options]\n\
                 scan  --src <dir> --tmp <dir> [--tmp2 <dir>] [--workers N] [--strip-clocks] [--max-ply N]\n\
                 sort  --tmp <dir> [--tmp2 <dir>] --out <dir> [--chunk-gb N] [--strip-clocks]"
            );
            std::process::exit(1);
        }
    }
}
