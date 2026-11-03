use std::cmp::Ordering;
use std::collections::BinaryHeap;
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
    println!("[fen_db] {msg}");
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
        let decoder =
            zstd::stream::read::Decoder::new(file).expect("failed to create zstd decoder");
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
            let t = token.trim_end_matches(['!', '?', '+', '#']);
            let t = t.strip_suffix("e.p.").unwrap_or(t);
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

fn flush_batch(
    batch: &mut Vec<String>,
    batch_idx: &mut u64,
    out_dirs: &[PathBuf],
    dir_idx: &mut usize,
) {
    if batch.is_empty() {
        return;
    }
    let out_dir = &out_dirs[*dir_idx % out_dirs.len()];
    *dir_idx += 1;
    let chunk_path = out_dir.join(format!(
        "chunk_{}_{:06}.fen",
        std::process::id(),
        *batch_idx
    ));
    let f = File::create(&chunk_path).expect("failed to create chunk file");
    let mut w = BufWriter::new(f);
    for line in &mut *batch {
        writeln!(w, "{line}").unwrap();
    }
    w.flush().unwrap();
    batch.clear();
    *batch_idx += 1;
}

#[allow(clippy::too_many_arguments)]
fn process_game_inner(
    current_game: &str,
    batch: &mut Vec<String>,
    batch_idx: &mut u64,
    out_dirs: &[PathBuf],
    dir_idx: &mut usize,
    strip_clocks: bool,
    max_ply: u32,
    total: &mut u64,
    skipped: &mut u64,
    game_num: u64,
) {
    if let Some((result_tag, san_moves)) = parse_pgn_moves(current_game) {
        let mut board = Board::default();
        let mut ply: u32 = 0;
        let game_start = std::time::Instant::now();

        for san in &san_moves {
            if ply >= max_ply {
                break;
            }
            let san_start = std::time::Instant::now();
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
                        flush_batch(batch, batch_idx, out_dirs, dir_idx);
                    }
                }
                Err(_) => break,
            }
            let san_elapsed = san_start.elapsed();
            if san_elapsed.as_millis() > 100 {
                eprintln!(
                    "SLOW SAN: ply={}, san='{}', took={} ms",
                    ply,
                    san,
                    san_elapsed.as_millis()
                );
            }
        }
        let elapsed = game_start.elapsed();
        if elapsed.as_secs_f64() > 1.0 {
            eprintln!(
                "SLOW GAME #{}: {:?} ms, san_count={}, game=\n{}",
                game_num,
                elapsed.as_millis(),
                san_moves.len(),
                current_game
            );
        }
    } else {
        eprintln!("WARNING: game #{}: parse error, skipping", game_num);
    }
}

#[allow(clippy::too_many_arguments)]
fn process_game(
    current_game: &str,
    batch: &mut Vec<String>,
    batch_idx: &mut u64,
    out_dirs: &[PathBuf],
    dir_idx: &mut usize,
    strip_clocks: bool,
    max_ply: u32,
    total: &mut u64,
    skipped: &mut u64,
    game_num: &mut u64,
) {
    *game_num += 1;
    let game_start = std::time::Instant::now();
    process_game_inner(
        current_game,
        batch,
        batch_idx,
        out_dirs,
        dir_idx,
        strip_clocks,
        max_ply,
        total,
        skipped,
        *game_num,
    );
    let elapsed = game_start.elapsed();
    if elapsed.as_secs_f64() > 0.5 {
        eprintln!(
            "SLOW GAME #{}: {:?} ms, {} san moves",
            game_num,
            elapsed.as_millis(),
            current_game.lines().count()
        );
    }
}

fn scan_pgn_file(
    path: &Path,
    tmp_dirs: &[PathBuf],
    dir_idx: &mut usize,
    strip_clocks: bool,
    max_ply: u32,
) -> (u64, u64) {
    for d in tmp_dirs {
        fs::create_dir_all(d).unwrap();
    }
    let mut total: u64 = 0;
    let mut skipped: u64 = 0;
    let reader = open_pgn_reader(path);

    let mut current_game: String = String::new();
    let mut batch: Vec<String> = Vec::with_capacity(BATCH_LINES);
    let mut batch_idx: u64 = 0;
    let mut seen_moves: bool = false;
    let mut game_count: u64 = 0;
    let mut line_count: u64 = 0;

    for line in reader.lines() {
        line_count += 1;
        if line_count.is_multiple_of(10000) {
            log(&format!(
                "Read {} lines, {} games, {} positions",
                line_count, game_count, total
            ));
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!(
                    "scan_pgn_file: read error at line {line_count}: {e}; stopping file scan"
                );
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current_game.is_empty() && seen_moves {
                game_count += 1;
                if game_count.is_multiple_of(1000) {
                    log(&format!(
                        "Processed {} games, {} positions",
                        game_count, total
                    ));
                }
                process_game(
                    &current_game,
                    &mut batch,
                    &mut batch_idx,
                    tmp_dirs,
                    dir_idx,
                    strip_clocks,
                    max_ply,
                    &mut total,
                    &mut skipped,
                    &mut game_count,
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
        game_count += 1;
        process_game(
            &current_game,
            &mut batch,
            &mut batch_idx,
            tmp_dirs,
            dir_idx,
            strip_clocks,
            max_ply,
            &mut total,
            &mut skipped,
            &mut game_count,
        );
    }
    flush_batch(&mut batch, &mut batch_idx, tmp_dirs, dir_idx);
    log(&format!(
        "Finished file: {} games, {} positions",
        game_count, total
    ));
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
    lines.sort_by_key(|a| line_key(a, strip_clocks));
}

#[derive(Eq)]
struct HeapItem {
    key: String,
    line: String,
    reader_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.cmp(&self.key)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

fn sort_state_path(out_dir: &Path) -> PathBuf {
    out_dir.join("sort.state")
}

fn save_sort_state(out_dir: &Path, processed: &[String], run_idx: u64) {
    let path = sort_state_path(out_dir);
    if let Ok(mut f) = File::create(&path) {
        for p in processed {
            let _ = writeln!(f, "{p}");
        }
        let _ = writeln!(f, "RUN_IDX:{}", run_idx);
    }
}

fn load_sort_state(out_dir: &Path) -> (Vec<String>, u64) {
    let path = sort_state_path(out_dir);
    let mut processed: Vec<String> = Vec::new();
    let mut run_idx: u64 = 0;
    if let Ok(content) = fs::read_to_string(&path) {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("RUN_IDX:") {
                run_idx = rest.trim().parse().unwrap_or(0);
            } else if !line.is_empty() {
                processed.push(line.to_string());
            }
        }
    }
    (processed, run_idx)
}

fn sort_and_dedup(tmp_dirs: &[PathBuf], out_dir: &Path, strip_clocks: bool, chunk_gb: f64) {
    fs::create_dir_all(out_dir).unwrap();
    let chunk_files = collect_chunk_files(tmp_dirs);
    if chunk_files.is_empty() {
        eprintln!("No .fen chunk files found in tmp dirs");
        std::process::exit(1);
    }
    log(&format!("Found {} chunk files", chunk_files.len()));

    let (mut processed_chunks, mut run_idx_global) = load_sort_state(out_dir);
    let processed_set: std::collections::HashSet<String> =
        processed_chunks.iter().cloned().collect();

    let mut run_files: Vec<PathBuf> = Vec::new();
    for d in tmp_dirs {
        if let Ok(entries) = fs::read_dir(d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|e| e.to_str()) == Some("tmp") {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("run_") {
                            run_files.push(p);
                        }
                    }
                }
            }
        }
    }

    for (file_idx, src) in chunk_files.iter().enumerate() {
        if processed_set.contains(src.to_str().unwrap_or("")) {
            continue;
        }
        log(&format!(
            "Sorting runs from {} ({}/{})",
            src.display(),
            file_idx + 1,
            chunk_files.len()
        ));
        let mut chunk: Vec<String> = Vec::with_capacity(RUN_LINES);
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
                let run = target_dir.join(format!("run_{:06}.tmp", run_idx_global));
                run_idx_global += 1;
                write_sorted_run(&mut chunk, &run);
                run_files.push(run);
            }
        }
        if !chunk.is_empty() {
            sort_by_key(&mut chunk, strip_clocks);
            let target_dir = &tmp_dirs[(run_idx_global % tmp_dirs.len() as u64) as usize];
            let run = target_dir.join(format!("run_{:06}.tmp", run_idx_global));
            run_idx_global += 1;
            write_sorted_run(&mut chunk, &run);
            run_files.push(run);
        }
        processed_chunks.push(src.to_str().unwrap_or("").to_string());
        save_sort_state(out_dir, &processed_chunks, run_idx_global);
    }

    log(&format!(
        "Produced {} sorted runs across {} tmp dir(s)",
        run_files.len(),
        tmp_dirs.len()
    ));

    log("Merging sorted runs -> output chunks (batched k-way merge)...");

    const BATCH_SIZE: usize = 200;

    run_files.sort();

    let mut batch_files: Vec<PathBuf> = Vec::new();
    for batch_start in (0..run_files.len()).step_by(BATCH_SIZE) {
        let batch_end = std::cmp::min(batch_start + BATCH_SIZE, run_files.len());
        let batch: Vec<&PathBuf> = run_files[batch_start..batch_end].iter().collect();
        let batch_out_path =
            tmp_dirs[0].join(format!("merged_batch_{:06}.tmp", batch_start / BATCH_SIZE));
        log(&format!(
            "Merging batch {}/{} ({} run files)",
            batch_start / BATCH_SIZE + 1,
            run_files.len().div_ceil(BATCH_SIZE),
            batch.len()
        ));

        let mut readers: Vec<BufReader<File>> = Vec::new();
        for r in &batch {
            let f = File::open(r).expect("failed to open run file");
            readers.push(BufReader::new(f));
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
        for (i, r) in readers.iter_mut().enumerate() {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap_or(0) > 0 {
                let line = line.trim_end().to_string();
                if !line.is_empty() {
                    let key = line_key(&line, strip_clocks);
                    heap.push(HeapItem {
                        key,
                        line,
                        reader_idx: i,
                    });
                }
            }
        }

        let mut batch_out =
            BufWriter::new(File::create(&batch_out_path).expect("failed to create batch output"));
        let mut batch_written: u64 = 0;
        let mut prev_key: Option<String> = None;

        while let Some(item) = heap.pop() {
            let key = item.key;
            let line = item.line;
            let reader_idx = item.reader_idx;

            if prev_key.as_ref() == Some(&key) {
            } else {
                prev_key = Some(key.clone());
                writeln!(batch_out, "{line}").unwrap();
                batch_written += 1;
            }

            let r = &mut readers[reader_idx];
            let mut next_line = String::new();
            if r.read_line(&mut next_line).unwrap_or(0) > 0 {
                let next_line = next_line.trim_end().to_string();
                if !next_line.is_empty() {
                    let next_key = line_key(&next_line, strip_clocks);
                    heap.push(HeapItem {
                        key: next_key,
                        line: next_line,
                        reader_idx,
                    });
                }
            }
        }

        batch_out.flush().unwrap();
        drop(batch_out);
        batch_files.push(batch_out_path);
        log(&format!(
            "Batch {} written: {} positions",
            batch_start / BATCH_SIZE + 1,
            batch_written
        ));
    }

    for r in &run_files {
        let _ = fs::remove_file(r);
    }

    log("Final merge of batch files (global dedup)...");

    let chunk_bytes = (chunk_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let mut part_idx: u64 = 0;
    let mut current_size: u64 = 0;
    let mut current: Option<BufWriter<File>> = None;
    let mut written: u64 = 0;
    let mut skipped_dup: u64 = 0;
    let mut seen_key: Option<String> = None;

    let mut readers: Vec<BufReader<File>> = Vec::new();
    for r in &batch_files {
        let f = File::open(r).expect("failed to open batch file");
        readers.push(BufReader::new(f));
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (i, r) in readers.iter_mut().enumerate() {
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) > 0 {
            let line = line.trim_end().to_string();
            if !line.is_empty() {
                let key = line_key(&line, strip_clocks);
                heap.push(HeapItem {
                    key,
                    line,
                    reader_idx: i,
                });
            }
        }
    }

    while let Some(item) = heap.pop() {
        let key = item.key;
        let line = item.line;
        let reader_idx = item.reader_idx;

        if seen_key.as_ref() == Some(&key) {
            skipped_dup += 1;
        } else {
            seen_key = Some(key.clone());

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
            if written.is_multiple_of(500_000) {
                log(&format!("Written {} positions...", written));
            }
        }

        let r = &mut readers[reader_idx];
        let mut next_line = String::new();
        if r.read_line(&mut next_line).unwrap_or(0) > 0 {
            let next_line = next_line.trim_end().to_string();
            if !next_line.is_empty() {
                let next_key = line_key(&next_line, strip_clocks);
                heap.push(HeapItem {
                    key: next_key,
                    line: next_line,
                    reader_idx,
                });
            }
        }
    }

    if let Some(mut w) = current.take() {
        w.flush().unwrap();
    }

    for r in &batch_files {
        let _ = fs::remove_file(r);
    }

    let _ = fs::remove_file(sort_state_path(out_dir));

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
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                        .to_string()
                })
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
            for p in pgn_files.iter() {
                let tx = tx.clone();
                let tmp_dirs = tmp_dirs.clone();
                let mut dir_idx: usize = 0;
                let p = p.clone();
                thread::spawn(move || {
                    let (total, skipped) =
                        scan_pgn_file(&p, &tmp_dirs, &mut dir_idx, strip_clocks, max_ply);
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
