use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use ember_chess::evaluate;
use ember_chess::Engine;

const DEFAULT_NODES: u64 = 5000;
const DEFAULT_DEPTH: i32 = 64;
const STATE_SAVE_EVERY: u64 = 1000;

fn log(msg: &str) {
    println!("[score_db] {msg}");
}

struct ScoreRecord {
    fen: String,
    score_type: String,
    score_val: i32,
    best_move: String,
    ply: String,
    result: String,
    position_index: u64,
}

fn score_position(
    engine: &mut Engine,
    fen: &str,
    nodes: u64,
    depth: i32,
    eval_only: bool,
) -> Option<(String, i32)> {
    engine.set_fen(fen);
    engine.searcher.stopped.store(false, Ordering::Relaxed);

    if eval_only {
        let score = engine.searcher.corrected_eval(&engine.st);
        return Some(("0000".to_string(), score));
    }

    let (best_move, score_cp, _nodes, _elapsed) =
        engine.find_best_move_prepared_untimed(depth, Some(nodes));
    if best_move == "0000" {
        None
    } else {
        Some((best_move, score_cp))
    }
}

fn state_path_for(out_dir: &Path) -> PathBuf {
    out_dir.join("score_db.state")
}

fn load_state(state_path: &Path) -> u64 {
    if let Ok(content) = fs::read_to_string(state_path) {
        if let Ok(n) = content.trim().parse::<u64>() {
            return n;
        }
    }
    0
}

fn save_state(state_path: &Path, next_index: u64) {
    if let Ok(mut f) = File::create(state_path) {
        let _ = writeln!(f, "{next_index}");
    }
}

fn find_next_part(out_dir: &Path) -> u64 {
    let mut max_part: u64 = 0;
    if let Ok(entries) = fs::read_dir(out_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("scored_") && name.ends_with(".tsv") {
                    let num: Option<u64> = name
                        .strip_prefix("scored_")
                        .and_then(|s| s.strip_suffix(".tsv"))
                        .and_then(|s| s.parse::<u64>().ok());
                    if let Some(n) = num {
                        if n >= max_part {
                            max_part = n + 1;
                        }
                    }
                }
            }
        }
    }
    max_part
}

fn chunk_writer_loop(
    rx: mpsc::Receiver<Vec<ScoreRecord>>,
    out_dir: &Path,
    chunk_bytes: u64,
    start_part: u64,
    state_path: &Path,
    next_index: Arc<AtomicU64>,
) {
    fs::create_dir_all(out_dir).unwrap();
    let mut part = start_part;
    let mut current_size: u64 = 0;
    let mut written: u64 = 0;
    let mut since_save: u64 = 0;

    let mut open_next = || -> BufWriter<File> {
        let p = out_dir.join(format!("scored_{:05}.tsv", part));
        part += 1;
        let f = File::create(&p).expect("failed to create scored chunk");
        log(&format!("opened output chunk {}", p.display()));
        BufWriter::new(f)
    };

    let mut current = Some(open_next());

    for batch in rx {
        for record in batch {
            let line = format!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                record.fen,
                record.score_type,
                record.score_val,
                record.best_move,
                record.ply,
                record.result
            );
            let data_len = line.len() as u64 + 1;
            if current_size >= chunk_bytes {
                if let Some(mut w) = current.take() {
                    w.flush().unwrap();
                }
                current = Some(open_next());
                current_size = 0;
            }
            let w = current.as_mut().unwrap();
            writeln!(w, "{line}").unwrap();
            current_size += data_len;
            written += 1;
            since_save += 1;

            let idx = record.position_index + 1;
            next_index.store(idx, Ordering::Relaxed);

            if since_save >= STATE_SAVE_EVERY {
                save_state(state_path, idx);
                since_save = 0;
            }
        }
    }

    if let Some(mut w) = current.take() {
        w.flush().unwrap();
    }

    save_state(state_path, next_index.load(Ordering::Relaxed));

    log(&format!(
        "SCORE DONE: {written} records written to {}",
        out_dir.display()
    ));
}

fn collect_input_files(inputs: &[String]) -> Vec<std::path::PathBuf> {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for glob_pattern in inputs {
        let path = Path::new(glob_pattern);
        if path.exists() && path.is_file() {
            files.push(path.to_path_buf());
        } else if let Some(parent) = path.parent() {
            if let Ok(entries) = fs::read_dir(parent) {
                let prefix = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                for e in entries.flatten() {
                    let p = e.path();
                    if p.to_string_lossy().contains(&prefix) {
                        files.push(p);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

#[allow(clippy::too_many_arguments)]
fn run_rescore(
    inputs: Vec<String>,
    out: &Path,
    workers: usize,
    nodes: u64,
    depth: i32,
    chunk_gb: f64,
    nnue_off: bool,
    nnue_on: bool,
    eval_only: bool,
    quiet: bool,
) -> std::io::Result<()> {
    let files = collect_input_files(&inputs);
    if files.is_empty() {
        eprintln!("No input files matched: {:?}", inputs);
        std::process::exit(1);
    }
    if !quiet {
        log(&format!("Found {} input files", files.len()));
    }

    let out_path = out.to_path_buf();
    fs::create_dir_all(&out_path).unwrap();
    let state_path = state_path_for(&out_path);
    let start_index = load_state(&state_path);
    if start_index > 0 {
        log(&format!("Resuming from position index {start_index}"));
    }

    let (task_tx, task_rx) = mpsc::channel::<(String, String, String, u64)>();
    let (score_tx, score_rx) = mpsc::channel::<Vec<ScoreRecord>>();

    let shared_rx = Arc::new(Mutex::new(task_rx));

    let feeder_tx = task_tx.clone();
    let feeder_files = files.clone();
    let feeder_start_index = Arc::new(AtomicU64::new(start_index));
    let feeder_handle = thread::spawn(move || {
        let mut global_index: u64 = 0;
        let mut total_queued: u64 = 0;

        let resume_from = feeder_start_index.load(Ordering::Relaxed);

        for path in &feeder_files {
            let f = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("failed to open {path:?}: {e}");
                    continue;
                }
            };
            let r = BufReader::new(f);
            let mut file_queued: u64 = 0;

            for line in r.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                if line.is_empty() {
                    continue;
                }

                if global_index < resume_from {
                    global_index += 1;
                    continue;
                }

                let parts: Vec<&str> = line.split('\t').collect();
                let fen = parts.first().unwrap_or(&"").to_string();
                if fen.is_empty() {
                    global_index += 1;
                    continue;
                }
                let ply = parts.get(4).unwrap_or(&"0").to_string();
                let result = parts.get(5).unwrap_or(&"0").to_string();

                if feeder_tx.send((fen, ply, result, global_index)).is_err() {
                    break;
                }
                file_queued += 1;
                total_queued += 1;
                global_index += 1;

                if total_queued.is_multiple_of(100_000) {
                    log(&format!("feeder: {} positions queued", total_queued));
                }
            }

            log(&format!(
                "feeder: file {} done — {} new",
                path.display(),
                file_queued
            ));
        }

        drop(feeder_tx);
        log(&format!(
            "feeder: done, {} total positions queued",
            total_queued
        ));
    });

    let start_part = find_next_part(&out_path);
    if start_part > 0 {
        log(&format!("resuming output chunks from part {start_part}"));
    }

    let writer_next_index = Arc::new(AtomicU64::new(start_index));
    let writer_state_path = state_path.clone();
    let writer_handle = thread::spawn(move || {
        chunk_writer_loop(
            score_rx,
            &out_path,
            (chunk_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            start_part,
            &writer_state_path,
            writer_next_index,
        );
    });

    let log_every = if quiet { 100_000 } else { 1_000 };
    let batch_cap = 64;

    let mut worker_handles = Vec::new();
    for i in 0..workers {
        let rx = Arc::clone(&shared_rx);
        let tx = score_tx.clone();
        let h = thread::spawn(move || {
            let mut engine = match Engine::try_new() {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("worker {i}: failed to create engine: {e}");
                    return;
                }
            };
            engine.set_hash_mb(256);
            engine.num_threads = 1;
            engine.quiet = quiet;

            if nnue_off {
                let _ = evaluate::reset_nnue();
            }
            if nnue_on {
                if let Err(e) = evaluate::init_embedded_nnue() {
                    eprintln!("worker {i}: failed to load embedded NNUE: {e}");
                }
            }
            engine.searcher.refresh_nnue_net();

            let mut processed: u64 = 0;
            let mut batch: Vec<ScoreRecord> = Vec::with_capacity(batch_cap);
            loop {
                let item = {
                    let guard = rx.lock().unwrap();
                    guard.recv().ok()
                };
                let Some((fen, ply, result, position_index)) = item else {
                    break;
                };
                match score_position(&mut engine, &fen, nodes, depth, eval_only) {
                    Some((best_move, score_cp)) => {
                        batch.push(ScoreRecord {
                            fen,
                            score_type: if eval_only { "ep" } else { "cp" }.to_string(),
                            score_val: score_cp,
                            best_move,
                            ply,
                            result,
                            position_index,
                        });
                    }
                    None => {
                        eprintln!("worker {i}: failed to rescore position: {fen}");
                    }
                }
                processed += 1;
                if processed.is_multiple_of(log_every) {
                    log(&format!("worker {i}: {processed} rescored"));
                }
                if batch.len() >= batch_cap && tx.send(std::mem::take(&mut batch)).is_err() {
                    break;
                }
            }
            if !batch.is_empty() {
                let _ = tx.send(batch);
            }
            log(&format!("worker {i}: finished {processed} rescored"));
        });
        worker_handles.push(h);
    }
    drop(task_tx);
    drop(score_tx);

    for h in worker_handles {
        let _ = h.join();
    }
    let _ = feeder_handle.join();
    let _ = writer_handle.join();

    let _ = fs::remove_file(&state_path);

    log("full run complete");
    Ok(())
}

fn parse_args() -> (String, Vec<(String, String)>) {
    let mut cmd = String::new();
    let mut opts: Vec<(String, String)> = Vec::new();
    let mut iter = env::args().skip(1).peekable();
    while let Some(arg) = iter.next() {
        if arg.starts_with("--") {
            let key = arg.clone();
            let val = if iter.peek().is_some_and(|next| next.starts_with("--")) {
                String::new()
            } else {
                iter.next().unwrap_or_default()
            };
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

    let inputs: Vec<String> = opts
        .iter()
        .filter(|(k, _)| k == "--input")
        .map(|(_, v)| v.clone())
        .collect();
    if inputs.is_empty() {
        eprintln!("--input required (glob pattern for positions_*.fen)");
        std::process::exit(1);
    }
    let out = get_opt(&opts, "--out").unwrap_or_else(|| "results/datagen/scored".to_string());
    let quiet = opts.iter().any(|(k, _)| k == "--quiet");
    let workers: usize = get_opt(&opts, "--workers")
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .to_string()
        })
        .parse()
        .expect("--workers must be a number");
    let nodes: u64 = get_opt(&opts, "--nodes")
        .unwrap_or_else(|| DEFAULT_NODES.to_string())
        .parse()
        .expect("--nodes must be a number");
    let depth: i32 = get_opt(&opts, "--depth")
        .unwrap_or_else(|| DEFAULT_DEPTH.to_string())
        .parse()
        .expect("--depth must be a number");
    let chunk_gb: f64 = get_opt(&opts, "--chunk-gb")
        .unwrap_or_else(|| "2".to_string())
        .parse()
        .expect("--chunk-gb must be a number");
    let nnue_off = opts.iter().any(|(k, _)| k == "--nnue-off");
    let nnue_on = opts.iter().any(|(k, _)| k == "--nnue-on");
    let eval_only = opts.iter().any(|(k, _)| k == "--eval-only");

    if cmd == "rescore" {
        let out_path = PathBuf::from(&out);
        if let Err(e) = run_rescore(
            inputs, &out_path, workers, nodes, depth, chunk_gb, nnue_off, nnue_on, eval_only, quiet,
        ) {
            eprintln!("rescore failed: {e}");
            std::process::exit(1);
        }
        log("full run complete");
        return;
    }

    if cmd != "score" {
        eprintln!(
            "Usage: score_db score --input <glob> [--input <glob> ...] --out <dir>\n\
             [--workers N] [--nodes N] [--depth N] [--chunk-gb N] [--nnue-off] [--nnue-on] [--quiet] [--eval-only]\n\
         score_db rescore --input <glob> [--input <glob> ...] --out <dir>\n\
             [--workers N] [--nodes N] [--depth N] [--chunk-gb N] [--nnue-off] [--nnue-on] [--quiet] [--eval-only]"
        );
        std::process::exit(1);
    }
    let out_path = PathBuf::from(&out);
    let state_path = state_path_for(&out_path);

    let start_index = load_state(&state_path);
    if start_index > 0 {
        log(&format!("Resuming from position index {start_index}"));
    }

    let files = collect_input_files(&inputs);
    if files.is_empty() {
        eprintln!("No input files matched: {:?}", inputs);
        std::process::exit(1);
    }
    log(&format!("Found {} input files", files.len()));

    let (task_tx, task_rx) = mpsc::channel::<(String, String, String, u64)>();
    let (score_tx, score_rx) = mpsc::channel::<Vec<ScoreRecord>>();

    let feeder_tx = task_tx.clone();
    let feeder_files = files.clone();
    let start_index = Arc::new(AtomicU64::new(start_index));
    let feeder_start_index = Arc::clone(&start_index);
    let feeder_handle = thread::spawn(move || {
        let mut global_index: u64 = 0;
        let mut total_queued: u64 = 0;
        let mut total_skipped: u64 = 0;

        let resume_from = feeder_start_index.load(Ordering::Relaxed);

        for path in &feeder_files {
            let f = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("failed to open {path:?}: {e}");
                    continue;
                }
            };
            let r = BufReader::new(f);
            let mut file_queued: u64 = 0;
            let file_skipped: u64 = 0;

            for line in r.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                if line.is_empty() {
                    continue;
                }

                if global_index < resume_from {
                    total_skipped += 1;
                    global_index += 1;
                    continue;
                }

                let parts: Vec<&str> = line.split('\t').collect();
                let fen = parts.first().unwrap_or(&"").to_string();
                let ply = parts.get(1).unwrap_or(&"0").to_string();
                let result = parts.get(2).unwrap_or(&"0").to_string();
                if feeder_tx.send((fen, ply, result, global_index)).is_err() {
                    break;
                }
                file_queued += 1;
                total_queued += 1;
                global_index += 1;

                if total_queued.is_multiple_of(100_000) {
                    log(&format!(
                        "feeder: {} positions queued ({} skipped/resumed)",
                        total_queued, total_skipped
                    ));
                }
            }

            total_skipped += file_skipped;
            log(&format!(
                "feeder: file {} done — {} new, {} skipped",
                path.display(),
                file_queued,
                file_skipped
            ));
        }

        drop(feeder_tx);
        log(&format!(
            "feeder: done, {} total positions queued, {} skipped",
            total_queued, total_skipped
        ));
    });

    let start_part = find_next_part(&out_path);
    if start_part > 0 {
        log(&format!("resuming output chunks from part {start_part}"));
    }

    let writer_next_index = Arc::clone(&start_index);
    let writer_state_path = state_path.clone();
    let writer_handle = thread::spawn(move || {
        chunk_writer_loop(
            score_rx,
            &out_path,
            (chunk_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            start_part,
            &writer_state_path,
            writer_next_index,
        );
    });

    let shared_rx = Arc::new(Mutex::new(task_rx));

    let log_every = if quiet { 100_000 } else { 1_000 };
    let batch_cap = 64;

    let mut worker_handles = Vec::new();
    for i in 0..workers {
        let rx = Arc::clone(&shared_rx);
        let tx = score_tx.clone();
        let h = thread::spawn(move || {
            let mut engine = match Engine::try_new() {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("worker {i}: failed to create engine: {e}");
                    return;
                }
            };
            engine.set_hash_mb(256);
            engine.num_threads = 1;
            engine.quiet = quiet;

            if nnue_off {
                let _ = evaluate::reset_nnue();
            }
            if nnue_on {
                if let Err(e) = evaluate::init_embedded_nnue() {
                    eprintln!("worker {i}: failed to load embedded NNUE: {e}");
                }
            }
            engine.searcher.refresh_nnue_net();

            let mut processed: u64 = 0;
            let mut batch: Vec<ScoreRecord> = Vec::with_capacity(batch_cap);
            loop {
                let item = {
                    let guard = rx.lock().unwrap();
                    guard.recv().ok()
                };
                let Some((fen, ply, result, position_index)) = item else {
                    break;
                };
                match score_position(&mut engine, &fen, nodes, depth, eval_only) {
                    Some((best_move, score_cp)) => {
                        batch.push(ScoreRecord {
                            fen,
                            score_type: if eval_only { "ep" } else { "cp" }.to_string(),
                            score_val: score_cp,
                            best_move,
                            ply,
                            result,
                            position_index,
                        });
                    }
                    None => {
                        eprintln!("worker {i}: failed to score position: {fen}");
                    }
                }
                processed += 1;
                if processed.is_multiple_of(log_every) {
                    log(&format!("worker {i}: {processed} scored"));
                }
                if batch.len() >= batch_cap && tx.send(std::mem::take(&mut batch)).is_err() {
                    break;
                }
            }
            if !batch.is_empty() {
                let _ = tx.send(batch);
            }
            log(&format!("worker {i}: finished {processed} scored"));
        });
        worker_handles.push(h);
    }
    drop(task_tx);
    drop(score_tx);

    for h in worker_handles {
        let _ = h.join();
    }

    let _ = feeder_handle.join();
    let _ = writer_handle.join();

    let _ = fs::remove_file(&state_path);

    log("full run complete");
}
