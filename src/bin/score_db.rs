use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use ember_chess::evaluate;
use ember_chess::Engine;

const DEFAULT_NODES: u64 = 5000;
const DEFAULT_DEPTH: i32 = 64;
const STATE_SAVE_EVERY: u64 = 10_000;

fn log(msg: &str) {
    print!("[score_db] {msg}\n");
    let _ = std::io::stdout().flush();
}

struct ScoreRecord {
    fen: String,
    score_type: String,
    score_val: i32,
    best_move: String,
    ply: String,
    result: String,
}

fn score_position(engine: &mut Engine, fen: &str, nodes: u64) -> Option<(String, i32)> {
    engine.set_fen(fen);
    let (best_move, score_cp, _nodes, _elapsed) =
        engine.find_best_move_prepared_untimed(DEFAULT_DEPTH, Some(nodes));
    if best_move == "0000" {
        None
    } else {
        Some((best_move, score_cp))
    }
}

fn state_path_for(out_dir: &Path) -> PathBuf {
    out_dir.join("score_db.state")
}

fn load_state(state_path: &Path) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    if let Ok(content) = fs::read_to_string(state_path) {
        for line in content.lines() {
            if let Some((p, count)) = line.split_once('\t') {
                if let Ok(n) = count.parse::<u64>() {
                    map.insert(p.to_string(), n);
                }
            }
        }
    }
    map
}

fn save_state(state_path: &Path, state: &HashMap<String, u64>) {
    let mut lines: Vec<String> = state
        .iter()
        .map(|(p, c)| format!("{}\t{}", p, c))
        .collect();
    lines.sort();
    if let Ok(mut f) = File::create(state_path) {
        for l in &lines {
            let _ = writeln!(f, "{l}");
        }
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
    rx: mpsc::Receiver<ScoreRecord>,
    out_dir: &Path,
    chunk_bytes: u64,
    start_part: u64,
) {
    fs::create_dir_all(out_dir).unwrap();
    let mut part = start_part;
    let mut current_size: u64 = 0;
    let mut written: u64 = 0;

    let mut open_next = || -> BufWriter<File> {
        let p = out_dir.join(format!("scored_{:05}.tsv", part));
        part += 1;
        let f = File::create(&p).expect("failed to create scored chunk");
        log(&format!("opened output chunk {}", p.display()));
        BufWriter::new(f)
    };

    let mut current = Some(open_next());

    for record in rx {
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
    }

    if let Some(mut w) = current.take() {
        w.flush().unwrap();
    }

    log(&format!(
        "SCORE DONE: {written} records written to {}",
        out_dir.display()
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

    if cmd != "score" {
        eprintln!(
            "Usage: score_db score --input <glob> [--input <glob> ...] --out <dir>\n\
             [--workers N] [--nodes N] [--chunk-gb N] [--nnue-off] [--state <path>]"
        );
        std::process::exit(1);
    }

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
    let workers: usize = get_opt(&opts, "--workers")
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).to_string())
        .parse()
        .expect("--workers must be a number");
    let nodes: u64 = get_opt(&opts, "--nodes")
        .unwrap_or_else(|| DEFAULT_NODES.to_string())
        .parse()
        .expect("--nodes must be a number");
    let chunk_gb: f64 = get_opt(&opts, "--chunk-gb")
        .unwrap_or_else(|| "2".to_string())
        .parse()
        .expect("--chunk-gb must be a number");
    let nnue_off = opts.iter().any(|(k, _)| k == "--nnue-off");
    let state_path = get_opt(&opts, "--state")
        .map(PathBuf::from)
        .unwrap_or_else(|| state_path_for(Path::new(&out)));

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for glob_pattern in &inputs {
        let path = Path::new(glob_pattern);
        if path.exists() && path.is_file() {
            files.push(path.to_path_buf());
        } else if let Some(parent) = path.parent() {
            if let Ok(entries) = fs::read_dir(parent) {
                let prefix = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                for e in entries.flatten() {
                    let p = e.path();
                    if p.to_string_lossy().contains(&prefix) {
                        files.push(p);
                    }
                }
            }
        }
    }
    if files.is_empty() {
        eprintln!("No input files matched: {:?}", inputs);
        std::process::exit(1);
    }
    files.sort();
    log(&format!("Found {} input files", files.len()));

    let state = load_state(&state_path);
    if !state.is_empty() {
        log(&format!("Resume state loaded: {} files with prior progress", state.len()));
    }

    let (task_tx, task_rx) = mpsc::channel::<(String, String, String)>();
    let (score_tx, score_rx) = mpsc::channel::<ScoreRecord>();

    let feeder_tx = task_tx.clone();
    let feeder_files = files.clone();
    let feeder_state = state.clone();
    let feeder_state_path = state_path.clone();
    let feeder_handle = thread::spawn(move || {
        let mut total_queued: u64 = 0;
        let mut total_skipped: u64 = 0;
        let mut since_save: u64 = 0;
        let mut local_state = feeder_state;

        for path in &feeder_files {
            let abs_path = match path.canonicalize() {
                Ok(p) => p.to_string_lossy().to_string(),
                Err(_) => path.to_string_lossy().to_string(),
            };
            let skip_lines = local_state.get(&abs_path).copied().unwrap_or(0);

            let f = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("failed to open {path:?}: {e}");
                    continue;
                }
            };
            let r = BufReader::new(f);
            let mut line_idx: u64 = 0;
            let mut file_queued: u64 = 0;
            let mut file_skipped: u64 = 0;

            for line in r.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                if line.is_empty() {
                    continue;
                }

                line_idx += 1;

                if line_idx <= skip_lines {
                    file_skipped += 1;
                    continue;
                }

                let parts: Vec<&str> = line.split('\t').collect();
                let fen = parts.get(0).unwrap_or(&"").to_string();
                let ply = parts.get(1).unwrap_or(&"0").to_string();
                let result = parts.get(2).unwrap_or(&"0").to_string();
                if feeder_tx.send((fen, ply, result)).is_err() {
                    break;
                }
                file_queued += 1;
                total_queued += 1;
                since_save += 1;

                if since_save >= STATE_SAVE_EVERY {
                    local_state.insert(abs_path.clone(), line_idx);
                    save_state(&feeder_state_path, &local_state);
                    since_save = 0;
                }

                if total_queued % 100_000 == 0 {
                    log(&format!(
                        "feeder: {} positions queued ({} skipped/resumed)",
                        total_queued,
                        total_skipped + file_skipped
                    ));
                }
            }

            local_state.insert(abs_path, line_idx);
            total_skipped += file_skipped;
            log(&format!(
                "feeder: file {} done — {} new, {} skipped (resume)",
                path.display(),
                file_queued,
                file_skipped
            ));
        }

        save_state(&feeder_state_path, &local_state);
        drop(feeder_tx);
        log(&format!(
            "feeder: done, {} total positions queued, {} skipped (resume)",
            total_queued, total_skipped
        ));
    });

    let out_path = PathBuf::from(&out);
    let start_part = find_next_part(&out_path);
    if start_part > 0 {
        log(&format!("resuming output chunks from part {start_part}"));
    }
    let writer_handle = thread::spawn(move || {
        chunk_writer_loop(
            score_rx,
            &out_path,
            (chunk_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            start_part,
        );
    });

    let shared_rx = Arc::new(Mutex::new(task_rx));

    let mut worker_handles = Vec::new();
    for i in 0..workers {
        let rx = Arc::clone(&shared_rx);
        let tx = score_tx.clone();
        let nodes = nodes;
        let nnue_off = nnue_off;
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

            if nnue_off {
                let _ = evaluate::reset_nnue();
            }

            let mut processed: u64 = 0;
            loop {
                let item = {
                    let guard = rx.lock().unwrap();
                    guard.recv().ok()
                };
                let Some((fen, ply, result)) = item else {
                    break;
                };
                match score_position(&mut engine, &fen, nodes) {
                    Some((best_move, score_cp)) => {
                        let _ = tx.send(ScoreRecord {
                            fen,
                            score_type: "cp".to_string(),
                            score_val: score_cp,
                            best_move,
                            ply,
                            result,
                        });
                    }
                    None => {
                        eprintln!("worker {i}: failed to score position: {fen}");
                    }
                }
                processed += 1;
                if processed % 1000 == 0 {
                    log(&format!("worker {i}: {processed} scored"));
                }
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
    log("state cleared — full run complete");
}
