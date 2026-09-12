use ember_chess::backend::{
    compiled_search_backends, parse_search_backend_name, search_backend_available,
};
use ember_chess::board::{piece_on, piece_type, EMPTY_SQ};
use ember_chess::book::{DEFAULT_BOOK_MIN_MOVE_WEIGHT, DEFAULT_BOOK_MIN_MOVE_WEIGHT_PERMILLE};
use ember_chess::evaluate;
use ember_chess::search::{
    active_search_backend, set_search_backend_override, SearchLearning, SEARCH_THREAD_STACK_SIZE,
};
use ember_chess::syzygy::SyzygyTables;
use ember_chess::time_management::TimeManager;
use ember_chess::tune::{self, TuneParam};
use ember_chess::zobrist::compute_hash;
use ember_chess::{book, Engine, EngineBookConfig, OpeningBook};
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[cfg(all(
    feature = "mimalloc",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
))]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MIN_HASH_MB: usize = 1;
const MAX_HASH_MB: usize = 4096;
const MIN_THREADS: usize = 1;
const MAX_THREADS: usize = 256;
const MAX_MULTI_PV: usize = 256;
const SHORT_SYNC_SEARCH_LIMIT_SECONDS: f64 = 0.050;
const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const BENCH_DEFAULT_DEPTH: i32 = 10;
const BENCH_MAX_DEPTH: i32 = 64;
const BENCH_TIME_LIMIT_SECONDS: f64 = 1_000_000_000.0;
const BENCH_POSITIONS: &[(&str, &str)] = &[
    ("startpos", "startpos"),
    (
        "kiwipete",
        "fen r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QKB1R w KQkq - 0 1",
    ),
    (
        "sicilian",
        "fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 8",
    ),
    (
        "queenless-middlegame",
        "fen 2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14",
    ),
    (
        "tactical",
        "fen r2q1rk1/ppp2ppp/2n1bn2/3pp3/1b2P3/2NP1N2/PPPBBPPP/R2Q1RK1 w - - 0 8",
    ),
    (
        "endgame-rooks",
        "fen 8/2p2pk1/1p4p1/p2Pp3/P1P1P1P1/1P3K2/8/8 w - - 0 40",
    ),
    (
        "minor-piece-endgame",
        "fen 8/5pk1/6p1/3N4/3P4/5P2/6PK/8 w - - 0 45",
    ),
    ("promotion-race", "fen 8/1P6/8/8/8/8/6p1/6Kk w - - 0 1"),
];
const FNV1A_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

struct SearchTask {
    id: u64,
    handle: Option<thread::JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    rx: mpsc::Receiver<SearchCompletion>,
    completion: Option<SearchCompletion>,
}

impl SearchTask {
    fn request_stop(&self) {
        self.pondering.store(false, Ordering::SeqCst);
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn collect_completion(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
        if self.completion.is_none() {
            self.completion = self.rx.recv().ok();
        }
    }

    fn has_finished(&self) -> bool {
        self.completion.is_some()
            || self
                .handle
                .as_ref()
                .is_some_and(thread::JoinHandle::is_finished)
    }
}

struct SearchCompletion {
    result: (String, i32, u64, f64),
    learning: SearchLearning,
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
    node_limit: Option<u64>,
    clock_managed: bool,
    ponder: bool,
}

fn apply_search_completion(
    engine: &mut Engine,
    task: &mut SearchTask,
    emit_bestmove: bool,
    ponder_enabled: bool,
) {
    if let Some(completion) = task.completion.take() {
        engine.searcher.import_learning(&completion.learning);
        if emit_bestmove {
            print_bestmove(engine, &completion.result.0, ponder_enabled);
        }
    }
}

fn cancel_search(
    engine: &mut Engine,
    task: &mut Option<SearchTask>,
    emit_bestmove: bool,
    ponder_enabled: bool,
) {
    let Some(mut running) = task.take() else {
        return;
    };
    running.request_stop();
    running.collect_completion();
    apply_search_completion(engine, &mut running, emit_bestmove, ponder_enabled);
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
        Ok(()) => true,
        Err(e) => {
            eprintln!("info string Failed to load NNUE ({}): {}", path, e);
            false
        }
    }
}

fn main() {
    let main_thread = std::thread::Builder::new()
        .name("main".into())
        .stack_size(SEARCH_THREAD_STACK_SIZE)
        .spawn(run_uci_loop)
        .expect("failed to spawn main thread");
    main_thread.join().expect("main thread panicked");
}

fn run_uci_loop() {
    let mut engine = Engine::new();
    let mut time_manager = TimeManager::default();
    let mut search_task: Option<SearchTask> = None;
    let mut next_search_id = 0u64;
    let mut ponder_enabled = false;

    eprintln!("info string Loading embedded NNUE...");
    match evaluate::init_embedded_nnue() {
        Ok(()) => eprintln!("info string Embedded NNUE loaded"),
        Err(e) => eprintln!("info string Failed to load embedded NNUE: {}", e),
    }

    eprintln!("info string Loading embedded book...");
    match OpeningBook::load_from_bytes(book::BOOK_DATA, "<embedded>") {
        Ok(book) => {
            engine.book = Some(book);
            eprintln!("info string Embedded book loaded");
        }
        Err(e) => {
            eprintln!("info string Failed to load embedded book: {}", e);
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
                let should_emit =
                    if let Some(task) = search_task.as_mut().filter(|task| task.id == id) {
                        task.collect_completion();
                        !task.pondering.load(Ordering::Relaxed)
                    } else {
                        false
                    };
                if should_emit {
                    let mut task = search_task.take().unwrap();
                    apply_search_completion(&mut engine, &mut task, true, ponder_enabled);
                }
                continue;
            }
            UciEvent::InputClosed => {
                cancel_search(&mut engine, &mut search_task, false, ponder_enabled);
                break;
            }
        };
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "uci" => {
                println!("id name Ember 1.3.0");
                println!("id author ExxDreamerCode");
                println!(
                    "option name Hash type spin default 256 min {} max {}",
                    MIN_HASH_MB, MAX_HASH_MB
                );
                println!(
                    "option name Threads type spin default 1 min {} max {}",
                    MIN_THREADS, MAX_THREADS
                );
                println!(
                    "option name MultiPV type spin default 1 min 1 max {}",
                    MAX_MULTI_PV
                );
                println!("option name Move Overhead type spin default 7 min 0 max 5000");
                println!("option name Ponder type check default false");
                println!("option name Book type string default <embedded>");
                println!("option name RandomBookMove type check default false");
                println!(
                    "option name BookMinMoveWeight type spin default {} min 1 max 65535",
                    DEFAULT_BOOK_MIN_MOVE_WEIGHT
                );
                println!(
                    "option name BookMinMoveWeightPermille type spin default {} min 0 max 1000",
                    DEFAULT_BOOK_MIN_MOVE_WEIGHT_PERMILLE
                );
                println!("option name NNUE type string default <embedded>");
                print!("option name NNUEBackend type combo default auto var auto");
                for backend in compiled_search_backends() {
                    print!(" var {}", backend.name());
                }
                println!();
                println!("option name SyzygyPath type string default <empty>");
                println!("option name UCI_Chess960 type check default false");
                println!("option name Tune type string default <empty>");
                #[cfg(feature = "decision-trace")]
                println!("option name TraceFile type string default <empty>");
                println!("uciok");
            }
            "isready" => {
                engine.ensure_hash_ready();
                println!("readyok");
            }
            "ucinewgame" => {
                cancel_search(&mut engine, &mut search_task, false, ponder_enabled);
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
                            match OpeningBook::load_from_bytes(book::BOOK_DATA, "<embedded>") {
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
                                Ok(()) => {}
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
                    "tune" => {
                        if val.is_empty() {
                            tune::reset();
                            eprintln!("info string Tune overrides cleared");
                        } else {
                            let parsed = parse_tune_value(&val);
                            match parsed {
                                Ok(overrides) => {
                                    for (param, value) in overrides {
                                        tune::set(param, value);
                                        eprintln!("info string Tune {} = {}", param.name(), value);
                                    }
                                }
                                Err(message) => {
                                    eprintln!(
                                        "info string Ignoring invalid Tune value: {}",
                                        message
                                    )
                                }
                            }
                        }
                    }
                    "move overhead" => {
                        let parsed = val.parse::<f64>();
                        if !parsed.is_ok_and(|value| time_manager.set_move_overhead_ms(value)) {
                            eprintln!("info string Ignoring out-of-range Move Overhead: {}", val);
                        }
                    }
                    "ponder" => match parse_check_value(&val) {
                        Some(enable) => ponder_enabled = enable,
                        None => eprintln!("info string Ignoring invalid Ponder value: {}", val),
                    },
                    _ => {
                        parse_setoption(&mut engine, &name, &val);
                    }
                }
            }
            "tune" => {
                let overrides = tune::active_overrides();
                if overrides.is_empty() {
                    println!("info string tune: no active overrides");
                } else {
                    for (param, value) in overrides {
                        println!("info string tune {} = {}", param.name(), value);
                    }
                }
            }
            "eval" => {
                let score = evaluate::evaluate_nnue(&engine.st);
                let classic = ember_chess::evaluate::evaluate(&engine.st);
                println!("info string NNUE eval: {} cp (from stm)", score);
                let stm_sign = if engine.st.w { 1 } else { -1 };
                println!(
                    "info string Classic eval: {} cp (from white), {} cp (from stm)",
                    classic,
                    classic * stm_sign
                );
            }
            "position" => {
                cancel_search(&mut engine, &mut search_task, false, ponder_enabled);
                parse_position(&mut engine, &parts);
            }
            "go" => {
                if search_task.is_some() {
                    continue;
                }

                let limits = parse_go_params(&parts, &engine, &mut time_manager);
                let search_start = Instant::now();
                if !limits.ponder
                    && limits.clock_managed
                    && limits.hard_seconds <= SHORT_SYNC_SEARCH_LIMIT_SECONDS
                {
                    let result = engine.find_best_move_with_time_limits_started_at(
                        limits.soft_seconds,
                        limits.hard_seconds,
                        limits.depth,
                        limits.node_limit,
                        search_start,
                    );
                    print_bestmove(&engine, &result.0, ponder_enabled);
                    continue;
                }
                let search_id = next_search_id;
                next_search_id = next_search_id.wrapping_add(1);

                let st = engine.st;
                let shared_tt = Arc::clone(&engine.shared_tt);
                let search_pool = Arc::clone(&engine.search_pool);
                let stopped = Arc::new(AtomicBool::new(false));
                let pondering = Arc::new(AtomicBool::new(limits.ponder));
                let num_threads = engine.num_threads;
                let multi_pv = engine.multi_pv;
                let book = if limits.ponder {
                    None
                } else {
                    engine.book.clone()
                };
                let book_config = EngineBookConfig::new(
                    book,
                    engine.book_min_move_weight,
                    engine.book_min_move_weight_permille,
                )
                .with_random_book_move(engine.random_book_move);

                let mut search_searcher = ember_chess::search::Searcher::new(
                    Arc::clone(&shared_tt),
                    Arc::clone(&stopped),
                );
                engine.searcher.copy_root_context_to(&mut search_searcher);
                search_searcher.pondering = Arc::clone(&pondering);
                search_searcher.tt_mb = engine.searcher.tt_mb;
                #[cfg(feature = "decision-trace")]
                let trace_path = engine.trace.path().map(|p| p.display().to_string());

                let stopped_for_search = Arc::clone(&stopped);
                let search_finished_tx = event_tx.clone();
                let (tx, rx) = mpsc::channel();

                let handle = thread::Builder::new()
                    .name("search".into())
                    .stack_size(SEARCH_THREAD_STACK_SIZE)
                    .spawn(move || {
                        let mut search_engine = Engine::new_with(
                            st,
                            search_searcher,
                            shared_tt,
                            search_pool,
                            num_threads,
                            stopped_for_search,
                            book_config,
                        );
                        search_engine.multi_pv = multi_pv;
                        #[cfg(feature = "decision-trace")]
                        if let Some(tp) = trace_path {
                            search_engine.set_trace_file(&tp);
                        }
                        let result = if limits.clock_managed {
                            search_engine.find_best_move_with_time_limits_prepared_started_at(
                                limits.soft_seconds,
                                limits.hard_seconds,
                                limits.depth,
                                limits.node_limit,
                                search_start,
                            )
                        } else {
                            search_engine.find_best_move_with_time_limits_prepared_with_node_limit(
                                limits.soft_seconds,
                                limits.hard_seconds,
                                limits.depth,
                                limits.node_limit,
                            )
                        };
                        let learning = search_engine.searcher.export_learning();
                        tx.send(SearchCompletion { result, learning }).ok();
                        search_finished_tx
                            .send(UciEvent::SearchFinished(search_id))
                            .ok();
                    })
                    .expect("failed to spawn search thread");

                search_task = Some(SearchTask {
                    id: search_id,
                    handle: Some(handle),
                    stopped,
                    pondering,
                    rx,
                    completion: None,
                });
            }
            "ponderhit" => {
                let should_emit = if let Some(task) = search_task.as_mut() {
                    task.pondering.store(false, Ordering::SeqCst);
                    if task.has_finished() {
                        task.collect_completion();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if should_emit {
                    let mut task = search_task.take().unwrap();
                    apply_search_completion(&mut engine, &mut task, true, ponder_enabled);
                }
            }
            "bench" => {
                cancel_search(&mut engine, &mut search_task, false, ponder_enabled);
                let (depth, count) = parse_bench_params(&parts);
                run_bench(&engine, depth, count);
            }
            "stop" => {
                cancel_search(&mut engine, &mut search_task, true, ponder_enabled);
            }
            "quit" => {
                cancel_search(&mut engine, &mut search_task, false, ponder_enabled);
                break;
            }
            _ => {}
        }
    }
}

fn print_bestmove(engine: &Engine, best_move: &str, ponder_enabled: bool) {
    let normalized = if best_move.len() >= 4 && best_move != "0000" {
        let mut normalized = best_move.to_string();
        if normalized.len() == 4 {
            let b = normalized.as_bytes();
            let sc = (b[0] - b'a') as usize;
            let sr = 8 - (b[1] - b'0') as usize;
            let er = 8 - (b[3] - b'0') as usize;
            if sr < 8 && sc < 8 && er < 8 {
                let piece_idx = piece_on(&engine.st.bb, sr * 8 + sc);
                if piece_idx != EMPTY_SQ && piece_type(piece_idx) == 0 && (er == 0 || er == 7) {
                    normalized.push('q');
                }
            }
        }
        Some(normalized)
    } else {
        None
    };

    if let Some(best_move) = normalized {
        if ponder_enabled {
            if let Some(ponder_move) = engine.ponder_move_after(&best_move) {
                println!("bestmove {} ponder {}", best_move, ponder_move);
                return;
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

fn parse_tune_value(value: &str) -> Result<Vec<(TuneParam, i64)>, &'static str> {
    let value = value.trim();
    let value = value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|rest| rest.strip_suffix('\''))
        })
        .unwrap_or(value)
        .trim();
    if value.is_empty() {
        return Ok(Vec::new());
    }
    if value.contains('=') {
        let mut overrides = Vec::new();
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let Some((name, raw)) = token.split_once('=') else {
                return Err("expected NAME=VALUE tokens separated by commas");
            };
            let Some(param) = TuneParam::from_name(name.trim()) else {
                return Err("unknown tune parameter");
            };
            let parsed = raw
                .trim()
                .parse::<i64>()
                .map_err(|_| "invalid tune value")?;
            overrides.push((param, parsed));
        }
        return Ok(overrides);
    }
    let mut parts = value.split_whitespace();
    let name = parts.next().unwrap_or_default();
    let raw = parts.next().unwrap_or_default();
    if parts.next().is_some() {
        return Err("expected NAME value");
    }
    let Some(param) = TuneParam::from_name(name) else {
        return Err("unknown tune parameter");
    };
    let parsed = raw.parse::<i64>().map_err(|_| "invalid tune value")?;
    Ok(vec![(param, parsed)])
}

fn parse_check_value(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn set_nnue_backend(value: &str) {
    let normalized = ember_chess::backend::normalize_backend_name(value);
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
        "multipv" => {
            if let Ok(n) = val.parse::<usize>() {
                if (1..=MAX_MULTI_PV).contains(&n) {
                    engine.multi_pv = n;
                    eprintln!("info string Set MultiPV to {}", engine.multi_pv);
                } else {
                    eprintln!("info string Ignoring out-of-range MultiPV value: {}", n);
                }
            }
        }
        "randombookmove" | "random book move" => match parse_check_value(val) {
            Some(enabled) => {
                engine.random_book_move = enabled;
                eprintln!("info string Set RandomBookMove to {}", enabled);
            }
            None => eprintln!("info string Ignoring invalid RandomBookMove value: {}", val),
        },
        "bookminmoveweight" | "book min move weight" => {
            if let Ok(weight) = val.parse::<u16>() {
                if weight >= 1 {
                    engine.book_min_move_weight = weight;
                    eprintln!("info string Set BookMinMoveWeight to {}", weight);
                } else {
                    eprintln!(
                        "info string Ignoring out-of-range BookMinMoveWeight: {}",
                        val
                    );
                }
            }
        }
        "bookminmoveweightpermille" | "book min move weight permille" => {
            if let Ok(permille) = val.parse::<u16>() {
                if permille <= 1000 {
                    engine.book_min_move_weight_permille = permille;
                    eprintln!("info string Set BookMinMoveWeightPermille to {}", permille);
                } else {
                    eprintln!(
                        "info string Ignoring out-of-range BookMinMoveWeightPermille: {}",
                        val
                    );
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
    let multi_pv = engine.multi_pv;
    let search_pool = Arc::clone(&engine.search_pool);
    let chess960 = engine.st.chess960;
    let syzygy = engine.searcher.syzygy.clone();
    let random_book_move = engine.random_book_move;
    let book_min_move_weight = engine.book_min_move_weight;
    let book_min_move_weight_permille = engine.book_min_move_weight_permille;
    #[cfg(feature = "decision-trace")]
    let trace = std::mem::take(&mut engine.trace);
    let tt_mb = engine.searcher.tt_mb;
    search_pool.clear_learning();
    *engine = Engine::new();
    engine.book = book;
    engine.random_book_move = random_book_move;
    engine.book_min_move_weight = book_min_move_weight;
    engine.book_min_move_weight_permille = book_min_move_weight_permille;
    engine.search_pool = search_pool;
    engine.num_threads = num_threads;
    engine.multi_pv = multi_pv;
    engine.searcher.syzygy = syzygy;
    set_chess960_mode(engine, chess960);
    #[cfg(feature = "decision-trace")]
    {
        engine.trace = trace;
    }
    engine.searcher.tt_mb = tt_mb;
    engine.ensure_hash_ready();
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
    if !matches!(mv.len(), 4 | 5) {
        return None;
    }
    let b = mv.as_bytes();
    if !(b'a'..=b'h').contains(&b[0])
        || !(b'1'..=b'8').contains(&b[1])
        || !(b'a'..=b'h').contains(&b[2])
        || !(b'1'..=b'8').contains(&b[3])
    {
        return None;
    }
    let sc = (b[0] - b'a') as usize;
    let sr = (b'8' - b[1]) as usize;
    let ec = (b[2] - b'a') as usize;
    let er = (b'8' - b[3]) as usize;
    let promotion = if mv.len() >= 5 {
        match b[4] {
            b'q' | b'Q' => b'Q',
            b'r' | b'R' => b'R',
            b'b' | b'B' => b'B',
            b'n' | b'N' => b'N',
            _ => return None,
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
    let mut node_limit = None;
    let mut movestogo = 0i32;
    let mut ponder = false;
    let mut has_clock_limit = false;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "wtime" if i + 1 < parts.len() => {
                wtime = parts[i + 1].parse().unwrap_or(300000.0);
                has_clock_limit = true;
                i += 1;
            }
            "btime" if i + 1 < parts.len() => {
                btime = parts[i + 1].parse().unwrap_or(300000.0);
                has_clock_limit = true;
                i += 1;
            }
            "winc" if i + 1 < parts.len() => {
                winc = parts[i + 1].parse().unwrap_or(0.0);
                has_clock_limit = true;
                i += 1;
            }
            "binc" if i + 1 < parts.len() => {
                binc = parts[i + 1].parse().unwrap_or(0.0);
                has_clock_limit = true;
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
            "nodes" if i + 1 < parts.len() => {
                node_limit = parts[i + 1].parse::<u64>().ok().filter(|&nodes| nodes > 0);
                i += 1;
            }
            "movestogo" if i + 1 < parts.len() => {
                movestogo = parts[i + 1].parse().unwrap_or(0);
                has_clock_limit = true;
                i += 1;
            }
            "infinite" => {
                movetime = 1_000_000.0;
            }
            "ponder" => {
                ponder = true;
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
    } else if depth < 64 || (node_limit.is_some() && !has_clock_limit) {
        (1_000_000_000.0, 1_000_000_000.0, false)
    } else {
        let budget = time_manager.clock_budget(time_ms, inc, movestogo, engine.st.mc);
        (budget.soft_seconds, budget.hard_seconds, true)
    };

    SearchLimits {
        soft_seconds,
        hard_seconds,
        depth,
        node_limit,
        clock_managed,
        ponder,
    }
}

fn parse_bench_params(parts: &[&str]) -> (i32, Option<usize>) {
    let mut depth = BENCH_DEFAULT_DEPTH;
    let mut count: Option<usize> = None;
    let mut positional = 0usize;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "depth" if i + 1 < parts.len() => {
                match parts[i + 1].parse::<i32>() {
                    Ok(value) => depth = value,
                    Err(_) => {
                        eprintln!("info string Ignoring invalid bench depth: {}", parts[i + 1])
                    }
                }
                positional = positional.max(1);
                i += 1;
            }
            "positions" if i + 1 < parts.len() => {
                match parts[i + 1].parse::<usize>() {
                    Ok(value) => count = Some(value),
                    Err(_) => eprintln!(
                        "info string Ignoring invalid bench position count: {}",
                        parts[i + 1]
                    ),
                }
                positional = positional.max(2);
                i += 1;
            }
            token if positional == 0 => {
                if let Ok(value) = token.parse::<i32>() {
                    depth = value;
                    positional += 1;
                }
            }
            token if positional == 1 => {
                if let Ok(value) = token.parse::<usize>() {
                    count = Some(value);
                    positional += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    (
        depth.clamp(1, BENCH_MAX_DEPTH),
        count.map(|count| count.min(BENCH_POSITIONS.len())),
    )
}

fn bench_signature_update(signature: u64, nodes: u64) -> u64 {
    let mut hash = signature;
    for byte in nodes.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

fn run_bench(engine: &Engine, depth: i32, count: Option<usize>) {
    let selected_count = count
        .filter(|count| *count > 0)
        .map(|count| count.min(BENCH_POSITIONS.len()))
        .unwrap_or(BENCH_POSITIONS.len());

    engine.search_pool.clear_learning();
    println!(
        "info string bench: {} positions, depth {}, threads {}, hash {} MB",
        selected_count, depth, engine.num_threads, engine.searcher.tt_mb
    );

    let mut total_nodes = 0u64;
    let mut signature = FNV1A_BASIS;
    let total_start = Instant::now();

    for (index, (label, command)) in BENCH_POSITIONS.iter().take(selected_count).enumerate() {
        let mut bench_engine = Engine::new();
        bench_engine.num_threads = engine.num_threads;
        bench_engine.searcher.tt_mb = engine.searcher.tt_mb;
        bench_engine.st.chess960 = engine.st.chess960;

        let mut parts = vec!["position"];
        parts.extend(command.split_whitespace());
        parse_position(&mut bench_engine, &parts);

        let start = Instant::now();
        let (_, _, nodes, _) = bench_engine
            .find_best_move_with_time_limits_prepared_with_node_limit(
                BENCH_TIME_LIMIT_SECONDS,
                BENCH_TIME_LIMIT_SECONDS,
                depth,
                None,
            );
        let elapsed = start.elapsed().as_secs_f64();
        let nps = if elapsed > 0.0 {
            (nodes as f64 / elapsed) as u64
        } else {
            0
        };
        total_nodes += nodes;
        signature = bench_signature_update(signature, nodes);
        println!(
            "info string bench {}/{} {} depth {} nodes {} time {}ms nps {}",
            index + 1,
            selected_count,
            label,
            depth,
            nodes,
            (elapsed * 1000.0) as u64,
            nps
        );
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    let total_nps = if total_elapsed > 0.0 {
        (total_nodes as f64 / total_elapsed) as u64
    } else {
        0
    };
    println!(
        "info string bench total: {} positions, depth {}, nodes {}, time {}ms, nps {}, signature {:016x}",
        selected_count,
        depth,
        total_nodes,
        (total_elapsed * 1000.0) as u64,
        total_nps,
        signature
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ember_chess::board::board_to_fen;

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
    fn tune_value_parses_csv_and_space_forms() {
        let csv = parse_tune_value("PROBCUT_MARGIN_CP=400,PROBCUT_MIN_DEPTH=12").unwrap();
        assert_eq!(csv.len(), 2);
        assert!(csv.contains(&(TuneParam::ProbCutMarginCp, 400)));
        assert!(csv.contains(&(TuneParam::ProbCutMinDepth, 12)));

        let spaced = parse_tune_value("ROOT_REPETITION_TIE_MIN_SCORE 320").unwrap();
        assert_eq!(spaced, vec![(TuneParam::RootRepetitionTieMinScore, 320)]);

        let quoted = parse_tune_value("\"PROBCUT_MIN_DEPTH=10\"").unwrap();
        assert_eq!(quoted, vec![(TuneParam::ProbCutMinDepth, 10)]);

        assert!(parse_tune_value("UNKNOWN=1").is_err());
        assert!(parse_tune_value("PROBCUT_MIN_DEPTH=abc").is_err());
        assert!(parse_tune_value("PROBCUT_MIN_DEPTH").is_err());
        assert!(parse_tune_value("").unwrap().is_empty());
    }

    #[test]
    fn uci_move_parser_rejects_malformed_input() {
        assert_eq!(parse_uci_move("e2e4"), Some((6, 4, 4, 4, 0)));
        assert_eq!(parse_uci_move("e7e8q"), Some((1, 4, 0, 4, b'Q')));

        for mv in ["", "e2", "0000", "zzzz", "i2e4", "e9e4", "e2e4x", "e2e4qq"] {
            assert!(
                parse_uci_move(mv).is_none(),
                "malformed UCI move {mv:?} must be rejected without panicking"
            );
        }
    }

    #[test]
    fn random_book_move_option_is_parsed_and_preserved_on_reset() {
        let mut engine = Engine::new();

        parse_setoption(&mut engine, "randombookmove", "true");
        assert!(engine.random_book_move);

        reset_engine(&mut engine);
        assert!(
            engine.random_book_move,
            "ucinewgame must preserve the configured RandomBookMove value"
        );

        parse_setoption(&mut engine, "randombookmove", "false");
        assert!(!engine.random_book_move);
    }

    #[test]
    fn multipv_option_is_parsed_clamped_and_preserved_on_reset() {
        let mut engine = Engine::new();
        assert_eq!(engine.multi_pv, 1, "MultiPV must default to 1");

        parse_setoption(&mut engine, "multipv", "4");
        assert_eq!(engine.multi_pv, 4);

        parse_setoption(&mut engine, "multipv", "0");
        assert_eq!(engine.multi_pv, 4, "out-of-range MultiPV must be rejected");

        parse_setoption(&mut engine, "multipv", "1000000");
        assert_eq!(
            engine.multi_pv, 4,
            "MultiPV above the advertised maximum must be rejected"
        );

        reset_engine(&mut engine);
        assert_eq!(
            engine.multi_pv, 4,
            "ucinewgame must preserve the configured MultiPV value"
        );

        parse_setoption(&mut engine, "multipv", "1");
        assert_eq!(engine.multi_pv, 1);
    }

    #[test]
    fn reset_preserves_chess960_hash_alignment() {
        let mut engine = Engine::new();
        engine.st.chess960 = true;

        reset_engine(&mut engine);

        let recomputed = ember_chess::zobrist::compute_hash(&engine.st);
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

    #[test]
    fn bench_params_accept_defaults_positional_and_named_forms() {
        assert_eq!(
            parse_bench_params(&["bench"]),
            (BENCH_DEFAULT_DEPTH, None),
            "bench without arguments must use the default depth and the full corpus"
        );
        assert_eq!(parse_bench_params(&["bench", "12"]), (12, None));
        assert_eq!(parse_bench_params(&["bench", "8", "3"]), (8, Some(3)));
        assert_eq!(
            parse_bench_params(&["bench", "depth", "5", "positions", "2"]),
            (5, Some(2))
        );
        assert_eq!(parse_bench_params(&["bench", "depth", "7"]), (7, None));
        assert_eq!(
            parse_bench_params(&["bench", "positions", "4"]),
            (BENCH_DEFAULT_DEPTH, Some(4))
        );
        assert_eq!(
            parse_bench_params(&["bench", "depth", "5", "3"]),
            (5, Some(3))
        );
    }

    #[test]
    fn bench_params_clamp_out_of_range_values() {
        assert_eq!(
            parse_bench_params(&["bench", "0"]),
            (1, None),
            "bench depth must stay within 1..=64"
        );
        assert_eq!(parse_bench_params(&["bench", "-5"]), (1, None));
        assert_eq!(
            parse_bench_params(&["bench", "999"]),
            (BENCH_MAX_DEPTH, None)
        );
        assert_eq!(
            parse_bench_params(&["bench", "10", "999"]),
            (10, Some(BENCH_POSITIONS.len())),
            "position count must be capped at the embedded corpus size"
        );
        assert_eq!(
            parse_bench_params(&["bench", "10", "0"]),
            (10, Some(0)),
            "count 0 keeps the full corpus (filtered at run time)"
        );
        assert_eq!(
            parse_bench_params(&["bench", "abc", "def"]),
            (BENCH_DEFAULT_DEPTH, None),
            "non-numeric tokens must fall back to the defaults"
        );
    }

    #[test]
    fn bench_signature_folds_node_counts_deterministically() {
        let basis = FNV1A_BASIS;
        let first = bench_signature_update(basis, 100);
        assert_eq!(first, bench_signature_update(basis, 100));
        assert_ne!(first, bench_signature_update(basis, 101));
        assert_ne!(
            bench_signature_update(first, 1),
            bench_signature_update(first, 2)
        );

        let forward = bench_signature_update(bench_signature_update(basis, 10), 20);
        let backward = bench_signature_update(bench_signature_update(basis, 20), 10);
        assert_ne!(forward, backward);
    }
}
