use chess_rs_lib::backend::{
    available_nnue_backends, available_search_backends, default_search_backend, NnueBackendKind,
};
use chess_rs_lib::board::{move_ec, move_er, move_promotion, move_sc, move_sr, BoardState};
use chess_rs_lib::evaluate::{self, EMBEDDED_NNUE};
use chess_rs_lib::movegen::{apply_move, generate_moves};
use chess_rs_lib::nnue::{NNUEAccumulator, NNUENet};
use chess_rs_lib::search::set_search_backend_override;
use chess_rs_lib::types::{BLACK, WHITE};
use chess_rs_lib::Engine;
use std::hint::black_box;
use std::time::Instant;

const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QK2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 8",
    "2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14",
    "8/2p2pk1/1p4p1/p2Pp3/P1P1P1P1/1P3K2/8/8 w - - 0 40",
    "8/5pk1/6p1/3N4/3P4/5P2/6PK/8 w - - 0 45",
    "8/P4k2/8/8/8/8/8/6K1 w - - 0 1",
    "8/8/8/R2pP1k1/8/8/6Q1/4K3 w - d6 0 1",
];

#[derive(Debug)]
struct Args {
    refresh_loops: usize,
    update_loops: usize,
    search_depth: i32,
    search_repeats: usize,
    hash_mb: usize,
    json_path: Option<String>,
}

#[derive(Debug)]
struct NnueResult {
    backend: &'static str,
    refresh_calls: usize,
    refresh_ns: u128,
    refresh_checksum: i64,
    update_calls: usize,
    update_ns: u128,
    update_checksum: i64,
}

#[derive(Debug)]
struct SearchRun {
    fen_index: usize,
    repeat: usize,
    best_move: String,
    score: i32,
    nodes: u64,
    elapsed_ns: u128,
}

#[derive(Debug)]
struct SearchResult {
    backend: &'static str,
    depth: i32,
    runs: Vec<SearchRun>,
}

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let prefix = format!("{name}=");
    std::env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix(&prefix)?.parse().ok())
        .unwrap_or(default)
}

fn parse_args() -> Args {
    let json_path = std::env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix("--json=").map(str::to_owned));
    Args {
        refresh_loops: parse_arg("--refresh-loops", 2000),
        update_loops: parse_arg("--update-loops", 200),
        search_depth: parse_arg("--search-depth", 11),
        search_repeats: parse_arg("--search-repeats", 2),
        hash_mb: parse_arg("--hash-mb", 128),
        json_path,
    }
}

fn states() -> Vec<BoardState> {
    FENS.iter()
        .map(|fen| {
            let mut engine = Engine::new();
            engine.book = None;
            engine.set_fen(fen);
            engine.st
        })
        .collect()
}

fn piece_count(st: &BoardState) -> u32 {
    (0..12).map(|idx| st.bb[idx].count_ones()).sum()
}

fn evaluate_with_backend(
    backend: NnueBackendKind,
    net: &NNUENet,
    acc: &NNUEAccumulator,
    st: &BoardState,
) -> i32 {
    let stm = if st.w { WHITE } else { BLACK };
    let score = net.forward_with_kind(backend, acc, stm, piece_count(st));
    if stm == WHITE {
        score
    } else {
        -score
    }
}

fn bench_refresh(
    backend: NnueBackendKind,
    net: &NNUENet,
    states: &[BoardState],
    loops: usize,
) -> (i64, usize) {
    let mut checksum = 0i64;
    let mut acc = NNUEAccumulator::new(net.hidden_size);
    for _ in 0..loops {
        for st in states {
            acc.refresh_with_kind(backend, black_box(net), black_box(st));
            checksum += evaluate_with_backend(backend, net, black_box(&acc), st) as i64;
        }
    }
    (checksum, loops * states.len())
}

fn bench_incremental(
    backend: NnueBackendKind,
    net: &NNUENet,
    states: &[BoardState],
    loops: usize,
) -> (i64, usize) {
    let mut checksum = 0i64;
    let mut updates = 0usize;

    for _ in 0..loops {
        for st in states {
            let moves = generate_moves(st, st.w, &st.cr, st.ep);
            let mut base_acc = NNUEAccumulator::new(net.hidden_size);
            base_acc.refresh_with_kind(backend, net, st);

            for &mv in &moves {
                let mut next_acc = base_acc.clone();
                if !next_acc.update_move_with_kind(
                    backend,
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
                checksum += evaluate_with_backend(backend, net, black_box(&next_acc), &next) as i64;
                updates += 1;
            }
        }
    }

    (checksum, updates)
}

fn bench_nnue(args: &Args, net: &NNUENet, states: &[BoardState]) -> Vec<NnueResult> {
    available_nnue_backends()
        .into_iter()
        .map(|backend| {
            let start = Instant::now();
            let (refresh_checksum, refresh_calls) =
                bench_refresh(backend, net, states, args.refresh_loops);
            let refresh_ns = start.elapsed().as_nanos();

            let start = Instant::now();
            let (update_checksum, update_calls) =
                bench_incremental(backend, net, states, args.update_loops);
            let update_ns = start.elapsed().as_nanos();

            NnueResult {
                backend: backend.name(),
                refresh_calls,
                refresh_ns,
                refresh_checksum,
                update_calls,
                update_ns,
                update_checksum,
            }
        })
        .collect()
}

fn bench_search(args: &Args) -> Vec<SearchResult> {
    available_search_backends()
        .into_iter()
        .map(|backend| {
            set_search_backend_override(Some(backend));
            let mut runs = Vec::new();

            for repeat in 0..args.search_repeats {
                for (fen_index, fen) in FENS.iter().enumerate() {
                    let mut engine = Engine::new();
                    engine.book = None;
                    engine.num_threads = 1;
                    engine.searcher.resize_tt(args.hash_mb);
                    engine.set_fen(fen);

                    let (best_move, score, nodes, elapsed) =
                        engine.find_best_move(60.0 * 60.0, args.search_depth);
                    runs.push(SearchRun {
                        fen_index,
                        repeat,
                        best_move,
                        score,
                        nodes,
                        elapsed_ns: (elapsed * 1_000_000_000.0) as u128,
                    });
                }
            }

            set_search_backend_override(None);
            SearchResult {
                backend: backend.name(),
                depth: args.search_depth,
                runs,
            }
        })
        .collect()
}

fn cpu_model() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for key in ["model name", "Hardware", "Processor"] {
            if let Some(value) = first_cpuinfo_value(&cpuinfo, key) {
                return value;
            }
        }
    }

    if let Some(value) = first_command_field("lscpu", &[], "Model name") {
        return value;
    }

    if let Ok(output) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if output.status.success() {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !value.is_empty() {
                return value;
            }
        }
    }

    "unknown".to_owned()
}

fn cpu_features() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for key in ["flags", "Features"] {
            if let Some(value) = first_cpuinfo_value(&cpuinfo, key) {
                return value;
            }
        }
    }
    first_command_field("lscpu", &[], "Flags")
        .or_else(|| first_command_field("lscpu", &[], "Features"))
        .unwrap_or_default()
}

fn first_command_field(command: &str, args: &[&str], key: &str) -> Option<String> {
    let output = std::process::Command::new(command)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    first_colon_value(&stdout, key)
}

fn first_colon_value(text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    text.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

fn first_cpuinfo_value(cpuinfo: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    cpuinfo.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .or_else(|| line.strip_prefix(&format!("{key}\t:")))
            .map(|value| value.trim().to_owned())
    })
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn ns_per_call(ns: u128, calls: usize) -> f64 {
    ns as f64 / calls.max(1) as f64
}

fn total_search_nps(result: &SearchResult) -> f64 {
    let nodes: u64 = result.runs.iter().map(|run| run.nodes).sum();
    let ns: u128 = result.runs.iter().map(|run| run.elapsed_ns).sum();
    nodes as f64 / (ns as f64 / 1_000_000_000.0).max(1e-9)
}

fn write_json(
    args: &Args,
    nnue: &[NnueResult],
    search: &[SearchResult],
) -> Result<String, std::io::Error> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!(
        "  \"arch\": \"{}\",\n",
        json_escape(std::env::consts::ARCH)
    ));
    json.push_str(&format!(
        "  \"os\": \"{}\",\n",
        json_escape(std::env::consts::OS)
    ));
    json.push_str(&format!(
        "  \"cpu_model\": \"{}\",\n",
        json_escape(&cpu_model())
    ));
    json.push_str(&format!(
        "  \"cpu_features\": \"{}\",\n",
        json_escape(&cpu_features())
    ));
    json.push_str(&format!(
        "  \"default_search_backend\": \"{}\",\n",
        json_escape(default_search_backend().name())
    ));
    json.push_str(&format!(
        "  \"refresh_loops\": {},\n  \"update_loops\": {},\n  \"search_depth\": {},\n  \"search_repeats\": {},\n  \"hash_mb\": {},\n",
        args.refresh_loops, args.update_loops, args.search_depth, args.search_repeats, args.hash_mb
    ));

    json.push_str("  \"nnue\": [\n");
    for (idx, result) in nnue.iter().enumerate() {
        if idx > 0 {
            json.push_str(",\n");
        }
        json.push_str(&format!(
            "    {{\"backend\": \"{}\", \"refresh_calls\": {}, \"refresh_ns\": {}, \"refresh_ns_per_call\": {:.4}, \"refresh_checksum\": {}, \"update_calls\": {}, \"update_ns\": {}, \"update_ns_per_call\": {:.4}, \"update_checksum\": {}}}",
            json_escape(result.backend),
            result.refresh_calls,
            result.refresh_ns,
            ns_per_call(result.refresh_ns, result.refresh_calls),
            result.refresh_checksum,
            result.update_calls,
            result.update_ns,
            ns_per_call(result.update_ns, result.update_calls),
            result.update_checksum
        ));
    }
    json.push_str("\n  ],\n");

    json.push_str("  \"search\": [\n");
    for (idx, result) in search.iter().enumerate() {
        if idx > 0 {
            json.push_str(",\n");
        }
        json.push_str(&format!(
            "    {{\"backend\": \"{}\", \"depth\": {}, \"nps\": {:.2}, \"runs\": [",
            json_escape(result.backend),
            result.depth,
            total_search_nps(result)
        ));
        for (run_idx, run) in result.runs.iter().enumerate() {
            if run_idx > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!(
                "{{\"fen_index\": {}, \"repeat\": {}, \"best_move\": \"{}\", \"score\": {}, \"nodes\": {}, \"elapsed_ns\": {}}}",
                run.fen_index,
                run.repeat,
                json_escape(&run.best_move),
                run.score,
                run.nodes,
                run.elapsed_ns
            ));
        }
        json.push_str("]}");
    }
    json.push_str("\n  ]\n");
    json.push_str("}\n");

    if let Some(path) = &args.json_path {
        std::fs::write(path, &json)?;
    }
    Ok(json)
}

fn print_summary(nnue: &[NnueResult], search: &[SearchResult]) {
    eprintln!("NNUE backends:");
    for result in nnue {
        eprintln!(
            "  {} refresh {:.2} ns/call checksum {} incremental {:.2} ns/call checksum {}",
            result.backend,
            ns_per_call(result.refresh_ns, result.refresh_calls),
            result.refresh_checksum,
            ns_per_call(result.update_ns, result.update_calls),
            result.update_checksum
        );
    }

    eprintln!("Search backends:");
    for result in search {
        let nodes: u64 = result.runs.iter().map(|run| run.nodes).sum();
        eprintln!(
            "  {} depth {} nps {:.0} nodes {} runs {}",
            result.backend,
            result.depth,
            total_search_nps(result),
            nodes,
            result.runs.len()
        );
    }
}

fn main() {
    let args = parse_args();
    evaluate::init_embedded_nnue().expect("embedded NNUE should load");
    let net = NNUENet::load_compact_from_bytes(EMBEDDED_NNUE, "<backend-bench>")
        .expect("embedded NNUE should load directly");
    let states = states();

    eprintln!(
        "arch={} os={}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    eprintln!("cpu={}", cpu_model());
    eprintln!(
        "nnue_backends={}",
        available_nnue_backends()
            .iter()
            .map(|backend| backend.name())
            .collect::<Vec<_>>()
            .join(",")
    );
    eprintln!(
        "search_backends={}",
        available_search_backends()
            .iter()
            .map(|backend| backend.name())
            .collect::<Vec<_>>()
            .join(",")
    );
    eprintln!("default_search_backend={}", default_search_backend().name());

    let nnue = bench_nnue(&args, &net, &states);
    let search = bench_search(&args);
    print_summary(&nnue, &search);

    let json = write_json(&args, &nnue, &search).expect("write benchmark json");
    if args.json_path.is_none() {
        println!("{json}");
    }
}
