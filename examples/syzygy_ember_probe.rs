use std::{
    io::{self, BufRead as _, BufWriter, Write as _},
    path::{Path, PathBuf},
    str::FromStr as _,
};

use cozy_chess::Board;
use ember_chess::{board::move_to_uci, movegen::generate_moves, syzygy::SyzygyTables, Engine};
use pyrrhic_rs::WdlProbeResult;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u8,
    id: String,
    fen: String,
    operation: Operation,
    #[serde(default)]
    required_files: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Wdl,
    Dtz,
    Root,
}

#[derive(Serialize)]
struct Response {
    version: u8,
    id: String,
    status: &'static str,
    error: Option<String>,
    wdl: Option<i32>,
    dtz: Option<i32>,
    wdl50: Option<String>,
    search_score: Option<i32>,
    selected_move: Option<String>,
}

fn wdl_number(wdl: WdlProbeResult) -> i32 {
    match wdl {
        WdlProbeResult::Loss => -2,
        WdlProbeResult::BlessedLoss => -1,
        WdlProbeResult::Draw => 0,
        WdlProbeResult::CursedWin => 1,
        WdlProbeResult::Win => 2,
    }
}

fn respond(req: Request, path: &Path, engine: &mut Engine) -> Response {
    let mut response = Response {
        version: 1,
        id: req.id,
        status: "ok",
        error: None,
        wdl: None,
        dtz: None,
        wdl50: None,
        search_score: None,
        selected_move: None,
    };
    if req.version != 1 {
        response.status = "invalid_request";
        response.error = Some("unsupported protocol version".into());
        return response;
    }
    for filename in &req.required_files {
        if filename.contains('/') || filename.contains('\\') || filename.starts_with('.') {
            response.status = "invalid_request";
            response.error = Some("invalid required filename".into());
            return response;
        }
        if !path.join(filename).is_file() {
            response.status = "missing_input_table";
            response.error = Some(filename.clone());
            return response;
        }
    }
    if let Err(err) = Board::from_str(&req.fen) {
        response.status = "invalid_position";
        response.error = Some(err.to_string());
        return response;
    }
    engine.set_fen(&req.fen);
    let tables: &SyzygyTables = &engine.searcher.syzygy;
    response.wdl = tables.probe_wdl(&engine.st).map(wdl_number);
    if response.wdl.is_none() {
        response.status = "probe_error";
        response.error = Some("WDL probe failed".into());
        return response;
    }
    if matches!(req.operation, Operation::Wdl) {
        return response;
    }
    response.dtz = tables.probe_dtz(&engine.st);
    if response.dtz.is_none() {
        response.status = "probe_error";
        response.error = Some("DTZ probe failed".into());
        return response;
    }
    response.wdl50 = tables
        .probe_wdl_50(&engine.st)
        .map(|value| format!("{value:?}"));
    response.search_score = tables.probe_search_score(&engine.st, 1);
    if matches!(req.operation, Operation::Root) {
        let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        response.selected_move = tables
            .probe_root_move(&engine.st, &legal)
            .map(|mv| move_to_uci(&engine.st, mv));
    }
    response
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--tables")) {
        return Err("usage: syzygy_ember_probe --tables DIRECTORY".into());
    }
    let path: PathBuf = args.next().ok_or("missing table directory")?.into();
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let mut engine = Engine::new();
    engine
        .searcher
        .syzygy
        .load(path.to_str().ok_or("non-UTF8 table directory")?)?;
    eprintln!(
        "Ember discovered {}-piece tables",
        engine.searcher.syzygy.max_pieces()
    );

    let stdin = io::stdin().lock();
    let mut stdout = BufWriter::new(io::stdout().lock());
    for line in stdin.lines() {
        let line = line?;
        let req: Request = serde_json::from_str(&line)?;
        let response = respond(req, &path, &mut engine);
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Ember probe: {err}");
        std::process::exit(1);
    }
}
