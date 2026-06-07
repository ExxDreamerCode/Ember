use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct DepthInfo {
    pub depth: i32,
    pub score_cp: i32,
    pub nodes: u64,
    pub elapsed_ms: u128,
    pub pv: String,
}

pub struct DecisionTrace<'a> {
    pub fen: &'a str,
    pub side: &'a str,
    pub legal_moves: &'a [String],
    pub chosen_move: &'a str,
    pub source: &'a str,
    pub depth_reached: i32,
    pub score_cp: i32,
    pub nodes: u64,
    pub elapsed_ms: u128,
    pub depth_infos: &'a [DepthInfo],
}

#[derive(Default)]
pub struct TraceLogger {
    file: Option<File>,
    seq: u64,
}

impl TraceLogger {
    pub fn from_env() -> Self {
        if let Ok(path) = std::env::var("EMBER_TRACE_FILE") {
            return Self::from_path(path);
        }
        if let Ok(dir) = std::env::var("EMBER_TRACE_DIR") {
            let mut path = PathBuf::from(dir);
            let _ = create_dir_all(&path);
            path.push(format!("ember-trace-{}.jsonl", std::process::id()));
            return Self::from_path(path);
        }
        Self::default()
    }

    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Some(parent) = path.parent() {
            let _ = create_dir_all(parent);
        }
        let file = OpenOptions::new().create(true).append(true).open(path).ok();
        Self { file, seq: 0 }
    }

    pub fn set_path(&mut self, path: impl Into<PathBuf>) {
        *self = Self::from_path(path);
    }

    pub fn emit_decision(&mut self, tr: DecisionTrace<'_>) {
        let Some(file) = self.file.as_mut() else { return; };
        self.seq += 1;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let mut line = String::new();
        line.push('{');
        field_u64(&mut line, "schema", 1);
        field_str(&mut line, "event", "ember_decision");
        field_u64(&mut line, "pid", std::process::id() as u64);
        field_u64(&mut line, "seq", self.seq);
        field_u64(&mut line, "time_unix_ms", now_ms as u64);
        field_str(&mut line, "fen", tr.fen);
        field_str(&mut line, "side", tr.side);
        field_array_str(&mut line, "legal_moves", tr.legal_moves);
        field_str(&mut line, "chosen_move", tr.chosen_move);
        field_str(&mut line, "source", tr.source);
        field_i32(&mut line, "depth_reached", tr.depth_reached);
        field_i32(&mut line, "score_cp", tr.score_cp);
        field_u64(&mut line, "nodes", tr.nodes);
        field_u64(&mut line, "elapsed_ms", tr.elapsed_ms as u64);
        field_depth_infos(&mut line, "depth_infos", tr.depth_infos);
        trim_trailing_comma(&mut line);
        line.push_str("}\n");
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
}

fn field_str(out: &mut String, key: &str, value: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    push_json_escaped(out, value);
    out.push_str("\",");
}

fn field_i32(out: &mut String, key: &str, value: i32) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    out.push(',');
}

fn field_u64(out: &mut String, key: &str, value: u64) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    out.push(',');
}

fn field_array_str(out: &mut String, key: &str, values: &[String]) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":[");
    for value in values {
        out.push('"');
        push_json_escaped(out, value);
        out.push_str("\",");
    }
    trim_trailing_comma(out);
    out.push_str("],");
}

fn field_depth_infos(out: &mut String, key: &str, values: &[DepthInfo]) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":[");
    for value in values {
        out.push('{');
        field_i32(out, "depth", value.depth);
        field_i32(out, "score_cp", value.score_cp);
        field_u64(out, "nodes", value.nodes);
        field_u64(out, "elapsed_ms", value.elapsed_ms as u64);
        field_str(out, "pv", &value.pv);
        trim_trailing_comma(out);
        out.push_str("},");
    }
    trim_trailing_comma(out);
    out.push_str("],");
}

fn trim_trailing_comma(out: &mut String) {
    if out.ends_with(',') {
        out.pop();
    }
}

fn push_json_escaped(out: &mut String, value: &str) {
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
}
