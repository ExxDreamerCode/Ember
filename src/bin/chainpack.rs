use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use ember_chess::board::{
    board_to_fen, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    move_to_uci, piece_on, piece_type, sq, sq_c, sq_r, BoardState, Move, EMPTY_SQ,
};
use ember_chess::movegen::{apply_move, generate_moves};
use ember_chess::search::SEARCH_THREAD_STACK_SIZE;
use ember_chess::Engine;

const MAGIC: &[u8; 4] = b"BINP";
const CHUNK_TARGET_BYTES: usize = 32 * 1024 * 1024;
const ENTRY_SIZE: usize = 32;
const DEFAULT_BATCH_SIZE: usize = 1_000_000;
const DEFAULT_NODES: u64 = 5000;
const DEFAULT_DEPTH: i32 = 64;
const CHAINPACK_STATE_SAVE_EVERY: u64 = 10_000;
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn signed_to_unsigned(value: i16) -> u16 {
    let bits = value as u16;
    if bits & 0x8000 != 0 {
        (bits ^ 0x7FFF).rotate_left(1)
    } else {
        bits.rotate_left(1)
    }
}

fn std_bitboards(st: &BoardState) -> [u64; 12] {
    let mut out = [0u64; 12];
    for (i, bb) in st.bb.iter().enumerate() {
        out[i] = bb.swap_bytes();
    }
    out
}

fn std_square(ember_sq: usize) -> usize {
    (7 - sq_r(ember_sq)) * 8 + sq_c(ember_sq)
}

fn ember_square(std_sq: usize) -> usize {
    sq(7 - std_sq / 8, std_sq % 8)
}

#[derive(Clone, Copy)]
struct ChainPosition {
    occupied: u64,
    packed_state: [u8; 16],
    _side_to_move: bool,
    rule50: u16,
}

impl ChainPosition {
    fn from_state(st: &BoardState) -> Self {
        let bbs = std_bitboards(st);
        let mut occupied: u64 = 0;
        for bb_val in bbs.iter() {
            occupied |= *bb_val;
        }

        let ep_pawn_sq = st.ep.and_then(|ep| {
            let ep = std_square(ep);
            if st.w {
                (ep / 8 == 5).then_some(ep - 8)
            } else {
                (ep / 8 == 2).then_some(ep + 8)
            }
        });

        let castling_rook_sqs: Vec<Option<usize>> = st
            .castling_rooks
            .iter()
            .map(|rook_sq| rook_sq.map(std_square))
            .collect();

        let mut packed_state = [0u8; 16];
        let mut nibble_idx = 0usize;

        for sq_int in 0..64usize {
            let pi = piece_on(&bbs, sq_int);
            if pi == EMPTY_SQ {
                continue;
            }

            let mut nibble = match pi {
                0 => 0,
                6 => 1,
                1 => 2,
                7 => 3,
                2 => 4,
                8 => 5,
                3 => 6,
                9 => 7,
                4 => 8,
                10 => 9,
                5 => 10,
                11 => 11,
                _ => 0,
            };

            if pi == 11 && !st.w {
                nibble = 15;
            } else if piece_type(pi) == 0 && ep_pawn_sq == Some(sq_int) {
                nibble = 12;
            } else if pi == 3 || pi == 9 {
                for (i, rook_sq) in castling_rook_sqs.iter().enumerate() {
                    if st.cr[i] && *rook_sq == Some(sq_int) {
                        nibble = if i < 2 { 13 } else { 14 };
                        break;
                    }
                }
            }

            if nibble_idx.is_multiple_of(2) {
                packed_state[nibble_idx / 2] = nibble;
            } else {
                packed_state[nibble_idx / 2] |= nibble << 4;
            }
            nibble_idx += 1;
        }

        ChainPosition {
            occupied,
            packed_state,
            _side_to_move: st.w,
            rule50: st.halfmove_clock as u16,
        }
    }

    fn to_bytes(self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[0..8].copy_from_slice(&self.occupied.to_be_bytes());
        out[8..24].copy_from_slice(&self.packed_state);
        out
    }
}

#[derive(Clone, Copy)]
struct ChainMove {
    from: usize,
    to: usize,
    promotion: u8,
    is_null: bool,
}

impl ChainMove {
    fn from_uci(uci: &str) -> Option<Self> {
        if uci.len() < 4 || uci == "0000" {
            return Some(Self {
                from: 0,
                to: 0,
                promotion: 0,
                is_null: true,
            });
        }
        let b = uci.as_bytes();
        let from = (b[1] - b'1') as usize * 8 + (b[0] - b'a') as usize;
        let to = (b[3] - b'1') as usize * 8 + (b[2] - b'a') as usize;
        let promotion = if uci.len() >= 5 {
            match b[4] {
                b'q' | b'Q' => 4,
                b'r' | b'R' => 3,
                b'b' | b'B' => 2,
                b'n' | b'N' => 1,
                _ => 0,
            }
        } else {
            0
        };
        Some(Self {
            from,
            to,
            promotion,
            is_null: false,
        })
    }

    fn compress(&self) -> u16 {
        if self.is_null {
            return 0;
        }
        let move_type: u16 = if self.promotion != 0 { 1 } else { 0 };
        let mut packed = move_type << 14;
        packed |= (self.from as u16) << 8;
        packed |= (self.to as u16) << 2;
        if self.promotion != 0 {
            packed |= (self.promotion as u16 - 1) & 0b11;
        }
        packed
    }
}

#[derive(Clone)]
struct ChainLink {
    score: i16,
    bestmove: String,
    ply: u16,
    result: i16,
}

#[derive(Clone)]
struct ChainEntryBytes {
    stem: [u8; ENTRY_SIZE],
    movetext: Vec<u8>,
}

fn pack_stem(pos: &ChainPosition, mv: &ChainMove, score: i16, ply: u16, result: i16) -> [u8; 32] {
    let mut out = [0u8; ENTRY_SIZE];
    let pos_bytes = pos.to_bytes();
    out[0..24].copy_from_slice(&pos_bytes);
    out[24..26].copy_from_slice(&mv.compress().to_be_bytes());
    out[26..28].copy_from_slice(&signed_to_unsigned(score).to_be_bytes());
    let pr = (signed_to_unsigned(result) & 0b11) << 14 | (ply & 0x3FFF);
    out[28..30].copy_from_slice(&pr.to_be_bytes());
    out[30..32].copy_from_slice(&pos.rule50.to_be_bytes());
    out
}

fn build_chain_entries(start: &BoardState, links: &[ChainLink]) -> Option<Vec<ChainEntryBytes>> {
    let mut st = *start;
    let mut entries = Vec::new();

    for link in links.iter() {
        let mv = ChainMove::from_uci(&link.bestmove)?;
        let pos = ChainPosition::from_state(&st);

        let stem = pack_stem(&pos, &mv, link.score, link.ply, link.result);
        entries.push(ChainEntryBytes {
            stem,
            movetext: Vec::new(),
        });

        if !mv.is_null {
            let promotion_char = match mv.promotion {
                1 => b'N',
                2 => b'B',
                3 => b'R',
                4 => b'Q',
                _ => 0,
            };
            let from = ember_square(mv.from);
            let to = ember_square(mv.to);
            apply_move(
                &mut st,
                sq_r(from),
                sq_c(from),
                sq_r(to),
                sq_c(to),
                promotion_char,
            );
        }
    }

    if entries.is_empty() {
        None
    } else {
        Some(entries)
    }
}

#[derive(Clone)]
struct PgnGame {
    result: i8,
    fen: Option<String>,
    moves: Vec<String>,
}

fn parse_pgn(text: &str) -> Option<PgnGame> {
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut movetext_lines: Vec<String> = Vec::new();
    let mut in_headers = true;

    for line in text.lines() {
        let line = line.trim();
        if in_headers {
            if line.is_empty() {
                in_headers = false;
                continue;
            }
            if line.starts_with('[') {
                let inner = line.trim_start_matches('[').trim_end_matches(']');
                if let Some(space_pos) = inner.find(' ') {
                    let key = inner[..space_pos].to_string();
                    let val = inner[space_pos + 1..].trim().trim_matches('"').to_string();
                    headers.push((key, val));
                }
            }
        } else {
            movetext_lines.push(line.to_string());
        }
    }

    let result_tag = headers
        .iter()
        .find(|(k, _)| k == "Result")
        .map(|(_, v)| v.as_str())
        .unwrap_or("*");
    let fen = headers
        .iter()
        .find(|(k, _)| k == "FEN")
        .map(|(_, v)| v.clone());

    let result = match result_tag {
        "1-0" => 1,
        "0-1" => -1,
        "1/2-1/2" | "*" => 0,
        _ => 0,
    };

    let movetext = movetext_lines.join(" ");
    let cleaned = remove_pgn_comments(&movetext);
    let moves = extract_san_moves(&cleaned);

    if moves.is_empty() {
        return None;
    }

    Some(PgnGame { result, fen, moves })
}

fn remove_pgn_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_comment = false;
    for ch in text.chars() {
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

fn extract_san_moves(text: &str) -> Vec<String> {
    let mut moves = Vec::new();
    for token in text.split_whitespace() {
        let t = token.trim_end_matches(['!', '?', '+', '#']);
        let t = t.trim_end_matches("e.p.");
        if t.is_empty() {
            continue;
        }
        if t == "1-0" || t == "0-1" || t == "1/2-1/2" || t == "*" {
            continue;
        }
        if t.chars().all(|c| c.is_ascii_digit() || c == '.') && t.contains('.') {
            continue;
        }
        let bytes = t.as_bytes();
        let digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
        let t = if digits > 0 && digits < bytes.len() && bytes[digits] == b'.' {
            t[digits..].trim_start_matches('.')
        } else {
            t
        };
        if t.is_empty() {
            continue;
        }
        moves.push(t.to_string());
    }
    moves
}

fn parse_san(st: &BoardState, san: &str) -> Option<Move> {
    let mut s = san.trim();
    while s.ends_with(['+', '#', '!', '?']) {
        s = &s[..s.len() - 1];
    }
    if let Some(rest) = s.strip_suffix("e.p.") {
        s = rest.trim_end();
    }
    if s.is_empty() {
        return None;
    }

    let moves = generate_moves(st, st.w, &st.cr, st.ep);

    match s {
        "O-O" | "0-0" | "OO" => {
            let target = if st.w { "e1g1" } else { "e8g8" };
            return moves.into_iter().find(|m| move_to_uci(st, *m) == target);
        }
        "O-O-O" | "0-0-0" | "OOO" => {
            let target = if st.w { "e1c1" } else { "e8c8" };
            return moves.into_iter().find(|m| move_to_uci(st, *m) == target);
        }
        _ => {}
    }

    let b = s.as_bytes();
    let mut end = s.len();
    let mut promo = 0u8;
    if end >= 3 && b[end - 2] == b'=' && b[end - 1].is_ascii_alphabetic() {
        promo = b[end - 1].to_ascii_uppercase();
        end -= 2;
    } else if end >= 3
        && matches!(b[end - 1], b'Q' | b'R' | b'B' | b'N')
        && b[end - 2].is_ascii_digit()
    {
        promo = b[end - 1];
        end -= 1;
    }
    if end < 2 || !(b'a'..=b'h').contains(&b[end - 2]) || !(b'1'..=b'8').contains(&b[end - 1]) {
        return None;
    }
    let to = sq(
        7 - (b[end - 1] - b'1') as usize,
        (b[end - 2] - b'a') as usize,
    );
    let prefix = &s[..end - 2];

    let bytes = prefix.as_bytes();
    let (pt, mut file_hint, mut rank_hint) = match bytes.first() {
        None => (0usize, None, None),
        Some(b'N') => (1, None, None),
        Some(b'B') => (2, None, None),
        Some(b'R') => (3, None, None),
        Some(b'Q') => (4, None, None),
        Some(b'K') => (5, None, None),
        Some(b'P') => (0, None, None),
        Some(c) if (b'a'..=b'h').contains(c) => (0, Some((c - b'a') as usize), None),
        _ => return None,
    };
    if !bytes.is_empty() {
        for &c in &bytes[1..] {
            match c {
                b'x' | b':' | b'-' => {}
                b'a'..=b'h' => {
                    if file_hint.is_some() {
                        return None;
                    }
                    file_hint = Some((c - b'a') as usize);
                }
                b'1'..=b'8' => {
                    if rank_hint.is_some() {
                        return None;
                    }
                    rank_hint = Some((c - b'1') as usize);
                }
                _ => return None,
            }
        }
    }

    let mut found: Option<Move> = None;
    for mv in moves {
        if move_to(mv) != to || move_promotion(mv).to_ascii_uppercase() != promo {
            continue;
        }
        let from = move_from(mv);
        let mover = st.mailbox[from];
        if mover == EMPTY_SQ || piece_type(mover) as usize != pt {
            continue;
        }
        if let Some(f) = file_hint {
            if sq_c(from) != f {
                continue;
            }
        }
        if let Some(r) = rank_hint {
            if sq_r(from) != 7 - r {
                continue;
            }
        }
        if found.is_some() {
            return None;
        }
        found = Some(mv);
    }
    found
}

fn evaluate_position(
    engine: &mut Engine,
    fen: &str,
    nodes: u64,
    depth: i32,
    eval_only: bool,
) -> Option<(String, i32)> {
    engine.set_fen(fen);
    engine
        .searcher
        .stopped
        .store(false, std::sync::atomic::Ordering::Relaxed);

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

struct GameProcessor {
    engine: Engine,
    nodes: u64,
    depth: i32,
    eval_only: bool,
    total_games: u64,
    total_positions: u64,
    skipped_games: u64,
    skipped_positions: u64,
}

impl GameProcessor {
    fn new(nodes: u64, depth: i32, eval_only: bool, nnue_off: bool, nnue_on: bool) -> Self {
        let mut engine = Engine::try_new().expect("failed to create engine");
        engine.set_hash_mb(256);
        engine.num_threads = 1;
        engine.quiet = true;

        if nnue_off {
            let _ = ember_chess::evaluate::reset_nnue();
        }
        if nnue_on {
            if let Err(e) = ember_chess::evaluate::init_embedded_nnue() {
                eprintln!("failed to load embedded NNUE: {e}");
            }
        }
        engine.searcher.refresh_nnue_net();

        Self {
            engine,
            nodes,
            depth,
            eval_only,
            total_games: 0,
            total_positions: 0,
            skipped_games: 0,
            skipped_positions: 0,
        }
    }

    fn process_game(&mut self, game: &PgnGame) -> Vec<ChainEntryBytes> {
        self.total_games += 1;

        let start_fen = game.fen.as_deref().unwrap_or(START_FEN);
        if self.engine.try_set_fen(start_fen).is_err() {
            self.skipped_games += 1;
            return Vec::new();
        }
        let start_state = self.engine.st;

        let mut board = start_state;
        let mut links: Vec<ChainLink> = Vec::new();
        let mut ply: u16 = 0;

        for san in &game.moves {
            let Some(mv) = parse_san(&board, san) else {
                self.skipped_positions += 1;
                break;
            };

            let stm_white = board.w;
            let game_result = match game.result {
                1 => {
                    if stm_white {
                        1
                    } else {
                        -1
                    }
                }
                -1 => {
                    if stm_white {
                        -1
                    } else {
                        1
                    }
                }
                _ => 0,
            };

            let current_fen = board_to_fen(&board);
            let eval = evaluate_position(
                &mut self.engine,
                &current_fen,
                self.nodes,
                self.depth,
                self.eval_only,
            );

            match eval {
                Some((_, score_cp)) => {
                    let score = score_cp.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    links.push(ChainLink {
                        score,
                        bestmove: move_to_uci(&board, mv),
                        ply,
                        result: game_result,
                    });
                    self.total_positions += 1;
                }
                None => {
                    self.skipped_positions += 1;
                }
            }

            self.engine.searcher.clear_learning();

            apply_move(
                &mut board,
                move_sr(mv),
                move_sc(mv),
                move_er(mv),
                move_ec(mv),
                move_promotion(mv),
            );
            ply = ply.wrapping_add(1);
        }

        if links.is_empty() {
            return Vec::new();
        }

        build_chain_entries(&start_state, &links).unwrap_or_default()
    }
}

struct ChunkedWriter {
    out_file: File,
    buf: Vec<u8>,
    buf_len: usize,
    entries_in_buffer: u64,
    flushed_entries: u64,
    total_chunks: u64,
    total_bytes: u64,
}

impl ChunkedWriter {
    fn new(out_file: File) -> Self {
        Self {
            out_file,
            buf: vec![0u8; CHUNK_TARGET_BYTES + 64],
            buf_len: 0,
            entries_in_buffer: 0,
            flushed_entries: 0,
            total_chunks: 0,
            total_bytes: 0,
        }
    }

    fn entries_flushed(&self) -> u64 {
        self.flushed_entries
    }

    fn write_entry(&mut self, stem: &[u8; ENTRY_SIZE], movetext: &[u8]) -> std::io::Result<()> {
        let needed = ENTRY_SIZE + 2 + movetext.len();
        if self.buf_len + needed > CHUNK_TARGET_BYTES && self.buf_len > 0 {
            self.flush_chunk()?;
        }
        self.buf[self.buf_len..self.buf_len + ENTRY_SIZE].copy_from_slice(stem);
        self.buf_len += ENTRY_SIZE;
        let movetext_len = movetext.len() as u16;
        self.buf[self.buf_len..self.buf_len + 2].copy_from_slice(&movetext_len.to_be_bytes());
        self.buf_len += 2;
        self.buf[self.buf_len..self.buf_len + movetext.len()].copy_from_slice(movetext);
        self.buf_len += movetext.len();
        self.entries_in_buffer += 1;
        Ok(())
    }

    fn flush_chunk(&mut self) -> std::io::Result<()> {
        if self.buf_len == 0 {
            return Ok(());
        }
        self.out_file.write_all(MAGIC)?;
        self.out_file
            .write_all(&(self.buf_len as u32).to_le_bytes())?;
        self.out_file.write_all(&self.buf[..self.buf_len])?;
        self.total_bytes += self.buf_len as u64 + 8;
        self.total_chunks += 1;
        self.flushed_entries += self.entries_in_buffer;
        self.entries_in_buffer = 0;
        self.buf_len = 0;
        Ok(())
    }

    fn flush(mut self) -> std::io::Result<(u64, u64, u64)> {
        if self.buf_len > 0 {
            self.out_file.write_all(MAGIC)?;
            self.out_file
                .write_all(&(self.buf_len as u32).to_le_bytes())?;
            self.out_file.write_all(&self.buf[..self.buf_len])?;
            self.total_bytes += self.buf_len as u64 + 8;
            self.total_chunks += 1;
            self.flushed_entries += self.entries_in_buffer;
            self.entries_in_buffer = 0;
        }
        self.out_file.flush()?;
        Ok((self.total_bytes, self.total_chunks, self.flushed_entries))
    }
}

#[derive(Clone, Debug, Default)]
struct ProgressState {
    positions_written: u64,
    games_completed: u64,
    current_file: String,
}

fn state_path_for(output: &Path) -> PathBuf {
    output.with_extension("chainpack.state")
}

fn load_state(path: &Path) -> Option<ProgressState> {
    let content = fs::read_to_string(path).ok()?;
    let mut state = ProgressState::default();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("positions_written:") {
            state.positions_written = rest.trim().parse().ok()?;
        } else if let Some(rest) = line.strip_prefix("games_completed:") {
            state.games_completed = rest.trim().parse().ok()?;
        } else if let Some(rest) = line.strip_prefix("current_file:") {
            state.current_file = rest.trim().to_string();
        }
    }
    Some(state)
}

fn save_state(path: &Path, state: &ProgressState) {
    let content = format!(
        "positions_written: {}\ngames_completed: {}\ncurrent_file: {}\n",
        state.positions_written, state.games_completed, state.current_file
    );
    let _ = fs::write(path, content);
}

struct ChainBatch {
    file_idx: usize,
    entries: Vec<ChainEntryBytes>,
}

fn open_input_reader(path: &str) -> std::io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if ext == "zst" {
        let decoder = zstd::stream::read::Decoder::new(file)?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

struct GameStream<R: BufRead> {
    reader: R,
    current: String,
    brace_depth: i32,
    in_headers: bool,
    movetext_started: bool,
}

impl<R: BufRead> GameStream<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            current: String::new(),
            brace_depth: 0,
            in_headers: true,
            movetext_started: false,
        }
    }
}

impl<R: BufRead> Iterator for GameStream<R> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        self.current.clear();

        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }

            let trimmed = line.trim();

            for ch in trimmed.chars() {
                if ch == '{' {
                    self.brace_depth += 1;
                } else if ch == '}' {
                    self.brace_depth = self.brace_depth.saturating_sub(1);
                }
            }

            if self.in_headers && trimmed.is_empty() && self.brace_depth == 0 {
                self.in_headers = false;
                self.movetext_started = false;
                self.current.push_str(trimmed);
                self.current.push('\n');
                continue;
            }

            if !self.in_headers && trimmed.is_empty() && self.brace_depth == 0 {
                if self.movetext_started && !self.current.is_empty() {
                    let game = std::mem::take(&mut self.current);
                    self.in_headers = true;
                    self.movetext_started = false;
                    return Some(game);
                }
                continue;
            }

            self.current.push_str(trimmed);
            self.current.push('\n');
            self.movetext_started = true;
        }

        if !self.current.is_empty() {
            let game = std::mem::take(&mut self.current);
            return Some(game);
        }

        None
    }
}

fn truncate_to_last_complete_chunk(file: &mut File) -> std::io::Result<(u64, u64)> {
    let mut pos: u64 = 0;
    let len = file.metadata()?.len();
    let mut total_entries: u64 = 0;
    while pos < len {
        file.seek(SeekFrom::Start(pos))?;
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_err() {
            break;
        }
        if &magic != MAGIC {
            break;
        }
        let mut size_buf = [0u8; 4];
        if file.read_exact(&mut size_buf).is_err() {
            break;
        }
        let size = u32::from_le_bytes(size_buf) as u64;
        let chunk_end = pos + 8 + size;
        if chunk_end > len {
            break;
        }
        let mut entry_pos: u64 = pos + 8;
        let chunk_payload_end: u64 = pos + 8 + size;
        let entry_size: u64 = ENTRY_SIZE as u64;
        while entry_pos + entry_size <= chunk_payload_end {
            file.seek(SeekFrom::Start(entry_pos + entry_size))?;
            let mut len_buf = [0u8; 2];
            if file.read_exact(&mut len_buf).is_err() {
                break;
            }
            let movetext_len = u16::from_be_bytes(len_buf) as u64;
            total_entries += 1;
            entry_pos += entry_size + 2 + movetext_len;
        }
        pos = chunk_end;
    }
    file.set_len(pos)?;
    Ok((pos, total_entries))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: chainpack [--threads N] [--nodes N] [--depth D] [--eval-only] [--nnue-off] [--nnue-on] [--batch-size N] <input.pgn|zst> [<input.pgn|zst> ...] <output.binpack>"
        );
        exit(1);
    }

    let mut threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut nodes: u64 = DEFAULT_NODES;
    let mut depth: i32 = DEFAULT_DEPTH;
    let mut eval_only = false;
    let mut nnue_off = false;
    let mut nnue_on = false;
    let mut batch_size: usize = DEFAULT_BATCH_SIZE;
    let mut input_paths: Vec<String> = Vec::new();
    let mut output_path: Option<String> = None;

    let mut iter = args[1..].iter().peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--threads" => {
                threads = iter
                    .next()
                    .map(|v| v.parse().unwrap_or(threads))
                    .unwrap_or(threads);
            }
            "--nodes" => {
                nodes = iter
                    .next()
                    .map(|v| v.parse().unwrap_or(nodes))
                    .unwrap_or(nodes);
            }
            "--depth" => {
                depth = iter
                    .next()
                    .map(|v| v.parse().unwrap_or(depth))
                    .unwrap_or(depth);
            }
            "--eval-only" => eval_only = true,
            "--nnue-off" => nnue_off = true,
            "--nnue-on" => nnue_on = true,
            "--batch-size" => {
                batch_size = iter
                    .next()
                    .map(|v| v.parse().unwrap_or(batch_size))
                    .unwrap_or(batch_size);
            }
            _ if output_path.is_none() && !input_paths.is_empty() && arg.ends_with(".binpack") => {
                output_path = Some(arg.clone());
            }
            _ => input_paths.push(arg.clone()),
        }
    }

    let output_path = match output_path {
        Some(p) => p,
        None => {
            eprintln!("chainpack: no output .binpack path specified");
            exit(1);
        }
    };

    if input_paths.is_empty() {
        eprintln!("chainpack: no input .pgn files specified");
        exit(1);
    }

    let output_path_buf = PathBuf::from(&output_path);
    let state_path = state_path_for(&output_path_buf);

    let state = Arc::new(Mutex::new(load_state(&state_path).unwrap_or_default()));
    let resume = !state.lock().unwrap().current_file.is_empty();

    let (start_file_index, skip_games_in_current) = {
        let s = state.lock().unwrap();
        let mut sfi = 0usize;
        let mut sgic = 0u64;
        if !s.current_file.is_empty() {
            for (i, p) in input_paths.iter().enumerate() {
                if p == &s.current_file {
                    sfi = i;
                    sgic = s.games_completed;
                    break;
                }
                sfi = i + 1;
            }
        }
        (sfi, sgic)
    };

    if resume {
        let s = state.lock().unwrap();
        eprintln!(
            "Resume: {} positions already written, skipping {} games in {}",
            s.positions_written, skip_games_in_current, s.current_file
        );
    }

    let (game_tx, game_rx) = mpsc::channel::<(usize, String)>();
    let (result_tx, result_rx) = mpsc::channel::<ChainBatch>();

    let game_rx_arc = Arc::new(Mutex::new(game_rx));

    let mut worker_handles = Vec::new();
    for _ in 0..threads {
        let game_rx = game_rx_arc.clone();
        let result_tx = result_tx.clone();
        let nodes = nodes;
        let depth = depth;
        let eval_only = eval_only;
        let nnue_off = nnue_off;
        let nnue_on = nnue_on;

        worker_handles.push(
            thread::Builder::new()
                .stack_size(SEARCH_THREAD_STACK_SIZE)
                .spawn(move || {
                    let mut processor =
                        GameProcessor::new(nodes, depth, eval_only, nnue_off, nnue_on);
                    loop {
                        let (file_idx, game_text) = {
                            let rx = game_rx.lock().unwrap();
                            match rx.recv() {
                                Ok(g) => g,
                                Err(_) => break,
                            }
                        };
                        let game = match parse_pgn(&game_text) {
                            Some(g) => g,
                            None => {
                                processor.skipped_games += 1;
                                continue;
                            }
                        };
                        let entries = processor.process_game(&game);
                        let _ = result_tx.send(ChainBatch { file_idx, entries });
                    }
                    eprintln!(
                        "worker: {} games, {} positions, {} skipped games, {} skipped positions",
                        processor.total_games,
                        processor.total_positions,
                        processor.skipped_games,
                        processor.skipped_positions,
                    );
                })
                .unwrap(),
        );
    }
    drop(result_tx);

    let producer = thread::spawn({
        let input_paths = input_paths.clone();
        let game_tx = game_tx;
        let state = state.clone();
        let skip_games = skip_games_in_current;
        move || {
            let mut games_sent: u64 = 0;
            for (file_idx, path) in input_paths.iter().enumerate() {
                if file_idx < start_file_index {
                    continue;
                }
                let reader = match open_input_reader(path) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("cannot open {path}: {e}");
                        continue;
                    }
                };
                let mut games_in_file: u64 = 0;
                for game in GameStream::new(reader) {
                    if file_idx == start_file_index && games_in_file < skip_games {
                        games_in_file += 1;
                        continue;
                    }
                    if game_tx.send((file_idx, game)).is_err() {
                        return;
                    }
                    games_in_file += 1;
                    games_sent += 1;
                    if games_sent.is_multiple_of(10_000) {
                        eprintln!("producer: {} games sent", games_sent);
                    }
                    {
                        let mut s = state.lock().unwrap();
                        s.current_file = path.clone();
                        s.games_completed = games_in_file;
                    }
                }
                eprintln!(
                    "producer: file {} done, {} games",
                    Path::new(path).display(),
                    games_in_file
                );
            }
        }
    });

    let mut out_file = if resume {
        match File::options().create(true).append(true).open(&output_path) {
            Ok(f) => f,
            Err(_) => File::create(&output_path).expect("failed to create output"),
        }
    } else {
        File::create(&output_path).expect("failed to create output")
    };

    let mut total_entries: u64 = 0;
    let mut games_completed_per_file: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();
    if resume {
        let (_bytes_on_disk, entries_on_disk) = match truncate_to_last_complete_chunk(&mut out_file)
        {
            Ok(n) => n,
            Err(e) => {
                eprintln!("chainpack: cannot rescan output file: {e}");
                exit(1);
            }
        };
        total_entries = entries_on_disk;
        let s = state.lock().unwrap();
        if !s.current_file.is_empty() {
            for (i, p) in input_paths.iter().enumerate() {
                if p == &s.current_file {
                    games_completed_per_file.insert(i, s.games_completed);
                    break;
                }
            }
        }
    }

    let mut writer = ChunkedWriter::new(out_file);
    let mut batch_entries: Vec<ChainEntryBytes> = Vec::new();
    let mut positions_since_state: u64 = 0;
    let mut last_report = Instant::now();

    for batch in result_rx {
        let n_entries = batch.entries.len() as u64;
        batch_entries.extend(batch.entries);
        positions_since_state += n_entries;

        *games_completed_per_file.entry(batch.file_idx).or_default() += 1;

        if last_report.elapsed().as_secs() >= 5 {
            eprintln!("progress: {} entries written", total_entries);
            last_report = Instant::now();
        }

        if batch_entries.len() >= batch_size {
            for entry in batch_entries.drain(..) {
                if let Err(e) = writer.write_entry(&entry.stem, &entry.movetext) {
                    eprintln!("chainpack: write error: {e}");
                    exit(1);
                }
            }
            total_entries = writer.entries_flushed();

            if positions_since_state >= CHAINPACK_STATE_SAVE_EVERY {
                {
                    let mut s = state.lock().unwrap();
                    s.positions_written = total_entries;
                    if let Some(path_idx) = input_paths.iter().position(|p| p == &s.current_file) {
                        s.games_completed = games_completed_per_file
                            .get(&path_idx)
                            .copied()
                            .unwrap_or(0);
                    }
                    save_state(&state_path, &s);
                }
                positions_since_state = 0;
            }
        }
    }

    if !batch_entries.is_empty() {
        for entry in batch_entries.drain(..) {
            if let Err(e) = writer.write_entry(&entry.stem, &entry.movetext) {
                eprintln!("chainpack: write error: {e}");
                exit(1);
            }
        }
    }

    let (total_bytes, total_chunks, total_entries_flushed) = match writer.flush() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("chainpack: final write error: {e}");
            exit(1);
        }
    };
    total_entries = total_entries_flushed;

    {
        let mut s = state.lock().unwrap();
        s.positions_written = total_entries;
        if let Some(path_idx) = input_paths.iter().position(|p| p == &s.current_file) {
            s.games_completed = games_completed_per_file
                .get(&path_idx)
                .copied()
                .unwrap_or(0);
        }
        save_state(&state_path, &s);
    }

    let _ = fs::remove_file(&state_path);

    let _ = producer.join();
    for h in worker_handles {
        let _ = h.join();
    }

    println!(
        "chainpack: wrote {} chain entries in {} chunks from {} input file(s) -> {} ({} bytes)",
        total_entries,
        total_chunks,
        input_paths.len(),
        output_path,
        total_bytes
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_from_fen(fen: &str) -> BoardState {
        let mut engine = Engine::try_new().expect("failed to create engine");
        engine.set_fen(fen);
        engine.st
    }

    fn nibbles(pos: &ChainPosition) -> Vec<u8> {
        let mut out = Vec::new();
        let mut occ = pos.occupied;
        let mut idx = 0usize;
        while occ != 0 {
            occ &= occ - 1;
            let byte = pos.packed_state[idx / 2];
            out.push(if idx.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            });
            idx += 1;
        }
        out
    }

    fn nibble_at(pos: &ChainPosition, sq_int: usize) -> u8 {
        let before = (pos.occupied & ((1u64 << sq_int) - 1)).count_ones() as usize;
        let byte = pos.packed_state[before / 2];
        if before.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    #[test]
    fn ep_pawn_nibble_marks_the_capturable_pawn() {
        let st = state_from_fen("rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq d6 0 2");
        assert_eq!(st.ep, Some(sq(2, 3)));
        let pos = ChainPosition::from_state(&st);
        assert_eq!(nibble_at(&pos, sq(3, 3)), 12);
    }

    #[test]
    fn black_king_nibble_encodes_side_to_move() {
        let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
        let pos = ChainPosition::from_state(&st);
        assert_eq!(nibble_at(&pos, sq(7, 4)), 15);

        let st = state_from_fen(START_FEN);
        let pos = ChainPosition::from_state(&st);
        assert_eq!(nibble_at(&pos, sq(7, 4)), 11);
    }

    #[test]
    fn rooks_without_castling_rights_stay_plain() {
        let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
        let pos = ChainPosition::from_state(&st);
        let nibs = nibbles(&pos);
        assert_eq!(nibs[0], 6);
        assert_eq!(nibs[7], 6);
        assert_eq!(nibs[24], 7);
        assert_eq!(nibs[31], 7);
    }

    #[test]
    fn stem_layout_roundtrip() {
        let st = state_from_fen(START_FEN);
        let pos = ChainPosition::from_state(&st);
        let mv = ChainMove::from_uci("e2e4").unwrap();
        let stem = pack_stem(&pos, &mv, -123, 7, -1);

        assert_eq!(&stem[0..8], &pos.occupied.to_be_bytes());
        assert_eq!(&stem[8..24], &pos.packed_state);
        assert_eq!(u16::from_be_bytes([stem[24], stem[25]]), mv.compress());
        assert_eq!(
            u16::from_be_bytes([stem[26], stem[27]]),
            signed_to_unsigned(-123)
        );
        let pr = u16::from_be_bytes([stem[28], stem[29]]);
        assert_eq!(pr >> 14, signed_to_unsigned(-1));
        assert_eq!(pr & 0x3FFF, 7);
        assert_eq!(u16::from_be_bytes([stem[30], stem[31]]), 0);
    }

    fn san(st: &BoardState, san: &str) -> Option<String> {
        parse_san(st, san).map(|mv| move_to_uci(st, mv))
    }

    #[test]
    fn san_pawn_captures() {
        let st = state_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3");
        assert_eq!(san(&st, "exd5"), Some("e4d5".to_string()));
        assert_eq!(san(&st, "exd5!"), Some("e4d5".to_string()));
    }

    #[test]
    fn san_en_passant() {
        let st = state_from_fen("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2");
        assert_eq!(san(&st, "dxe3e.p."), Some("d4e3".to_string()));
        assert_eq!(san(&st, "dxe3"), Some("d4e3".to_string()));
    }

    #[test]
    fn san_promotions() {
        let st = state_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        assert_eq!(san(&st, "a8=Q"), Some("a7a8q".to_string()));
        assert_eq!(san(&st, "a8=N"), Some("a7a8n".to_string()));
        assert_eq!(san(&st, "a8=Q!"), Some("a7a8q".to_string()));
    }

    #[test]
    fn san_castling() {
        let st = state_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(san(&st, "O-O"), Some("e1g1".to_string()));
        assert_eq!(san(&st, "0-0"), Some("e1g1".to_string()));
        assert_eq!(san(&st, "O-O-O"), Some("e1c1".to_string()));
        assert_eq!(san(&st, "0-0-0"), Some("e1c1".to_string()));
    }

    #[test]
    fn san_disambiguation_and_invalid() {
        let st = state_from_fen("4k3/8/3R4/8/8/8/8/R3K3 w - - 0 1");
        assert_eq!(san(&st, "Rd1"), None);
        assert_eq!(san(&st, "R1d1"), Some("a1d1".to_string()));
        assert_eq!(san(&st, "R6d1"), Some("d6d1".to_string()));
        assert_eq!(san(&st, "zzz"), None);
        assert_eq!(san(&st, ""), None);
        assert_eq!(san(&st, "e4"), None);
    }
}
