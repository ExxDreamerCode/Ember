use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::process::exit;

const MAGIC: &[u8; 4] = b"BINP";
const CHUNK_TARGET_SIZE: usize = 1_000_000;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Square(u8);

impl Square {
    fn from_uci(s: &str) -> Option<Self> {
        if s.len() != 2 {
            return None;
        }
        let file = s.as_bytes()[0] - b'a';
        let rank = s.as_bytes()[1] - b'1';
        if file > 7 || rank > 7 {
            return None;
        }
        Some(Square(rank * 8 + file))
    }

    fn index(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White = 0,
    Black = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Piece {
    None,
    WhitePawn,
    WhiteKnight,
    WhiteBishop,
    WhiteRook,
    WhiteQueen,
    WhiteKing,
    BlackPawn,
    BlackKnight,
    BlackBishop,
    BlackRook,
    BlackQueen,
    BlackKing,
}

impl Piece {
    fn from_char(c: char) -> Option<Self> {
        match c {
            'P' => Some(Piece::WhitePawn),
            'N' => Some(Piece::WhiteKnight),
            'B' => Some(Piece::WhiteBishop),
            'R' => Some(Piece::WhiteRook),
            'Q' => Some(Piece::WhiteQueen),
            'K' => Some(Piece::WhiteKing),
            'p' => Some(Piece::BlackPawn),
            'n' => Some(Piece::BlackKnight),
            'b' => Some(Piece::BlackBishop),
            'r' => Some(Piece::BlackRook),
            'q' => Some(Piece::BlackQueen),
            'k' => Some(Piece::BlackKing),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Position {
    board: [Piece; 64],
    side_to_move: Color,
    castling: u8,
    ep_square: Option<Square>,
    halfmove_clock: u8,
    fullmove_number: u16,
}

impl Position {
    fn from_fen(fen: &str) -> Option<Self> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        let mut board = [Piece::None; 64];
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return None;
        }
        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let board_rank = 7 - rank_idx;
            let mut file_idx: usize = 0;
            for ch in rank_str.chars() {
                if ch.is_ascii_digit() {
                    file_idx += ch.to_digit(10)? as usize;
                } else {
                    let piece = Piece::from_char(ch)?;
                    if file_idx >= 64 {
                        return None;
                    }
                    board[board_rank * 8 + file_idx] = piece;
                    file_idx += 1;
                }
            }
            if file_idx != 8 {
                return None;
            }
        }

        let side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return None,
        };

        let mut castling: u8 = 0;
        for ch in parts[2].chars() {
            match ch {
                'K' => castling |= 1,
                'Q' => castling |= 2,
                'k' => castling |= 4,
                'q' => castling |= 8,
                '-' => {}
                _ => return None,
            }
        }

        let ep_square = if parts[3] == "-" {
            None
        } else {
            Square::from_uci(parts[3])
        };

        let halfmove_clock: u8 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let fullmove_number: u16 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        Some(Position {
            board,
            side_to_move,
            castling,
            ep_square,
            halfmove_clock,
            fullmove_number,
        })
    }

    fn pack_sfen(&self) -> [u8; 32] {
        let mut bits: Vec<bool> = Vec::with_capacity(256);

        bits.push(self.side_to_move == Color::Black);

        bits.push(self.castling & 1 != 0);
        bits.push(self.castling & 2 != 0);
        bits.push(self.castling & 4 != 0);
        bits.push(self.castling & 8 != 0);

        let has_ep = self.ep_square.is_some();
        bits.push(has_ep);
        if let Some(sq) = self.ep_square {
            for i in 0..6 {
                bits.push((sq.index() >> i) & 1 == 1);
            }
        } else {
            bits.extend(std::iter::repeat_n(false, 6));
        }

        for i in 0..8 {
            bits.push((self.halfmove_clock >> i) & 1 == 1);
        }

        for i in 0..12 {
            bits.push((self.fullmove_number >> i) & 1 == 1);
        }

        let mut occupied = 0u64;
        for sq in 0..64u8 {
            if self.board[sq as usize] != Piece::None {
                occupied |= 1u64 << sq;
            }
        }
        for i in 0..64 {
            bits.push((occupied >> i) & 1 == 1);
        }

        for sq in 0..64u8 {
            let piece = self.board[sq as usize];
            if piece == Piece::None {
                continue;
            }
            let (code, code_len) = self.piece_code(piece);
            for i in 0..code_len as usize {
                bits.push(((code >> i) & 1) == 1);
            }
        }

        bits.resize(256, false);

        let mut bytes = [0u8; 32];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        bytes
    }

    fn piece_code(&self, piece: Piece) -> (u16, u8) {
        match piece {
            Piece::WhitePawn => (0b0000, 4),
            Piece::BlackPawn => (0b0001, 4),
            Piece::WhiteKnight => (0b0010, 4),
            Piece::BlackKnight => (0b0011, 4),
            Piece::WhiteBishop => (0b0100, 4),
            Piece::BlackBishop => (0b0101, 4),
            Piece::WhiteRook => (0b0110, 4),
            Piece::BlackRook => (0b0111, 4),
            Piece::WhiteQueen => (0b1000, 4),
            Piece::BlackQueen => (0b1001, 4),
            Piece::WhiteKing => (0b1010, 4),
            Piece::BlackKing => (0b1011, 4),
            Piece::None => (0, 0),
        }
    }
}

#[derive(Clone, Copy)]
struct PackedMove {
    from: u8,
    to: u8,
    promo: u8,
    flags: u8,
}

impl PackedMove {
    fn null() -> Self {
        PackedMove {
            from: 0,
            to: 0,
            promo: 0,
            flags: 0,
        }
    }

    fn from_uci(uci: &str) -> Self {
        if uci.len() < 4 || uci == "0000" {
            return Self::null();
        }
        let from = Square::from_uci(&uci[0..2]).map(|s| s.index()).unwrap_or(0);
        let to = Square::from_uci(&uci[2..4]).map(|s| s.index()).unwrap_or(0);
        let promo = if uci.len() >= 5 {
            match uci.as_bytes()[4] {
                b'q' | b'Q' => 1,
                b'r' | b'R' => 2,
                b'b' | b'B' => 3,
                b'n' | b'N' => 4,
                _ => 0,
            }
        } else {
            0
        };
        PackedMove {
            from,
            to,
            promo,
            flags: 0,
        }
    }

    fn to_bytes(self) -> [u8; 4] {
        [self.from, self.to, self.promo, self.flags]
    }
}

#[derive(Clone)]
struct Entry {
    pos: Position,
    mv: PackedMove,
    score: i16,
    ply: u16,
    result: i16,
}

struct ChunkWriter {
    entries: Vec<Entry>,
    current_size: usize,
}

impl ChunkWriter {
    fn new() -> Self {
        ChunkWriter {
            entries: Vec::new(),
            current_size: 8,
        }
    }

    fn add_entry(&mut self, entry: &Entry) {
        let entry_size = 32 + 4 + 2 + 2 + 2;
        self.entries.push(entry.clone());
        self.current_size += entry_size;
    }

    fn is_full(&self) -> bool {
        self.current_size >= CHUNK_TARGET_SIZE && !self.entries.is_empty()
    }

    fn write_to<W: Write + Seek>(&mut self, out: &mut W) -> std::io::Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }

        let chunk_start = out.stream_position()?;
        out.write_all(MAGIC)?;
        let size_pos = out.stream_position()?;
        out.write_all(&[0u8; 4])?;

        let mut chain_data: Vec<u8> = Vec::new();

        for entry in &self.entries {
            let packed = entry.pos.pack_sfen();
            chain_data.extend_from_slice(&packed);

            chain_data.extend_from_slice(&1u32.to_le_bytes());

            chain_data.extend_from_slice(&entry.mv.to_bytes());
            chain_data.extend_from_slice(&entry.score.to_le_bytes());
            chain_data.extend_from_slice(&entry.ply.to_le_bytes());
            chain_data.extend_from_slice(&entry.result.to_le_bytes());
        }

        let chunk_size = chain_data.len() as u32;
        out.seek(SeekFrom::Start(size_pos))?;
        out.write_all(&chunk_size.to_le_bytes())?;
        out.seek(SeekFrom::Start(chunk_start + 8))?;

        out.write_all(&chain_data)?;

        self.entries.clear();
        self.current_size = 8;
        Ok(())
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: binpack_writer <input_scored.tsv> <output.binpack>");
        exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let reader = match File::open(input_path) {
        Ok(f) => BufReader::new(f),
        Err(e) => {
            eprintln!("binpack_writer: cannot open input {input_path}: {e}");
            exit(1);
        }
    };

    let mut out_file = match File::create(output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("binpack_writer: cannot create output {output_path}: {e}");
            exit(1);
        }
    };

    let mut chunk = ChunkWriter::new();
    let mut count: u64 = 0;
    let mut skipped: u64 = 0;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() < 3 {
            skipped += 1;
            continue;
        }

        let fen = parts[0];
        let _score_type = parts[1];
        let score_val: i32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let pos = match Position::from_fen(fen) {
            Some(p) => p,
            None => {
                skipped += 1;
                continue;
            }
        };

        let score = score_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let mv = if parts.len() >= 4 && !parts[3].is_empty() && parts[3] != "0000" {
            PackedMove::from_uci(parts[3])
        } else {
            PackedMove::null()
        };

        let entry = Entry {
            pos,
            mv,
            score,
            ply: 0,
            result: 0,
        };

        if chunk.is_full() {
            if let Err(e) = chunk.write_to(&mut out_file) {
                eprintln!("binpack_writer: write error: {e}");
                exit(1);
            }
        }

        chunk.add_entry(&entry);
        count += 1;
    }

    if let Err(e) = chunk.write_to(&mut out_file) {
        eprintln!("binpack_writer: final write error: {e}");
        exit(1);
    }

    out_file.flush().unwrap();

    println!("binpack_writer: wrote {count} entries, skipped {skipped} -> {output_path}");
}
