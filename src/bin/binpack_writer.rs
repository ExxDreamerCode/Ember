use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::exit;

const MAGIC: &[u8; 4] = b"BINP";
const CHUNK_TARGET_BYTES: usize = 32 * 1024 * 1024;
const MAX_SCORE_ABS: i32 = 1500;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Square(u8);

impl Square {
    fn from_uci(s: &str) -> Option<Self> {
        if s.len() != 2 {
            return None;
        }
        let file = s.as_bytes()[0].checked_sub(b'a')?;
        let rank = s.as_bytes()[1].checked_sub(b'1')?;
        if file > 7 || rank > 7 {
            return None;
        }
        Some(Square(rank * 8 + file))
    }

    fn index(self) -> u8 {
        self.0
    }

    fn file(self) -> u8 {
        self.0 % 8
    }

    fn rank(self) -> u8 {
        self.0 / 8
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
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

    fn ptype(self) -> Option<PieceType> {
        match self {
            Piece::WhitePawn | Piece::BlackPawn => Some(PieceType::Pawn),
            Piece::WhiteKnight | Piece::BlackKnight => Some(PieceType::Knight),
            Piece::WhiteBishop | Piece::BlackBishop => Some(PieceType::Bishop),
            Piece::WhiteRook | Piece::BlackRook => Some(PieceType::Rook),
            Piece::WhiteQueen | Piece::BlackQueen => Some(PieceType::Queen),
            Piece::WhiteKing | Piece::BlackKing => Some(PieceType::King),
            Piece::None => None,
        }
    }

    fn color(self) -> Option<Color> {
        match self {
            Piece::WhitePawn
            | Piece::WhiteKnight
            | Piece::WhiteBishop
            | Piece::WhiteRook
            | Piece::WhiteQueen
            | Piece::WhiteKing => Some(Color::White),
            Piece::BlackPawn
            | Piece::BlackKnight
            | Piece::BlackBishop
            | Piece::BlackRook
            | Piece::BlackQueen
            | Piece::BlackKing => Some(Color::Black),
            Piece::None => None,
        }
    }

    fn nibble(self) -> u8 {
        match self {
            Piece::WhitePawn => 0,
            Piece::BlackPawn => 1,
            Piece::WhiteKnight => 2,
            Piece::BlackKnight => 3,
            Piece::WhiteBishop => 4,
            Piece::BlackBishop => 5,
            Piece::WhiteRook => 6,
            Piece::BlackRook => 7,
            Piece::WhiteQueen => 8,
            Piece::BlackQueen => 9,
            Piece::WhiteKing => 10,
            Piece::BlackKing => 11,
            Piece::None => 0,
        }
    }
}

#[derive(Clone, Copy)]
struct Position {
    board: [Piece; 64],
    side_to_move: Color,
    white_ks: bool,
    white_qs: bool,
    black_ks: bool,
    black_qs: bool,
    ep_square: Option<Square>,
    halfmove_clock: u16,
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
                    if file_idx > 7 {
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

        let (mut white_ks, mut white_qs, mut black_ks, mut black_qs) =
            (false, false, false, false);
        for ch in parts[2].chars() {
            match ch {
                'K' => white_ks = true,
                'Q' => white_qs = true,
                'k' => black_ks = true,
                'q' => black_qs = true,
                '-' => {}
                _ => return None,
            }
        }

        let ep_square = if parts[3] == "-" {
            None
        } else {
            Square::from_uci(parts[3])
        };

        let halfmove_clock: u16 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

        Some(Position {
            board,
            side_to_move,
            white_ks,
            white_qs,
            black_ks,
            black_qs,
            ep_square,
            halfmove_clock,
        })
    }

    fn piece_at(&self, sq: Square) -> Piece {
        self.board[sq.index() as usize]
    }

    fn is_square_attacked(&self, target: Square, by: Color) -> bool {
        let tf = target.file() as i32;
        let tr = target.rank() as i32;
        let t = target.index() as i32;

        let pawn = match by {
            Color::White => Piece::WhitePawn,
            Color::Black => Piece::BlackPawn,
        };
        let pawn_origins: [i32; 2] = match by {
            Color::White => [t - 7, t - 9],
            Color::Black => [t + 7, t + 9],
        };
        let pawn_ok: [bool; 2] = match by {
            Color::White => [tf < 7, tf > 0],
            Color::Black => [tf > 0, tf < 7],
        };
        for (origin, ok) in pawn_origins.iter().zip(pawn_ok.iter()) {
            if *ok && (0..64).contains(origin) && self.piece_at(Square(*origin as u8)) == pawn {
                return true;
            }
        }

        let knight = match by {
            Color::White => Piece::WhiteKnight,
            Color::Black => Piece::BlackKnight,
        };
        const KNIGHT_DELTAS: [(i32, i32); 8] = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        for (df, dr) in KNIGHT_DELTAS {
            let f = tf + df;
            let r = tr + dr;
            if (0..8).contains(&f) && (0..8).contains(&r) && self.piece_at(Square((r * 8 + f) as u8)) == knight {
                return true;
            }
        }

        let king = match by {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        };
        const KING_DELTAS: [(i32, i32); 8] = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];
        for (df, dr) in KING_DELTAS {
            let f = tf + df;
            let r = tr + dr;
            if (0..8).contains(&f) && (0..8).contains(&r) && self.piece_at(Square((r * 8 + f) as u8)) == king {
                return true;
            }
        }

        let rook = match by {
            Color::White => Piece::WhiteRook,
            Color::Black => Piece::BlackRook,
        };
        let bishop = match by {
            Color::White => Piece::WhiteBishop,
            Color::Black => Piece::BlackBishop,
        };
        let queen = match by {
            Color::White => Piece::WhiteQueen,
            Color::Black => Piece::BlackQueen,
        };

        const ORTHO_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        const DIAG_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

        for (dirs, slider) in [(&ORTHO_DIRS, rook), (&DIAG_DIRS, bishop)] {
            for (df, dr) in *dirs {
                let mut f = tf + df;
                let mut r = tr + dr;
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    let piece = self.piece_at(Square((r * 8 + f) as u8));
                    if piece != Piece::None {
                        if piece == slider || piece == queen {
                            return true;
                        }
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
        }

        false
    }

    fn in_check(&self, color: Color) -> Option<bool> {
        let target = match color {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        };
        for sq in 0..64u8 {
            if self.board[sq as usize] == target {
                return Some(self.is_square_attacked(Square(sq), color.flip()));
            }
        }
        None
    }

    fn compress_position(&self) -> [u8; 24] {
        let mut out = [0u8; 24];

        let mut occupied: u64 = 0;
        for sq in 0..64u8 {
            if self.board[sq as usize] != Piece::None {
                occupied |= 1u64 << sq;
            }
        }
        out[0..8].copy_from_slice(&occupied.to_be_bytes());

        let mut nibble_idx = 0usize;
        for sq in 0..64u8 {
            if self.board[sq as usize] == Piece::None {
                continue;
            }
            let nibble = self.compress_piece_nibble(Square(sq), self.board[sq as usize]);
            if nibble_idx.is_multiple_of(2) {
                out[8 + nibble_idx / 2] = nibble;
            } else {
                out[8 + nibble_idx / 2] |= nibble << 4;
            }
            nibble_idx += 1;
        }

        out
    }

    fn compress_piece_nibble(&self, sq: Square, piece: Piece) -> u8 {
        match piece.ptype() {
            Some(PieceType::Pawn) => {
                if let Some(ep) = self.ep_square {
                    let same_file = sq.file() == ep.file();
                    let white_ep_pawn = sq.rank() == 3 && self.side_to_move == Color::Black;
                    let black_ep_pawn = sq.rank() == 4 && self.side_to_move == Color::White;
                    if same_file && (white_ep_pawn || black_ep_pawn) {
                        return 12;
                    }
                }
                piece.nibble()
            }
            Some(PieceType::Rook) => {
                if piece.color() == Some(Color::White) {
                    if (sq.index() == 0 && self.white_qs) || (sq.index() == 7 && self.white_ks) {
                        return 13;
                    }
                } else if (sq.index() == 56 && self.black_qs) || (sq.index() == 63 && self.black_ks)
                {
                    return 14;
                }
                piece.nibble()
            }
            Some(PieceType::King) => {
                if piece.color() == Some(Color::White) {
                    10
                } else if self.side_to_move == Color::White {
                    11
                } else {
                    15
                }
            }
            _ => piece.nibble(),
        }
    }
}

#[derive(Clone, Copy)]
struct UciMove {
    from: Square,
    to: Square,
    promotion: Option<PieceType>,
    is_null: bool,
}

impl UciMove {
    fn parse(uci: &str) -> Self {
        if uci.len() < 4 || uci == "0000" {
            return UciMove {
                from: Square(0),
                to: Square(0),
                promotion: None,
                is_null: true,
            };
        }
        let (from, to) = match (Square::from_uci(&uci[0..2]), Square::from_uci(&uci[2..4])) {
            (Some(f), Some(t)) => (f, t),
            _ => {
                return UciMove {
                    from: Square(0),
                    to: Square(0),
                    promotion: None,
                    is_null: true,
                }
            }
        };
        let promotion = if uci.len() >= 5 {
            match uci.as_bytes()[4] {
                b'q' | b'Q' => Some(PieceType::Queen),
                b'r' | b'R' => Some(PieceType::Rook),
                b'b' | b'B' => Some(PieceType::Bishop),
                b'n' | b'N' => Some(PieceType::Knight),
                _ => None,
            }
        } else {
            None
        };
        UciMove {
            from,
            to,
            promotion,
            is_null: false,
        }
    }

    fn compress(&self) -> u16 {
        if self.is_null {
            return 0;
        }
        let move_type: u16 = if self.promotion.is_some() { 1 } else { 0 };
        let mut packed = move_type << 14;
        packed |= (self.from.index() as u16) << 8;
        packed |= (self.to.index() as u16) << 2;
        if let Some(pt) = self.promotion {
            packed |= (pt as u16 - PieceType::Knight as u16) & 0b11;
        }
        packed
    }

    fn is_tactical(&self, pos: &Position) -> bool {
        if self.is_null {
            return false;
        }
        if self.promotion.is_some() {
            return true;
        }
        if pos.piece_at(self.to) != Piece::None {
            return true;
        }
        let mover = pos.piece_at(self.from);
        match mover.ptype() {
            Some(PieceType::Pawn) => self.from.file() != self.to.file(),
            Some(PieceType::King) => {
                (self.from.file() as i32 - self.to.file() as i32).abs() == 2
            }
            _ => false,
        }
    }
}

fn signed_to_unsigned(value: i16) -> u16 {
    let bits = value as u16;
    if bits & 0x8000 != 0 {
        (bits ^ 0x7FFF).rotate_left(1)
    } else {
        bits.rotate_left(1)
    }
}

struct Entry {
    pos: Position,
    mv: UciMove,
    score: i16,
    ply: u16,
    result: i16,
    rule50: u16,
}

impl Entry {
    fn chain_bytes(&self) -> [u8; 34] {
        let mut out = [0u8; 34];
        out[0..24].copy_from_slice(&self.pos.compress_position());
        out[24..26].copy_from_slice(&self.mv.compress().to_be_bytes());
        out[26..28].copy_from_slice(&signed_to_unsigned(self.score).to_be_bytes());
        let pr = signed_to_unsigned(self.result) << 14 | (self.ply & 0x3FFF);
        out[28..30].copy_from_slice(&pr.to_be_bytes());
        out[30..32].copy_from_slice(&self.rule50.to_be_bytes());
        out[32..34].copy_from_slice(&0u16.to_be_bytes());
        out
    }
}

struct ChunkWriter {
    buf: Vec<u8>,
}

impl ChunkWriter {
    fn new() -> Self {
        ChunkWriter { buf: Vec::new() }
    }

    fn add_entry(&mut self, entry: &Entry) {
        self.buf.extend_from_slice(&entry.chain_bytes());
    }

    fn is_full(&self) -> bool {
        self.buf.len() >= CHUNK_TARGET_BYTES
    }

    fn write_to<W: Write>(&mut self, out: &mut W) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        out.write_all(MAGIC)?;
        out.write_all(&(self.buf.len() as u32).to_le_bytes())?;
        out.write_all(&self.buf)?;
        self.buf.clear();
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
    let mut skipped_check = 0u64;
    let mut skipped_tactical = 0u64;
    let mut skipped_score = 0u64;
    let mut skipped_malformed = 0u64;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => {
                skipped_malformed += 1;
                continue;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() < 6 {
            skipped_malformed += 1;
            continue;
        }

        let fen = parts[0];
        let score_val: i32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped_malformed += 1;
                continue;
            }
        };
        let bestmove_str = parts[3];
        let ply: u16 = match parts[4].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped_malformed += 1;
                continue;
            }
        };
        let result: i16 = match parts[5].parse::<i16>() {
            Ok(v) => v.clamp(-1, 1),
            Err(_) => {
                skipped_malformed += 1;
                continue;
            }
        };

        let pos = match Position::from_fen(fen) {
            Some(p) => p,
            None => {
                skipped_malformed += 1;
                continue;
            }
        };

        if pos.in_check(pos.side_to_move).unwrap_or(true) {
            skipped_check += 1;
            continue;
        }

        if score_val.abs() > MAX_SCORE_ABS {
            skipped_score += 1;
            continue;
        }

        let mv = UciMove::parse(bestmove_str);
        if mv.is_tactical(&pos) {
            skipped_tactical += 1;
            continue;
        }

        let score = score_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

        let entry = Entry {
            pos,
            mv,
            score,
            ply,
            result,
            rule50: pos.halfmove_clock,
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

    println!(
        "binpack_writer: wrote {count} entries (check {skipped_check}, tactical {skipped_tactical}, score {skipped_score}, malformed {skipped_malformed}) -> {output_path}"
    );
}
