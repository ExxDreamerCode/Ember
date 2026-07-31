use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use sfbinpack::{
    chess::{
        color::Color,
        coords::Square,
        piece::Piece,
        piecetype::PieceType,
        r#move::MoveType,
    },
    CompressedTrainingDataEntryReader, TrainingDataEntry,
};

const RECORD_SIZE: usize = 77;

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    max_samples: usize,
    min_ply: u16,
    max_abs_score: u16,
    quiet_only: bool,
}

fn main() {
    let args = parse_args();
    fs::create_dir_all(args.output.parent().unwrap()).expect("create output directory");

    let input = File::open(&args.input).expect("open binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(input).expect("read binpack");
    let mut out = BufWriter::new(File::create(&args.output).expect("create output"));

    out.write_all(b"EGNNROOT1").expect("write magic");
    out.write_all(&(RECORD_SIZE as u32).to_le_bytes())
        .expect("write record size");
    out.write_all(&0u32.to_le_bytes()).expect("write reserved");

    let started = Instant::now();
    let mut seen = 0usize;
    let mut kept = 0usize;
    let mut skipped_filter = 0usize;
    let mut skipped_map = 0usize;

    while reader.has_next() && kept < args.max_samples {
        let entry = reader.next();
        seen += 1;

        if !accept_entry(&entry, &args) {
            skipped_filter += 1;
            continue;
        }

        let uci = entry.mv.as_uci();
        let Some((move_index, from, to, promo)) = move_index_from_uci(&uci) else {
            skipped_map += 1;
            continue;
        };

        write_record(&mut out, &entry, move_index, from, to, promo).expect("write record");
        kept += 1;

        if kept % 1_000_000 == 0 {
            eprintln!(
                "kept={kept} seen={seen} skipped_filter={skipped_filter} skipped_map={skipped_map} elapsed={:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
    }

    out.flush().expect("flush output");
    eprintln!(
        "done kept={kept} seen={seen} skipped_filter={skipped_filter} skipped_map={skipped_map} elapsed={:.1}s output={}",
        started.elapsed().as_secs_f64(),
        args.output.display()
    );
}

fn parse_args() -> Args {
    let mut input = None;
    let mut output = None;
    let mut max_samples = 5_000_000usize;
    let mut min_ply = 16u16;
    let mut max_abs_score = 10_000u16;
    let mut quiet_only = true;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--max-samples" => {
                max_samples = it
                    .next()
                    .expect("value for --max-samples")
                    .parse()
                    .expect("parse --max-samples")
            }
            "--min-ply" => {
                min_ply = it
                    .next()
                    .expect("value for --min-ply")
                    .parse()
                    .expect("parse --min-ply")
            }
            "--max-abs-score" => {
                max_abs_score = it
                    .next()
                    .expect("value for --max-abs-score")
                    .parse()
                    .expect("parse --max-abs-score")
            }
            "--quiet-only" => quiet_only = true,
            "--all-moves" => quiet_only = false,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }

    Args {
        input: input.expect("--input is required"),
        output: output.expect("--output is required"),
        max_samples,
        min_ply,
        max_abs_score,
        quiet_only,
    }
}

fn print_help() {
    println!(
        "extract_sfbinpack --input FILE.binpack --output samples.bin [--max-samples N] [--all-moves]"
    );
}

fn accept_entry(entry: &TrainingDataEntry, args: &Args) -> bool {
    if entry.ply < args.min_ply {
        return false;
    }
    if entry.score.unsigned_abs() > args.max_abs_score {
        return false;
    }
    if entry.pos.is_checked(entry.pos.side_to_move()) {
        return false;
    }
    if args.quiet_only {
        if entry.mv.mtype() != MoveType::Normal {
            return false;
        }
        if entry.pos.piece_at(entry.mv.to()).piece_type() != PieceType::None {
            return false;
        }
    }
    true
}

fn write_record<W: Write>(
    out: &mut W,
    entry: &TrainingDataEntry,
    move_index: u16,
    from: u8,
    to: u8,
    promo: u8,
) -> std::io::Result<()> {
    let mut record = [0u8; RECORD_SIZE];
    for sq in 0..64u8 {
        record[sq as usize] = piece_code(entry.pos.piece_at(Square::new(u32::from(sq))));
    }

    record[64] = match entry.pos.side_to_move() {
        Color::White => 0,
        Color::Black => 1,
    };
    record[65] = from;
    record[66] = to;
    record[67] = promo;
    record[68..70].copy_from_slice(&move_index.to_le_bytes());
    record[70..72].copy_from_slice(&entry.score.to_le_bytes());
    record[72] = match entry.result {
        -1 => 0,
        0 => 1,
        1 => 2,
        other => panic!("unexpected result {other}"),
    };
    record[73..75].copy_from_slice(&entry.ply.to_le_bytes());
    record[75..77].copy_from_slice(&0u16.to_le_bytes());
    out.write_all(&record)
}

fn piece_code(piece: Piece) -> u8 {
    match piece.piece_type() {
        PieceType::None => 0,
        piece_type => {
            let base = match piece.color() {
                Color::White => 0,
                Color::Black => 6,
            };
            base + piece_type.ordinal() + 1
        }
    }
}

fn move_index_from_uci(uci: &str) -> Option<(u16, u8, u8, u8)> {
    let bytes = uci.as_bytes();
    if bytes.len() != 4 && bytes.len() != 5 {
        return None;
    }
    let from = parse_square(&bytes[0..2])?;
    let to = parse_square(&bytes[2..4])?;
    let promo = if bytes.len() == 5 {
        match bytes[4] {
            b'n' => 1,
            b'b' => 2,
            b'r' => 3,
            b'q' => 4,
            _ => return None,
        }
    } else {
        0
    };
    let plane = move_plane(from, to, promo)?;
    Some((u16::from(from) * 73 + u16::from(plane), from, to, promo))
}

fn parse_square(s: &[u8]) -> Option<u8> {
    if s.len() != 2 || !(b'a'..=b'h').contains(&s[0]) || !(b'1'..=b'8').contains(&s[1]) {
        return None;
    }
    Some((s[0] - b'a') + 8 * (s[1] - b'1'))
}

fn move_plane(from: u8, to: u8, promo: u8) -> Option<u8> {
    let fx = (from % 8) as i8;
    let fy = (from / 8) as i8;
    let tx = (to % 8) as i8;
    let ty = (to / 8) as i8;
    let dx = tx - fx;
    let dy = ty - fy;

    if promo != 0 && promo != 4 {
        if !(-1..=1).contains(&dx) || dy == 0 || dy.abs() != 1 {
            return None;
        }
        let dir = (dx + 1) as u8;
        let piece = match promo {
            1 => 0,
            2 => 1,
            3 => 2,
            _ => return None,
        };
        return Some(64 + dir * 3 + piece);
    }

    if let Some(knight_plane) = knight_plane(dx, dy) {
        return Some(56 + knight_plane);
    }

    let (dir, dist) = ray_direction(dx, dy)?;
    Some(dir * 7 + (dist - 1))
}

fn knight_plane(dx: i8, dy: i8) -> Option<u8> {
    const KNIGHTS: [(i8, i8); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    KNIGHTS
        .iter()
        .position(|&(kx, ky)| kx == dx && ky == dy)
        .map(|idx| idx as u8)
}

fn ray_direction(dx: i8, dy: i8) -> Option<(u8, u8)> {
    const DIRS: [(i8, i8); 8] = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ];
    for (idx, (ux, uy)) in DIRS.iter().enumerate() {
        for dist in 1..=7i8 {
            if dx == ux * dist && dy == uy * dist {
                return Some((idx as u8, dist as u8));
            }
        }
    }
    None
}
