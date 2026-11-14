use ember_chess::board::{bit, see};

const WP: usize = 0;
const WR: usize = 3;
const WQ: usize = 4;
const WK: usize = 5;
const BP: usize = 6;
const BN: usize = 7;
const BR: usize = 9;
const BQ: usize = 10;
const BK: usize = 11;

type Piece = (usize, char, usize);

struct SeeCase {
    name: &'static str,
    pieces: Vec<Piece>,
    from: (char, usize),
    to: (char, usize),
    expected: i32,
}

fn sq(file: usize, rank: usize) -> usize {
    (8 - rank) * 8 + file
}

fn board(pieces: &[Piece]) -> [u64; 12] {
    let mut bb = [0u64; 12];
    for &(piece, file_char, rank) in pieces {
        bb[piece] |= bit(sq(file(file_char), rank));
    }
    bb
}

fn file(file_char: char) -> usize {
    file_char as usize - b'a' as usize
}

#[test]
fn see_reference_positions() {
    let cases = [
        SeeCase {
            name: "pawn takes defended pawn (dxe5, d6 defends)",
            pieces: vec![
                (WK, 'e', 1),
                (BK, 'e', 8),
                (WP, 'd', 4),
                (BP, 'e', 5),
                (BP, 'd', 6),
            ],
            from: ('d', 4),
            to: ('e', 5),
            expected: 0,
        },
        SeeCase {
            name: "pawn takes free pawn (dxe5, undefended)",
            pieces: vec![(WK, 'e', 1), (BK, 'e', 8), (WP, 'd', 4), (BP, 'e', 5)],
            from: ('d', 4),
            to: ('e', 5),
            expected: 100,
        },
        SeeCase {
            name: "queen takes defended pawn (Qxe5, d6 defends)",
            pieces: vec![
                (WK, 'e', 1),
                (BK, 'e', 8),
                (WQ, 'e', 2),
                (BP, 'e', 5),
                (BP, 'd', 6),
            ],
            from: ('e', 2),
            to: ('e', 5),
            expected: -850,
        },
        SeeCase {
            name: "rook takes defended rook (Rxd7, Rd8 defends)",
            pieces: vec![
                (WK, 'a', 1),
                (BK, 'a', 8),
                (WR, 'd', 2),
                (BR, 'd', 7),
                (BR, 'd', 8),
            ],
            from: ('d', 2),
            to: ('d', 7),
            expected: 0,
        },
        SeeCase {
            name: "queen takes defended queen (Qxd5, c6 pawn defends)",
            pieces: vec![
                (WK, 'a', 1),
                (BK, 'a', 8),
                (WQ, 'd', 1),
                (BQ, 'd', 5),
                (BP, 'c', 6),
            ],
            from: ('d', 1),
            to: ('d', 5),
            expected: 0,
        },
    ];
    for case in &cases {
        let bb = board(&case.pieces);
        let got = see(
            &bb,
            sq(file(case.from.0), case.from.1),
            sq(file(case.to.0), case.to.1),
        );
        println!("{}: see={got} expected={}", case.name, case.expected);
        assert_eq!(got, case.expected, "{}", case.name);
    }
}

#[test]
fn see_multi_capture_sequences() {
    let knight_defended_pawn = board(&[
        (WK, 'e', 1),
        (BK, 'e', 8),
        (WQ, 'd', 1),
        (BP, 'd', 5),
        (BN, 'e', 7),
    ]);
    let got = see(&knight_defended_pawn, sq(file('d'), 1), sq(file('d'), 5));
    println!("queen takes pawn, knight recaptures: see={got} expected=-850");
    assert_eq!(got, -850);

    let exchange_chain = board(&[
        (WK, 'e', 1),
        (BK, 'e', 8),
        (WP, 'd', 4),
        (WP, 'f', 4),
        (BQ, 'e', 5),
        (BP, 'd', 6),
    ]);
    let got = see(&exchange_chain, sq(file('d'), 4), sq(file('e'), 5));
    println!("pawn takes queen, pawn recaptures, pawn recaptures: see={got} expected=950");
    assert_eq!(got, 950);
}
