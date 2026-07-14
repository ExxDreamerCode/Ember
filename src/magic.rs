use std::sync::OnceLock;

struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

static TABLES: OnceLock<MagicTables> = OnceLock::new();
fn tables() -> &'static MagicTables {
    TABLES.get_or_init(MagicTables::init)
}

struct MagicTables {
    bishop: [Magic; 64],
    rook: [Magic; 64],
    bishop_attacks: Vec<u64>,
    rook_attacks: Vec<u64>,
}

#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let t = tables();
    let m = &t.bishop[sq];
    let idx = m.offset + (occ & m.mask).wrapping_mul(m.magic).wrapping_shr(m.shift) as usize;
    t.bishop_attacks[idx]
}

#[inline(always)]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let t = tables();
    let m = &t.rook[sq];
    let idx = m.offset + (occ & m.mask).wrapping_mul(m.magic).wrapping_shr(m.shift) as usize;
    t.rook_attacks[idx]
}

#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
    0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
    0x0000040404040400, 0x0000020202020200, 0x0000040102021400, 0x0000040400808000,
    0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
    0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
    0x0000800400a00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
    0xfb14bb07d2903020, 0x2d68a026481a2789, 0x1a740201f002e745, 0x0000404004010200,
    0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
    0x1fd00972f0e13c3e, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
    0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
    0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
    0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
    0x0000410410400000, 0x0000208208200000, 0x8cfc97efaec81f14, 0x0000000020880000,
    0x0000001002020000, 0x99a1414e5c5902b4, 0x784c3fa5d7bd2e3c, 0x913f3601049e38c5,
    0xaf43ffd19863fb28, 0x86c94bfb6e38d1ac, 0x4b11cc520d08981b, 0x76f537381e104413,
    0xd3123cf3d20a060c, 0xd1581e7160131c40, 0x0edafff355812a5c, 0xea7ffd02d3ac6446,
];

#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
    0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
    0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
    0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
    0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
    0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
    0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
    0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
    0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
    0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
    0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
    0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
    0x00fffcddfced714a, 0x007ffcddfced714a, 0x003fffcdffd88096, 0x0000040810002101,
    0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001fffaabfad1a2,
];

fn bishop_mask(sq: usize) -> u64 {
    let (r, c) = (sq / 8, sq % 8);
    let mut m = 0u64;
    for (dr, dc) in [(-1i32, -1i32), (-1, 1), (1, -1), (1, 1)] {
        let (mut rr, mut rc) = (r as i32 + dr, c as i32 + dc);
        while rr > 0 && rr < 7 && rc > 0 && rc < 7 {
            m |= 1u64 << (rr * 8 + rc);
            rr += dr;
            rc += dc;
        }
    }
    m
}

fn rook_mask(sq: usize) -> u64 {
    let (r, c) = (sq / 8, sq % 8);
    let mut m = 0u64;
    for rr in 1..7i32 {
        if rr != r as i32 {
            m |= 1u64 << (rr * 8 + c as i32);
        }
    }
    for rc in 1..7i32 {
        if rc != c as i32 {
            m |= 1u64 << (r as i32 * 8 + rc);
        }
    }
    m
}

fn slow_bishop(sq: usize, occ: u64) -> u64 {
    let (r, c) = (sq / 8, sq % 8);
    let mut att = 0u64;
    for (dr, dc) in [(-1i32, -1i32), (-1, 1), (1, -1), (1, 1)] {
        let (mut rr, mut rc) = (r as i32 + dr, c as i32 + dc);
        while (0..8).contains(&rr) && (0..8).contains(&rc) {
            let b = 1u64 << (rr * 8 + rc);
            att |= b;
            if occ & b != 0 {
                break;
            }
            rr += dr;
            rc += dc;
        }
    }
    att
}

fn slow_rook(sq: usize, occ: u64) -> u64 {
    let (r, c) = (sq / 8, sq % 8);
    let mut att = 0u64;
    for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let (mut rr, mut rc) = (r as i32 + dr, c as i32 + dc);
        while (0..8).contains(&rr) && (0..8).contains(&rc) {
            let b = 1u64 << (rr * 8 + rc);
            att |= b;
            if occ & b != 0 {
                break;
            }
            rr += dr;
            rc += dc;
        }
    }
    att
}

fn subsets(mask: u64) -> Vec<u64> {
    let mut v = Vec::new();
    let mut sub = 0u64;
    loop {
        v.push(sub);
        if sub == mask {
            break;
        }
        sub = sub.wrapping_sub(mask) & mask;
    }
    v
}

fn shift_for(mask: u64) -> u32 {
    64 - mask.count_ones()
}

impl MagicTables {
    fn init() -> Self {
        let mut bishop_attacks_vec = Vec::new();
        let mut offset = 0usize;

        let bishop_magics_arr = std::array::from_fn(|sq| {
            let mask = bishop_mask(sq);
            let shift = shift_for(mask);
            let magic = BISHOP_MAGICS[sq];
            let size = 1usize << mask.count_ones();

            let mut table = vec![0u64; size];
            for sub in subsets(mask) {
                let idx = sub.wrapping_mul(magic).wrapping_shr(shift) as usize;
                table[idx] = slow_bishop(sq, sub);
            }

            let entry = Magic {
                mask,
                magic,
                shift,
                offset,
            };
            bishop_attacks_vec.extend_from_slice(&table);
            offset += size;
            entry
        });

        let mut rook_attacks_vec = Vec::new();
        let mut offset = 0usize;

        let rook_magics_arr = std::array::from_fn(|sq| {
            let mask = rook_mask(sq);
            let shift = shift_for(mask);
            let magic = ROOK_MAGICS[sq];
            let size = 1usize << mask.count_ones();

            let mut table = vec![0u64; size];
            for sub in subsets(mask) {
                let idx = sub.wrapping_mul(magic).wrapping_shr(shift) as usize;
                table[idx] = slow_rook(sq, sub);
            }

            let entry = Magic {
                mask,
                magic,
                shift,
                offset,
            };
            rook_attacks_vec.extend_from_slice(&table);
            offset += size;
            entry
        });

        MagicTables {
            bishop: bishop_magics_arr,
            rook: rook_magics_arr,
            bishop_attacks: bishop_attacks_vec,
            rook_attacks: rook_attacks_vec,
        }
    }
}
