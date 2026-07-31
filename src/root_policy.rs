use crate::board::{move_from, move_promotion, move_to, BoardState, Move, EMPTY_SQ};
use std::io::Cursor;
use std::sync::{Arc, OnceLock};
use tract_onnx::prelude::*;

const MODEL_BYTES: &[u8] = include_bytes!("root_policy.onnx");
const FEATURES: usize = 16;
const POLICY_PLANES: usize = 73;
const POLICY_SIZE: usize = 64 * POLICY_PLANES;

type RunnableRootPolicy = Arc<TypedRunnableModel>;

pub struct RootPolicy {
    plan: RunnableRootPolicy,
}

impl RootPolicy {
    fn load() -> Result<Self, String> {
        let mut cursor = Cursor::new(MODEL_BYTES);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|error| format!("read embedded root policy: {error}"))?
            .with_input_fact(0, f32::fact([1, 64, FEATURES]).into())
            .map_err(|error| format!("set root policy input fact: {error}"))?
            .into_optimized()
            .map_err(|error| format!("optimize root policy: {error}"))?
            .into_runnable()
            .map_err(|error| format!("prepare root policy: {error}"))?;
        Ok(Self { plan: model })
    }

    pub fn score_moves(&self, st: &BoardState, moves: &[Move]) -> Result<Vec<f32>, String> {
        if moves.is_empty() {
            return Ok(Vec::new());
        }

        let input = Tensor::from_shape(&[1, 64, FEATURES], &root_features(st))
            .map_err(|error| format!("build root policy input tensor: {error}"))?;
        let outputs = self
            .plan
            .run(tvec!(input.into_tvalue()))
            .map_err(|error| format!("run root policy: {error}"))?;
        let logits = outputs
            .first()
            .ok_or_else(|| "root policy returned no outputs".to_string())?
            .to_plain_array_view::<f32>()
            .map_err(|error| format!("read root policy output: {error}"))?;
        let logits = logits
            .as_slice_memory_order()
            .ok_or_else(|| "root policy output is not contiguous".to_string())?;
        if logits.len() != POLICY_SIZE {
            return Err(format!(
                "root policy output has {} logits, expected {POLICY_SIZE}",
                logits.len()
            ));
        }

        moves
            .iter()
            .map(|&mv| {
                policy_index(mv)
                    .map(|idx| logits[idx])
                    .ok_or_else(|| format!("legal move cannot be mapped to policy index: {mv}"))
            })
            .collect()
    }
}

static EMBEDDED_ROOT_POLICY: OnceLock<Result<RootPolicy, String>> = OnceLock::new();

pub fn warm_up() -> Result<(), String> {
    embedded().map(|_| ())
}

pub fn score_moves(st: &BoardState, moves: &[Move]) -> Result<Vec<f32>, String> {
    embedded()?.score_moves(st, moves)
}

fn embedded() -> Result<&'static RootPolicy, String> {
    match EMBEDDED_ROOT_POLICY.get_or_init(RootPolicy::load) {
        Ok(policy) => Ok(policy),
        Err(error) => Err(error.clone()),
    }
}

fn root_features(st: &BoardState) -> Vec<f32> {
    let mut features = vec![0.0f32; 64 * FEATURES];
    let stm = if st.w { 0.0 } else { 1.0 };

    for policy_sq in 0..64 {
        let ember_sq = policy_to_ember_square(policy_sq);
        let base = policy_sq * FEATURES;
        let pi = st.mailbox[ember_sq];
        let piece_code = if pi == EMPTY_SQ {
            0usize
        } else {
            usize::from(pi) + 1
        };
        features[base + piece_code] = 1.0;
        features[base + 13] = stm;
        features[base + 14] = (policy_sq % 8) as f32 / 7.0;
        features[base + 15] = (policy_sq / 8) as f32 / 7.0;
    }

    features
}

fn policy_index(mv: Move) -> Option<usize> {
    let from = ember_to_policy_square(move_from(mv));
    let to = ember_to_policy_square(move_to(mv));
    let promotion = match move_promotion(mv).to_ascii_uppercase() {
        0 => 0,
        b'N' => 1,
        b'B' => 2,
        b'R' => 3,
        b'Q' => 4,
        _ => return None,
    };
    let plane = move_plane(from, to, promotion)?;
    Some(from * POLICY_PLANES + plane)
}

fn ember_to_policy_square(square: usize) -> usize {
    let row = square / 8;
    let col = square % 8;
    col + (7 - row) * 8
}

fn policy_to_ember_square(square: usize) -> usize {
    let file = square % 8;
    let rank_from_white = square / 8;
    (7 - rank_from_white) * 8 + file
}

fn move_plane(from: usize, to: usize, promotion: u8) -> Option<usize> {
    let fx = (from % 8) as i8;
    let fy = (from / 8) as i8;
    let tx = (to % 8) as i8;
    let ty = (to / 8) as i8;
    let dx = tx - fx;
    let dy = ty - fy;

    if promotion != 0 && promotion != 4 {
        if !(-1..=1).contains(&dx) || dy == 0 || dy.abs() != 1 {
            return None;
        }
        let direction = (dx + 1) as usize;
        let piece = match promotion {
            1 => 0,
            2 => 1,
            3 => 2,
            _ => return None,
        };
        return Some(64 + direction * 3 + piece);
    }

    if let Some(knight_plane) = knight_plane(dx, dy) {
        return Some(56 + knight_plane);
    }

    let (direction, distance) = ray_direction(dx, dy)?;
    Some(direction * 7 + distance - 1)
}

fn knight_plane(dx: i8, dy: i8) -> Option<usize> {
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
    KNIGHTS.iter().position(|&(kx, ky)| kx == dx && ky == dy)
}

fn ray_direction(dx: i8, dy: i8) -> Option<(usize, usize)> {
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
                return Some((idx, dist as usize));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::encode_move;

    #[test]
    fn policy_square_mapping_round_trips() {
        for square in 0..64 {
            assert_eq!(
                ember_to_policy_square(policy_to_ember_square(square)),
                square
            );
            assert_eq!(
                policy_to_ember_square(ember_to_policy_square(square)),
                square
            );
        }
    }

    #[test]
    fn policy_index_matches_extractor_layout_for_basic_moves() {
        let e2e4 = encode_move(6, 4, 4, 4, 0);
        let g1f3 = encode_move(7, 6, 5, 5, 0);
        let e7e8q = encode_move(1, 4, 0, 4, b'Q');
        let e7f8n = encode_move(1, 4, 0, 5, b'N');

        assert_eq!(policy_index(e2e4), Some(12 * 73 + 1));
        assert_eq!(policy_index(g1f3), Some(6 * 73 + 56 + 7));
        assert_eq!(policy_index(e7e8q), Some(52 * 73));
        assert_eq!(policy_index(e7f8n), Some(52 * 73 + 70));
    }

    #[test]
    fn features_use_policy_square_order() {
        let mut st = BoardState::empty();
        st.mailbox[policy_to_ember_square(0)] = 0;
        st.mailbox[policy_to_ember_square(63)] = 11;
        st.w = false;

        let features = root_features(&st);
        assert_eq!(features[1], 1.0, "a1 should contain a white pawn");
        assert_eq!(
            features[13], 1.0,
            "black side-to-move feature should be set"
        );

        let h8 = 63 * FEATURES;
        assert_eq!(features[h8 + 12], 1.0, "h8 should contain a black king");
        assert_eq!(features[h8 + 14], 1.0);
        assert_eq!(features[h8 + 15], 1.0);
    }
}
