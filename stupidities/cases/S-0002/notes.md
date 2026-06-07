# S-0002: Futility pruning in a sparse promotion endgame

Ember reached a rook/pawn endgame where Black must keep checking or block the
promotion/mating net. The old search chose `e1b1` at depths 6 and 8. Stockfish
depth 16 shows that this allows a forced mate after `h3h4`.

The fixed behavior chooses `e1e4` at depths 6, 8, and 10. The horizontal mirror
reproduces the same old failure as `d1g1`; the fixed build chooses `d1d4`.

Root cause:

- The futility cutoff ran in sparse promotion endgames.
- Static evaluation plus a small margin is not a safe bound there because quiet
  promotion and mating resources can decide the position.
- Disabling futility alone removed the bad move at depths 6 and 8. Other
  individual pruning toggles did not remove it at both depths.

Fix:

- Keep futility pruning enabled.
- Skip futility pruning in promotion-race and sparse-endgame nodes.
