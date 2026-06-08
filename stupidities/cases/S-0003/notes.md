# S-0003: Selective reductions in sparse promotion races

Ember reached a sparse rook/pawn endgame against Blunder 7.2.0 and chose
`f4g4` at the traced depth 8. The move abandons the critical blockade while
Black has connected passed pawns on `a3` and `b2`.

The fixed behavior no longer repeats `f4g4` at depth 8:

- old base depth 8: `f4g4`
- fixed base depth 8: `f4e4`

The verifier also reproduced the same class in a color-swapped promotion race:

- old variant depth 6, 8, and 10: `f5g5`
- fixed variant depth 6, 8, and 10: `f5e5`

Root cause:

- Late move pruning hid the base position's quiet rook-defense move at the
  traced depth.
- Late move reductions hid the color-swapped variant's quiet king-blockade
  move at depths 6 and 8.
- The passed-pawn evaluator also mixed side-relative ranks with board rows when
  checking blockers, making advanced pawn endgames less reliable.

Fix:

- Keep LMP and LMR enabled in normal play.
- Skip LMP and LMR in promotion-race and sparse-endgame nodes, matching the
  existing futility-pruning safety class.
- Evaluate passed-pawn blockers using actual board rows for both sides.
