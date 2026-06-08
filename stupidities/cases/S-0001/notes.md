# S-0001: Non-defensive rook moves under mating attack

Ember reached the recorded FEN and chose passive rook moves while Black had a
forcing attack on the white king. The clearest exact-depth reproduction is:

- old behavior at depth 6: `b2b1`
- fixed behavior at depth 6: `b3c4`
- old behavior at depth 8 and 10: `b2a2`
- fixed behavior at depth 8 and 10: `b3d5`

The bad rook moves allow `...Rc5-h5`, followed by captures on f3 and a mating
queen invasion. The color-swapped mirror reproduced the same class of failure:
the old depth-8 behavior chose `b7a7`, while the fixed behavior chooses `b6d4`.

Root cause:

- Correction history was updated inside recursive search.
  That makes sibling evaluation order-dependent during a single search.
- Late move pruning removed quiet defensive resources in the mirrored variant.
- Null move pruning hid the shallow defensive move in the original position even
  after correction history and LMP were disabled.

Fix:

- Keep correction history enabled, but update it only once after a completed
  root search instead of mutating it inside recursive negamax.
- Keep LMP enabled, but skip it while either king has substantial local attack
  pressure.
- Keep null move pruning enabled, but skip it under the same tactical
  king-pressure condition.
