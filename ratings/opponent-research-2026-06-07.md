# Opponent Research: Mid-Strength Engines

Date: 2026-06-07

This note summarizes candidate opponents for Ember Elo measurement and
stupidity hunting. Ember's recent measured strength is roughly 2360-2380
on the current project scale, so the best next opponents are close to
that band or slightly stronger.

## Recommendation

Add Blunder 7.2.0 next.

Blunder is an open-source UCI engine written in Go. Its upstream README
lists version 7.2.0 at about 2395 estimated Elo and 2425 CCRL Blitz Elo,
which is close enough to Ember to produce useful results without being a
too-weak opponent. Blunder 6.1.0 is likely too weak for the next slot,
while Blunder 8.x is probably too strong for first-pass near-peer
measurement.

Source: https://github.com/deanmchris/blunder

## Candidate Notes

| Engine | Assessment |
| --- | --- |
| Blunder 7.2.0 | Best next add. Close to Ember, UCI, open source, and easy to build reproducibly with a Nix-provided Go toolchain. |
| Hedgehog 2.0 | Strong strength fit. CCRL 40/15 lists Hedgehog 2.0 around 2393, but source/package provenance is less convenient. Current Hedgehog 2.6 is much stronger, around 2651. |
| GNU Chess 6.3.0 | Best nixpkgs-only fallback. Nix provides `gnuchess`, and GNU lists 6.3.0 as current. Exact CCRL matching for this version is unclear, so use a wide rating prior. |
| Clovis 1.8 | CCRL 40/15 lists it around 2647. Useful later as a stronger punishment opponent, but not ideal as the first new near-peer add. |
| Ekagine | Author reports 2900+ with NNUE. Too strong for the immediate close-opponent slot. |
| Cesso | No clean reliable package/source/rating target was found. Skip for now. |

## Package Availability

The named candidates are not top-level nixpkgs chess engine packages in
the checked environment. The practical top-level chess engines available
from nixpkgs are `stockfish`, `gnuchess`, and `fairymax`.

For Blunder, use a pinned source build instead of relying on a nixpkgs
package. That gives a stable versioned opponent while keeping the build
reproducible.

## References

- Blunder upstream README and rating table:
  https://github.com/deanmchris/blunder
- CCRL 40/15 complete list:
  https://computerchess.org.uk/ccrl/4040/rating_list_all.html
- GNU Chess project page:
  https://www.gnu.org/software/chess/
