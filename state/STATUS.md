# Status

_What is in flight. Layer 4: volatile, rewritten freely; GitHub issues are the source of truth for tasks, this file is
the index an agent reads first._ Updated: 2026-11-22

## Current focus

Six small issues on branch `easy-fixes`, one commit each, for review: #95, #72, #89, #93, #51, #40 (docs part). #93 and
#89 are breaking (0.7, `NEWS.md`). Last release 0.6.3 (Enzyme on COS without float ranges, Mooncake `sqrt_mod` for any real complex).

## Active work

| Item | Branch | Issue | Next step |
|:--|:--|:--|:--|
| Easy fixes | `easy-fixes` | #95 #72 #89 #93 #51 #40 | review, then release 0.7 |

## Known issues

| Issue | Symptom | Workaround |
|:--|:--|:--|
| #63, chalk-lab/Mooncake.jl#1450 | Mooncake on the GPU: 2 of 48 cases work | `Broken` in `mooncake_gpu` |
| EnzymeAD/Enzyme.jl#4108 | Enzyme silently wrong on Float32 Heston (complex `sqrt`/`inv`) | use Float64 |
| #96 | COS on Heston: Float32 2.2x slower than Float64 on the host | — |
| #88 | density pricers and Carr-Madan stop at 1e-8 to 1e-10 | larger `M` / `A` |
| #40 | Lewis error rises at the last convergence point (rounding floor) | — |
| #81 | test suite 30-38 minutes on CI (~50 locally) | select suites with `test_args` |

## Enhancements and technical debt

- #98 `@ignore_derivatives` closures on the pricing paths (Zygote).
- #92 surfaces in one call; #91 barriers for Lévy models; #90 Heston from an arbitrary starting variance; #82
  LogNormalMixture exponent and ShiftedProcess.
- #64 differentiate the COS range; #48 Lewis and COS from a tolerance; #38 point 1 (`value_model`).
- #74 Clenshaw pullback without imaginary terms; #100 the remaining real multiply-adds; #43 complex-to-real FFT for Carr-Madan; #42
  complex arrays on the GPU.
- #78 coverage gaps; #49 docs on the Dual Spectral Theorem.

## Working notes

Notes of an active initiative live next to this file as `state/YYYY-MM-DD-<topic>.md`. When the initiative ends, move
what stays true into `references/` or `references/decisions.md` and delete the note.
