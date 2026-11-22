# Status

_What is in flight. Layer 4: volatile, rewritten freely; GitHub issues are the source of truth for tasks, this file is
the index an agent reads first._ Updated: 2026-11-22

## Current focus

0.7.0 released 2026-11-22 (localregistry, tag `v0.7.0`; #95, #72, #89, #93, #51, #40; breaking #89, #93, `NEWS.md`);
VibratoMonteCarlo and FinancialCalibrator take it. On `master` since, for the next release (`NEWS.md`, Unreleased): #74,
#78, #100, Mooncake on the GPU (#63).

## Active work

| Item | Branch | Issue | Next step |
|:--|:--|:--|:--|
| — | — | — | — |

## Known issues

| Issue | Symptom | Workaround |
|:--|:--|:--|
| #63 | Mooncake on the GPU: Lewis, COS, `CosDensityMethod` work (Mooncake 0.5.72); Carr-Madan and `FFTDensityInverter` do not | `Broken` in `mooncake_gpu` |
| EnzymeAD/Enzyme.jl#4108 | Enzyme silently wrong on Float32 Heston (complex `sqrt`/`inv`) | use Float64 |
| #96 | COS on Heston: Float32 2.2x slower than Float64 on the host | — |
| #88 | density pricers and Carr-Madan stop at 1e-8 to 1e-10 | larger `M` / `A` |
| #99 | Lewis, COS and the other methods sit on a Float64 floor of about `eps(S0)` (the subtraction from `S0 e^{-dT}`) | the convergence figure draws it (#40); control variate open |
| #81 | test suite 30-38 minutes on CI (~50 locally) | select suites with `test_args` |

## Enhancements and technical debt

- #98 `@ignore_derivatives` closures on the pricing paths (Zygote).
- #92 surfaces in one call; #91 barriers for Lévy models; #90 Heston from an arbitrary starting variance; #82
  LogNormalMixture exponent and ShiftedProcess.
- #64 differentiate the COS range; #48 Lewis and COS from a tolerance; #38 point 1 (`value_model`).
- #43 complex-to-real FFT for Carr-Madan; #42
  complex arrays on the GPU.
- #49 docs on the Dual Spectral Theorem.

## Working notes

Notes of an active initiative live next to this file as `state/YYYY-MM-DD-<topic>.md`. When the initiative ends, move
what stays true into `references/` or `references/decisions.md` and delete the note.
