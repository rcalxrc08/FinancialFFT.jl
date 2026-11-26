# Status

_What is in flight. Layer 4: volatile, rewritten freely; GitHub issues are the source of truth for tasks, this file is
the index an agent reads first._ Updated: 2026-11-26

## Current focus

0.8.0 (2026-11-25, breaking, `NEWS.md`): FinancialMonteCarlo 0.10, the underlying is an argument,
`pricer(model, underlying, zero_rate, method, payoff)` and `density(model, underlying, zero_rate, method, T)`
(FinancialMonteCarlo#72); also #74, #78, #100 and Mooncake on the GPU (#63). VibratoMonteCarlo 0.7 and FinancialCalibrator
0.3 take it. 0.8.1: the local `[sources]` path committed in 0.8.0 removed (CI), README and docs converted. Open: #101
(Kou Chernoff range with rare up jumps). #102 done on `master` (2026-11-26): the calibration page's Levenberg–Marquardt
stops on the gradient and accepts a step within the rounding of the loss that halves the gradient. 0.9.0 (2026-11-26,
breaking, `NEWS.md`): `LewisMethod(ε)` (#48, Lewis part; `grid` field); VibratoMonteCarlo and FinancialCalibrator
widen their compat. COS from a tolerance next, together with #101. 0.10.0 (2026-11-26, breaking):
trapezoidal weights for the Carr-Madan methods (#104), `CosMethod(ε)` (#48, fixes #101). #48 open for the densities.

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
