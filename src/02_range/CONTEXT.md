# Stage 02: truncation range

The range `[a, b]` of COS and of the densities from a Chernoff bound, and the root finder behind it.

## Inputs

| Load | Why |
|:--|:--|
| `extrema_computation_opt.jl` | `compute_chernorff_limits`, `chernoff_optimum`, `chernoff_g`, `TailExponent`, `brent_root`, `BoundedCoordinate`/`UnboundedCoordinate`, `exponential_moment_bound`, `heston_explosion_rate` |
| `references/range.md` | the bound, the bracket, Brent's coordinates, tolerances, the moment bounds per model, open points of #38 |
| `references/autodiff.md` | the range is a constant for AD (`ignore_derivatives_mod`) |

## Process

- Keep `docs/src/methods.md` ("Truncation range", figure `chernoff`) in step with this file.
- Never evaluate the exponent beyond `B (1 - sqrt(eps))`; a new process with semi-heavy tails needs an
  `exponential_moment_bound` method.
- Results stay type stable in Float32 (no `Inf` returns that make a `Union`).
- Enzyme marks `chernoff_optimum` and `exponential_moment_bound` inactive: keep them named functions.

## Outputs

`[a, b]` per process and maturity, reused by `fft_parameters.jl` (`AliasingExponent`). Checked by
`test/extrema_computation_test.jl`.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `run_tests()` with `test_args = ["extrema_computation"]` | pass |
