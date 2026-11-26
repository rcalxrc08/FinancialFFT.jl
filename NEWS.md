# FinancialFFT.jl release notes

## Unreleased

### Breaking

- `LewisMethod` holds its parameters in a `grid` field (`QuadratureGrid(A, N)` or `QuadratureTolerance(ε, N_max)`), as
  the Carr-Madan methods: `method.A` and `method.N` are now `method.grid.A` and `method.grid.N`. The constructors
  `LewisMethod(A, N; mode)` are unchanged.

### New

- `LewisMethod(ε; N_max = 2^20 + 1)` (#48): `A` and `N` chosen at pricing time from a price tolerance, for the model,
  the maturity and the strikes. `A` from the tail of the modulus of the characteristic function on the line
  `Im(u) = 1/2`, the step from the aliasing copies (closed form for vanilla options, a Chernoff bound for binaries).
  Errors 0.23-0.6 ε measured; a Black–Scholes smile at ε = 1e-9 takes 547 nodes instead of the 20000 of
  `LewisMethod(600, 20000)`. `FinancialFFT.quadrature_grid` returns the chosen `(A, N)` for repeated pricing.

### Other

- Docs, calibration page (#102): the Levenberg–Marquardt loop stops on the gradient (`norm(Jᵀe) < 1e-12`) instead of
  a damped step, and accepts a step that leaves the loss within its rounding if it halves the gradient; it reaches the
  stationary point in 6 steps instead of stopping 2e-8 from it after 18 rejected steps.
- `test/Project.toml`: compat entry for FinancialFFT itself.

## 0.8.1 (2026-11-25)

- 0.8.0 committed a `[sources]` path to a local FinancialMonteCarlo in `Project.toml`, which broke CI (installs from a
  registry were not affected); removed. The README, the docs figures and the docs text still built the models with an
  underlying; converted to FinancialMonteCarlo 0.10.

## 0.8.0 (2026-11-25)

### Breaking

- FinancialMonteCarlo 0.10 (FinancialMonteCarlo#72): the models hold only their parameters, and the underlying is an
  argument: `pricer(model, underlying, zero_rate, method, payoff)` and `density(model, underlying, zero_rate, method, T)`,
  with `underlying = Underlying(S0, d)`. `ZeroRate` takes a number or a curve (`ZeroRate(FinancialMonteCarlo.Curve(r, T))`,
  `ZeroRate(FinancialMonteCarlo.ImpliedCurve(r, T))`); `ZeroRateCurve` and `ImpliedZeroRate` are gone. The
  characteristic exponents and the truncation range take the model alone.

### Other

- Real-series pullback of `clenshaw_cos_sum` (CosDensity under Zygote), 4-14% faster (#74).
- `density_array` is required of a density type, no default (#78).
- `@muladd` in the remaining real multiply-adds where not slower (#100).
- Mooncake differentiates Lewis, COS and `CosDensityMethod` on the GPU with Mooncake 0.5.72 (#63).

## 0.7.0 (2026-11-22)

### Breaking

- Invalid arguments of the method constructors (`A`, `Npow`, `α`, `ε`, `Npow_max`, `N`, `M`) throw a `DomainError` with
  the offending value instead of an `AssertionError` (#89, as FinancialMonteCarlo#71): code that catches
  `AssertionError` must catch `DomainError`. The checks can no longer be disabled.
- The five-argument `pricer(model, zero_rate, method, payoff, mode)` and `density(model, zero_rate, method, T, mode)`,
  deprecated since 0.4, are removed (#93): the mode is part of the method, use
  `pricer(model, zero_rate, FinancialMonteCarlo.with_mode(method, mode), payoff)` (or the `mode` keyword of the
  constructor).
