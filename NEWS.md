# FinancialFFT.jl release notes

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
