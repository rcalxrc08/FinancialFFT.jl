# FinancialFFT.jl release notes

## 0.7.0 (unreleased)

### Breaking

- Invalid arguments of the method constructors (`A`, `Npow`, `α`, `ε`, `Npow_max`, `N`, `M`) throw a `DomainError` with
  the offending value instead of an `AssertionError` (#89, as FinancialMonteCarlo#71): code that catches
  `AssertionError` must catch `DomainError`. The checks can no longer be disabled.
- The five-argument `pricer(model, zero_rate, method, payoff, mode)` and `density(model, zero_rate, method, T, mode)`,
  deprecated since 0.4, are removed (#93): the mode is part of the method, use
  `pricer(model, zero_rate, FinancialMonteCarlo.with_mode(method, mode), payoff)` (or the `mode` keyword of the
  constructor).
