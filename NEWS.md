# FinancialFFT.jl release notes

## 0.7.0 (unreleased)

### Breaking

- Invalid arguments of the method constructors (`A`, `Npow`, `α`, `ε`, `Npow_max`, `N`, `M`) throw a `DomainError` with
  the offending value instead of an `AssertionError` (#89, as FinancialMonteCarlo#71): code that catches
  `AssertionError` must catch `DomainError`. The checks can no longer be disabled.
