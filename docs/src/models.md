# Models and curves

## Supported processes

Every process is defined in FinancialMonteCarlo.jl and wraps an `Underlying(S0, d)` holding the spot
and the dividend yield.

| Process | Constructor |
|:--|:--|
| Black–Scholes | `BlackScholesProcess(σ, underlying)` |
| Merton jump diffusion | `MertonProcess(σ, λ, μ_jump, σ_jump, underlying)` |
| Kou double exponential | `KouProcess(σ, λ, p, λ₊, λ₋, underlying)` |
| Variance Gamma | `VarianceGammaProcess(σ, θ, κ, underlying)` |
| Normal Inverse Gaussian | `NormalInverseGaussianProcess(σ, θ, κ, underlying)` |
| Heston | `HestonProcess(σ, σ₀, λ, κ, ρ, θ, underlying)` |

## Curves

`ZeroRate(r)` builds a flat curve. Non-constant zero rates and dividends are supported: an implied curve
is built at time zero, able to return the right implied zero/dividend at a given time without carrying
the integral structure of the curve. Multi-currency is not supported.

## Adding a model

A method only needs the characteristic exponent of the process. FinancialFFT implements it as a real
function of an imaginary argument, `ψ(v) = characteristic_exponent_i(v * im, process)`, so that drift and
martingale corrections can be computed without complex arithmetic:

```julia
function FinancialFFT.characteristic_exponent_i(imv, process::MyProcess)
    # characteristic exponent per unit of time, evaluated at imv
end
```

The exponent is per unit of time (methods multiply it by `T`) and must not contain any drift: the methods add
the drift implied by the zero and dividend curves and the martingale correction `characteristic_exponent_i(1, process)`.
A vectorized variant `characteristic_exponent_vi` is defined next to each exponent in `src/methods/charexp.jl`. The function is broadcast over arrays and
differentiated by every AD back-end, so it must be written generically, without type restrictions on `imv`.

Methods call `characteristic_exponent_i(imv, process, T)`, the log of ``E[\exp(imv\, X_T)]``, which defaults to the
exponent above times `T`. A process that is not Lévy, such as Heston, has no exponent per unit of time and defines this
three argument method (and `characteristic_exponent_vi`) directly. A process with semi-heavy tails also needs
`exponential_moment_bound(process, mul[, T])`, which limits the truncation range of `CosMethod`: pass `T` when the bound
depends on the maturity, as the moment explosion of Heston does.
