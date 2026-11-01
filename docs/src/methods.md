# Methods

Every method is a configurator struct passed to `pricer`, in the place of the `MonteCarloConfiguration` of
FinancialMonteCarlo.jl. All of them price every payoff and model of the package.

| Method | Constructor | How it works |
|:--|:--|:--|
| [`LewisMethod`](@ref) | `LewisMethod(A, N)` | Direct quadrature of the Lewis formula on `N` nodes in `[-A, A]` |
| [`CosMethod`](@ref) | `CosMethod(N)` | Cosine expansion of the put with `N` terms, the truncation range is computed from the model |
| [`CarrMadanLewisMethod`](@ref) | `CarrMadanLewisMethod(A, Npow)` | One FFT of the Lewis integrand on `2^Npow` nodes, then a cubic spline on the strikes |
| [`CarrMadanMethod`](@ref) | `CarrMadanMethod(A, Npow, α = 0.75)` | One FFT of the damped call transform on `2^Npow` nodes, then a cubic spline on the strikes |
| [`FinancialFFT.DensityInverter`](@ref) | `FinancialFFT.DensityInverter(N, xmax = NaN)` | FFT of the characteristic function on `2^N` points, then integration of the payoff against the density |

All of them agree with the closed formula of Black and Scholes:

```@example methods
using FinancialFFT, FinancialMonteCarlo, FinancialToolbox

S0, K, r, d, T, σ = 100.0, 105.0, 0.02, 0.01, 1.1, 0.2
model = BlackScholesProcess(σ, Underlying(S0, d))
zero_rate = ZeroRate(r)
option = EuropeanOption(T, K)
exact = blsprice(S0, K, r, T, σ, d)

methods = [LewisMethod(600.0, 20000), CosMethod(2^12), CarrMadanLewisMethod(600.0, 16), CarrMadanMethod(400.0, 18), FinancialFFT.DensityInverter(18)]
[nameof(typeof(method)) => abs(pricer(model, zero_rate, method, option) - exact) for method in methods]
```

## Choosing a method

- `LewisMethod` and `CosMethod` are the most accurate, and the only ones whose Zygote gradients and TaylorSeries results can both be relied upon, see [Automatic differentiation](@ref).
- `CosMethod` has no range parameter: its range comes from the tails of the model and covers the forward of every strike. A smile shares one range, so all of its strikes reuse the same coefficients, at the price of a larger `N` when the strikes are many standard deviations apart. It expands the put and recovers the call by parity, so calls stay accurate on models with a heavy right tail.
- `CarrMadanLewisMethod` and `CarrMadanMethod` run one FFT for all strikes, so their cost does not grow with the number of strikes; prices between the nodes of the strike grid come from the spline, which is what limits their accuracy.
- `CarrMadanMethod` needs `A / 2^Npow` to be small with respect to the damping, which is capped by the exponential moments of the model. It ignores the execution mode and runs on the CPU, see [GPU](@ref).
- `DensityInverter` needs `xmax` for every model but Black and Scholes. A smile shares one grid, wide enough for all of its strikes, and one characteristic exponent, so with the default `xmax` its prices differ from the prices of the single options by the discretization error.

```@example methods
model = MertonProcess(σ, 5.0, 0.03, 0.02, Underlying(S0, d))
pricer(model, zero_rate, FinancialFFT.DensityInverter(18, 8.0), option) - pricer(model, zero_rate, LewisMethod(600.0, 20000), option)
```
