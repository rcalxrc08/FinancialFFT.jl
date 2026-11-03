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
| [`CosDensityMethod`](@ref) | `CosDensityMethod(N = 128, M = 2^10)` | Cosine expansion of the density with `N` terms, then Simpson's rule for the put on `M` intervals |

All of them agree with the closed formula of Black and Scholes:

```@example methods
using FinancialFFT, FinancialMonteCarlo, FinancialToolbox

S0, K, r, d, T, σ = 100.0, 105.0, 0.02, 0.01, 1.1, 0.2
model = BlackScholesProcess(σ, Underlying(S0, d))
zero_rate = ZeroRate(r)
option = EuropeanOption(T, K)
exact = blsprice(S0, K, r, T, σ, d)

methods = [LewisMethod(600.0, 20000), CosMethod(2^12), CarrMadanLewisMethod(600.0, 16), CarrMadanMethod(400.0, 18), FinancialFFT.DensityInverter(18), CosDensityMethod()]
[nameof(typeof(method)) => abs(pricer(model, zero_rate, method, option) - exact) for method in methods]
```

## Choosing a method

- `LewisMethod` and `CosMethod` are the most accurate, and with `DensityInverter` the only ones whose Zygote gradients can be relied upon, see [Automatic differentiation](@ref).
- `CosMethod` has no range parameter: its range comes from the tails of the model and covers the forward of every strike. A smile shares one range, so all of its strikes reuse the same coefficients, at the price of a larger `N` when the strikes are many standard deviations apart. It expands the put and recovers the call by parity, so calls stay accurate on models with a heavy right tail. On the CPU the strikes of a smile share the phases of the characteristic function, and each strike sums the series with a Clenshaw recurrence, at the cost of one `sincos` instead of one `cos` per term: in Black Scholes with `N = 2^12`, a smile of 100 strikes costs about 3.5 times a single option, rather than about 18. On the GPU the series stays one broadcast over terms and strikes.
- `LewisMethod` prices the strikes of a smile from one set of evaluations of the characteristic function: on the CPU each strike sums the quadrature with a Clenshaw recurrence (in Reinsch's form, which stays accurate at the money), at the cost of one `sincos` per strike instead of an `exp` and a `cos` per node, about 6 times faster for smiles of 100-500 strikes with `N = 20000`.
- `CarrMadanLewisMethod` and `CarrMadanMethod` run one FFT for all strikes, so their cost does not grow with the number of strikes; prices between the nodes of the strike grid come from the spline, which is what limits their accuracy.
- `CarrMadanMethod` needs `A / 2^Npow` to be small with respect to the damping, which is capped by the exponential moments of the model. It ignores the execution mode and runs on the CPU, see [GPU](@ref).
- `DensityInverter` needs `xmax` for every model but Black and Scholes. A smile shares one grid, wide enough for all of its strikes, and one characteristic exponent, so with the default `xmax` its prices differ from the prices of the single options by the discretization error.

```@example methods
model = MertonProcess(σ, 5.0, 0.03, 0.02, Underlying(S0, d))
pricer(model, zero_rate, FinancialFFT.DensityInverter(18, 8.0), option) - pricer(model, zero_rate, LewisMethod(600.0, 20000), option)
```

- `CosDensityMethod` integrates against its [`density`](@ref), whose range is the one of `CosMethod`, so it needs no range parameter either. It is slower than `CosMethod` (on the CPU the density at each Simpson node costs one Clenshaw recurrence), which integrates the payoff exactly against the cosine expansion: its purpose is to price with the density that other methods can use.

## Densities

[`density`](@ref) is the interface for the density of the log-return `log(S_T / S_0)`, for methods that need the density of a process without a closed formula, such as the likelihood ratio method or Vibrato. Like `pricer`, it dispatches on the method: `CosDensityMethod` expands it on cosines, `DensityInverter` inverts the characteristic function by an FFT and interpolates it with a cubic spline. The setup runs once, then the density is evaluated at any number of points; it does not depend on `S_0`, so the density of `log(S_T)` given `S_t` is `f(x - log(S_t))`. Scores come from automatic differentiation:

```@example methods
using ForwardDiff

f = density(model, zero_rate, CosDensityMethod(), T)
score(y) = ForwardDiff.derivative(s -> log(density(MertonProcess(s, 5.0, 0.03, 0.02, Underlying(S0, d)), zero_rate, CosDensityMethod(), T)(y)), σ)
f(0.1), score(0.1)
```

A density is zero outside its range `[f.a, f.b]`: the range of `CosMethod` for `CosDensityMethod`, beyond which the density is below `1e-14`, and the grid for `DensityInverter`. The COS density costs little to set up and about a microsecond per point; the FFT density costs an FFT on `2^N` points to set up, and a spline evaluation per point. Their errors are absolute: far in the tails, the relative error, which is the one of a score, grows.
