# Methods

Every method is a configurator struct passed to `pricer`, in the place of the `MonteCarloConfiguration` of
FinancialMonteCarlo.jl. All of them price every payoff and model of the package.

| Method | Constructor | How it works |
|:--|:--|:--|
| [`LewisMethod`](@ref) | `LewisMethod(A, N)` | Direct quadrature of the Lewis formula on `N` nodes in `[-A, A]` |
| [`CosMethod`](@ref) | `CosMethod(N)` | Cosine expansion of the put with `N` terms, the truncation range is computed from the model |
| [`CarrMadanLewisMethod`](@ref) | `CarrMadanLewisMethod(A, Npow)` | One FFT of the Lewis integrand on `2^Npow` nodes, then Lagrange interpolation on six nodes at the strikes |
| [`CarrMadanMethod`](@ref) | `CarrMadanMethod(A, Npow, α = 0.75)` | One FFT of the damped call transform on `2^Npow` nodes, then Lagrange interpolation on six nodes at the strikes |
| [`CosDensityMethod`](@ref) | `CosDensityMethod(N = 128, M = 2^10)` | Cosine expansion of the density with `N` terms, then Simpson's rule for the put on `M` intervals |
| [`FFTDensityInverter`](@ref) | `FFTDensityInverter(N = 14, M = 2^10)` | One complex-to-real FFT on `2^N` points for the density, from the characteristic function on `2^(N-1) + 1` frequencies, then Simpson's rule for the put on `M` intervals |

All of them agree with the closed formula of Black and Scholes:

```@example methods
using FinancialFFT, FinancialMonteCarlo, FinancialToolbox

S0, K, r, d, T, σ = 100.0, 105.0, 0.02, 0.01, 1.1, 0.2
model = BlackScholesProcess(σ, Underlying(S0, d))
zero_rate = ZeroRate(r)
option = EuropeanOption(T, K)
exact = blsprice(S0, K, r, T, σ, d)

methods = [LewisMethod(600.0, 20000), CosMethod(2^12), CarrMadanLewisMethod(600.0, 16), CarrMadanMethod(400.0, 18), CosDensityMethod(), FFTDensityInverter()]
[nameof(typeof(method)) => abs(pricer(model, zero_rate, method, option) - exact) for method in methods]
```

The error depends on how many times a method evaluates the characteristic exponent, which is most of its cost. COS
converges exponentially and reaches the rounding floor with a few dozen terms; Lewis does too, once its nodes cover
the decay of the integrand; the Carr-Madan methods stop at the error of the interpolation on the strike grid, and the two
density methods at the error of Simpson's rule on `M` intervals:

![Error against Black and Scholes as a function of the evaluations of the characteristic exponent](assets/figures/convergence-light.svg)![Error against Black and Scholes as a function of the evaluations of the characteristic exponent](assets/figures/convergence-dark.svg)

## Choosing a method

- `LewisMethod` and `CosMethod` are the most accurate. Every method is differentiable with every back-end, Zygote included, see [Automatic differentiation](@ref).
- `CosMethod` has no range parameter: its range comes from the tails of the model and covers the forward of every strike. A smile shares one range, so all of its strikes reuse the same coefficients, at the price of a larger `N` when the strikes are many standard deviations apart. It expands the put and recovers the call by parity, so calls stay accurate on models with a heavy right tail. On the CPU the strikes of a smile share the phases of the characteristic function, and each strike sums the series with a Clenshaw recurrence, at the cost of one `sincos` instead of one `cos` per term: in Black Scholes with `N = 2^12`, a smile of 100 strikes costs about 3.5 times a single option, rather than about 18. On the GPU the series stays one broadcast over terms and strikes.
- `LewisMethod` evaluates the characteristic function on the nodes in `[0, A]` only, since the integrand is even. It prices the strikes of a smile from one set of these evaluations: on the CPU each strike sums the quadrature with a Clenshaw recurrence (in Reinsch's form, which stays accurate at the money), at the cost of one `sincos` per strike instead of an `exp` and a `cos` per node, about 6 times faster for smiles of 100-500 strikes with `N = 20000`.
- `CarrMadanLewisMethod` and `CarrMadanMethod` run one FFT for all strikes, so their cost does not grow with the number of strikes; prices between the nodes of the strike grid come from Lagrange polynomials on six nodes, of order `h⁶` in the step `h = 2π / A` of the grid, which is what limits their accuracy. The interpolant is continuous but its derivatives in the strike jump at the nodes, by about 1e-8 for `CarrMadanLewisMethod(600, 16)`.
- `CarrMadanMethod` needs `A / 2^Npow` to be small with respect to the damping, which is capped by the exponential moments of the model. Both Carr-Madan methods can choose `A` and `Npow` from a price tolerance, see [below](#Carr-Madan-parameters-from-a-tolerance).
- `CosDensityMethod` and `FFTDensityInverter` price with their [`density`](@ref), the cosine expansion and the FFT inversion: they need no range parameter, and the strikes of a smile share the density, so a smile costs one density and one Simpson's rule per strike. Their error is the one of Simpson's rule on `M` intervals (about `1e-9` with the default `M = 2^10`), and they are slower than `CosMethod`, which integrates the payoff exactly against the cosine expansion: their purpose is to price with the densities that other estimators use.

```@example methods
model = MertonProcess(σ, 5.0, 0.03, 0.02, Underlying(S0, d))
pricer(model, zero_rate, FFTDensityInverter(), option) - pricer(model, zero_rate, LewisMethod(600.0, 20000), option)
```


## Carr-Madan parameters from a tolerance

`CarrMadanMethod(ε)` and `CarrMadanLewisMethod(ε)` choose `A` and `Npow` at pricing time, for the model and the strikes
of the payoff (an option or a smile), so that the error of the prices is below `ε`. They bound the three errors of the
FFT methods:

- **Interpolation and truncation:** the FFT samples the integral up to the frequency `A` on a strike grid of step
  `2π / A`, and the six-point Lagrange interpolation errs on a frequency `v` by at most `L₆ (2π v / A)⁶`. Weighting the
  modulus of the integrand with this error (with a cap, and with 1 beyond `A`) gives a bound decreasing in `A`, solved for `A`.
- **Aliasing:** the result is periodic in the log-strike. Simpson's weights also add copies at half the period, which
  is the half-width `b = π 2^Npow / A` of the strike grid, so `b` must reach the distance at which the damped price is below
  `ε`. It is a closed form on one side, and on the other, where the price can have a heavy tail, a Chernoff bound minimised
  as for the range of `CosMethod`.

`2^Npow` is the smallest power of two with both, and the spare nodes go to `A`. The bounds are model-free apart from the
characteristic function, and their error is typically a quarter to a half of `ε`:

```@example methods
model = BlackScholesProcess(σ, Underlying(S0, d))
smile = FinancialMonteCarlo.EuropeanOptionSmile(T, collect(50.0:5.0:200.0), fill(true, 31))
maximum(abs.(pricer(model, zero_rate, CarrMadanMethod(1e-8), smile) .- blsprice.(S0, smile.K, r, T, σ, d)))
```

The parameters are constants for differentiation. Choosing them costs about one price, at every price: when the same
model and strikes are priced many times (a calibration), compute `A` and `Npow` once with the tolerance and pass them to
`CarrMadanMethod(A, Npow)`.

## Floating-point types

Prices have the floating-point type of the model, the rates and the payoff: the grids, the truncation ranges and the parameters of the method (`A`, `α`, given in Float64 in the examples) are converted to it, so Float32 inputs give Float32 prices and BigFloat inputs BigFloat prices. FFTW transforms Float32 and Float64 only: in other types (BigFloat, DoubleFloats, …) the FFT methods use [FFTA.jl](https://github.com/dannys4/FFTA.jl), slower than FFTW but in their precision, and DualNumbers, HyperDualNumbers, TaylorSeries, ForwardDiff and Zygote differentiate them in these types too. The truncation range of the COS methods comes from a tail probability of `1e-14` in Float64, scaled by the machine epsilon in other types: in BigFloat the range is wider, so reaching its precision needs a larger `N` (and `M` for `CosDensityMethod`).

```julia
setprecision(BigFloat, 128) do
    model = BlackScholesProcess(BigFloat(0.2), Underlying(BigFloat(100), BigFloat(0.01)))
    pricer(model, ZeroRate(BigFloat(0.02)), CosMethod(2^10), EuropeanOption(BigFloat(1), BigFloat(100)))
end
```

## Densities

[`density`](@ref) returns the density of the log-return `log(S_T / S_0)`, for estimators that need it without a closed
formula, such as the likelihood ratio method or Vibrato. Like `pricer`, it dispatches on the method:

| Method | Density | Range | Cost |
|:--|:--|:--|:--|
| [`FFTDensityInverter`](@ref) | one complex-to-real FFT on `2^N` points (the characteristic function on `2^(N-1) + 1` frequencies), cubic interpolation between them | the range of `CosMethod`, from the model | an FFT to set up, about 30 ns a point |
| [`CosDensityMethod`](@ref) | cosine expansion on `N + 1` terms | the range of `CosMethod`, from the model | `N` exponents to set up, about 300 ns a point (`N = 128`) |

The setup runs once, then the density is evaluated at any number of points: `f(y)` at a point, on the CPU, and
[`FinancialFFT.density_values`](@ref)`(f, y)` at an array of points, on the GPU too. It does not depend on `S_0`, so the
density of `log(S_T)` given `S_t` is `f(x - log(S_t))`; outside its range `[f.a, f.b]` it is zero.

```@example methods
f = density(model, zero_rate, FFTDensityInverter(), T)
f(0.1), f.a, f.b
```

![Densities of the log-return of six models](assets/figures/densities-light.svg)![Densities of the log-return of six models](assets/figures/densities-dark.svg)

The density is generic in the number type of the model, so its scores, the weights of the likelihood ratio method,
come from automatic differentiation of `log(f(y))`; the range is a constant for differentiation:

```@example methods
using ForwardDiff

merton(s) = MertonProcess(s, 5.0, 0.03, 0.02, Underlying(S0, d))
score(y) = ForwardDiff.derivative(s -> log(density(merton(s), zero_rate, FFTDensityInverter(), T)(y)), σ)
score(0.1)
```

![Scores of the volatility](assets/figures/scores-light.svg)![Scores of the volatility](assets/figures/scores-dark.svg)

- `FFTDensityInverter` is the one to use for many evaluations, e.g. one per path of a simulation: its interpolation is a
  gather of four values. Its error is of order `dx⁴` (about `1e-13` with the default `N = 14` in Black and Scholes),
  and the grid narrows with the horizon, so short steps keep their resolution. Every AD back-end differentiates it, Zygote
  included, on the CPU and on the GPU.
- `CosDensityMethod` converges exponentially on smooth densities, and is the more accurate of the two there; it
  struggles where the density is not smooth, as Variance Gamma over a short step, whose density has a cusp at its mode
  (with 512 terms 3% of the maximum off at the cusp and 2e-3 away from it, against 3e-4 and below 1e-7 for
  `FFTDensityInverter(16)`).

All errors are absolute: far in the tails, where the density is close to the floor of the machine epsilon, the relative
error, which is the one of a score, grows.
