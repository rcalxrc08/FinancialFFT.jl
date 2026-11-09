# Automatic differentiation

Prices are generic functions of the inputs of the model, the curve, and the payoff, so they can be differentiated by the
following back-ends, loaded with the corresponding package:

| Method | DualNumbers | HyperDualNumbers | TaylorSeries | ForwardDiff | ReverseDiff | Zygote | Mooncake | Enzyme |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `LewisMethod` | yes | yes | yes | yes | yes | yes | yes | yes |
| `CosMethod` | yes | yes | yes | yes | yes | yes | yes | yes |
| `CarrMadanLewisMethod` | yes | yes | yes | yes | yes | yes | yes | yes |
| `CarrMadanMethod` | yes | yes | yes | yes | yes | yes | yes | yes |
| `FFTDensityInverter` | yes | yes | yes | yes | yes | yes | yes | yes |
| `CosDensityMethod` | yes | yes | yes | yes | yes | yes | yes | yes |

The test suite checks each entry against the closed formulas of Black and Scholes, for European and binary options and
their smiles, with all of the inputs `S0`, `K`, `r`, `T`, `σ`, `d`. Gradients can be taken through the Heston model too.
TaylorSeries covers both `Taylor1` and `TaylorN`, which gives the mixed derivatives of several inputs at once:

```@example ad
using FinancialFFT, FinancialMonteCarlo, ForwardDiff

price(x) = pricer(BlackScholesProcess(x[3], Underlying(x[1], x[4])), ZeroRate(x[2]), LewisMethod(600.0, 20000), EuropeanOption(1.0, 100.0))
ForwardDiff.gradient(price, [100.0, 0.02, 0.2, 0.01])
```

`TaylorN` works in the same way; its variables are declared with `variables!`, or `set_variables` before TaylorSeries 0.23:

```julia
using TaylorSeries

δS, δσ = TaylorSeries.variables!("δS δσ"; order = 2)
pricer(BlackScholesProcess(0.2 + δσ, Underlying(100.0 + δS, 0.01)), ZeroRate(0.02), LewisMethod(600.0, 20000), EuropeanOption(1.0, 100.0))
```

Densities are differentiated in the same way, through `log(f(y))` (see [Densities](methods.md#Densities)):

| Density | DualNumbers | HyperDualNumbers | TaylorSeries | ForwardDiff | ReverseDiff | Zygote | Mooncake | Enzyme |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `FFTDensityInverter` | yes | yes | yes | yes | yes | yes | yes | yes |
| `CosDensityMethod` | yes | yes | yes | yes | yes | yes | yes | yes |

## Choosing a back-end

All the back-ends give the same derivatives, at very different costs. A forward-mode back-end (DualNumbers,
HyperDualNumbers, TaylorSeries, ForwardDiff) carries the derivatives along with the values through the characteristic
function, the sums and the FFTs: its cost grows with the number of inputs and does not depend on the number of
outputs. A reverse-mode back-end (ReverseDiff, Zygote, Mooncake) records or transforms the pricer and runs it
backwards: its cost does not depend on the number of inputs, but each output needs its own backward pass. The tables
give the time of each derivative as a multiple of the time of the price it differentiates (of one smile for the tasks
on smiles), measured with [`benchmark/ad.jl`](https://github.com/rcalxrc08/FinancialFFT.jl/blob/master/benchmark/ad.jl)
and [`benchmark/ad_gpu.jl`](https://github.com/rcalxrc08/FinancialFFT.jl/blob/master/benchmark/ad_gpu.jl) on an Intel
Core i7-8550U and an NVIDIA GeForce MX150 (Julia 1.13); the ratios carry over to other machines better than the times.

The gradient of a Black–Scholes price in its three inputs ``(S_0, r, \sigma)``, for every method:

| Method | price | DualNumbers | HyperDualNumbers | TaylorN | ForwardDiff | ReverseDiff | ReverseDiff, compiled tape | Zygote | Mooncake | Enzyme, forward | Enzyme, reverse |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `LewisMethod(600, 20000)` | 360 μs | 6.1 | 6.6 | 120 | **1.9** | 240 | 39 | 5.6 | 8.7 | 2.3 | 3.7 |
| `CosMethod(2^12)` | 401 μs | 3.5 | 4.4 | 47 | **1.4** | 110 | 15 | 6.3 | 8.0 | 2.3 | 3.5 |
| `CarrMadanLewisMethod(600, 16)` | 5.4 ms | 8.0 | 18 | 46 | **4.8** | 310 | 24 | 10 | 14 | 3.3 | 3.2 |
| `CarrMadanMethod(400, 18)` | 29 ms | 7.7 | 19 | 60 | 3.6 | 290 | 28 | 7.8 | 6.5 | 3.0 | **2.6** |
| `CosDensityMethod()` | 441 μs | 7.5 | 8.2 | 200 | 3.3 | 870 | 95 | 5.0 | 7.5 | **1.9** | 2.9 |
| `FFTDensityInverter()` | 752 μs | 5.6 | 9.7 | 120 | **2.7** | 130 | 23 | 4.7 | 7.3 | 2.4 | 2.6 |

Heston with `CosMethod(2^10)`, whose inputs are ``(S_0, r, \sigma, \sigma_0, \kappa, \rho, \theta)``, by task (one
price takes 279 μs, a smile of 101 strikes 714 μs):

| Task | DualNumbers | HyperDualNumbers | TaylorN | ForwardDiff | ReverseDiff | ReverseDiff, compiled tape | Zygote | Mooncake | Enzyme, forward | Enzyme, reverse |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gradient of a price, 7 inputs | 25 | 52 | 130 | 3.3 | 530 | 79 | 18 | 24 | **3.0** | 3.7 |
| gradient of the sum of a smile of 101 strikes, 7 inputs | 33 | 61 | 1500 | 8.2 | 2700 | 270 | 22 | 25 | 6.6 | **3.5** |
| Jacobian of a smile, 101 × 7 | 32 |  | 2500 | 9.2 | 33000 |  | 6000 |  | **7.3** | 370 |
| gradient of the sum of a smile in its 101 strikes | 360 | 490 |  | 49 | 2000 | 210 | 27 | 20 | 150 | **3.4** |
| Hessian of a price, 7 × 7 |  | 290 | 4200 | 90 | fails¹ |  | **77**² |  |  |  |

¹ `ReverseDiff.hessian` nests ReverseDiff in itself, and ReverseDiff has no arithmetic between nested tracked numbers.
² `Zygote.hessian` is forward over reverse: ForwardDiff through the reverse pass of Zygote.

On the GPU (`CudaMode`), where only ForwardDiff and Zygote are supported, the same Heston tasks (a price takes 1.4 ms,
a smile 1.8 ms, mostly in kernel launches):

| Task (GPU) | ForwardDiff | Zygote |
|:--|--:|--:|
| gradient of a price, 7 inputs | **1.4** | 13 |
| gradient of the sum of a smile of 101 strikes, 7 inputs | **1.3** | 7.2 |
| Jacobian of a smile, 101 × 7 | **1.1** | 580 |

The tables leave out the first call, which compiles the derivative. In a new session, the first gradient of a
`CosMethod` price takes, including the compilation of the price itself (4.1 s): DualNumbers 5.0 s, ForwardDiff 5.2 s,
HyperDualNumbers 6.3 s, ReverseDiff 9.3 s, TaylorN 25 s, Zygote 55 s, Mooncake 116 s (with the preparation of its
cache, which the tables leave out too), Enzyme 63 s in forward and 65 s in reverse mode. The Enzyme columns were measured
in a later session, when the price was about 20% faster: their ratios are to the price of that session.

Which back-end to use:

- **Greeks, and a gradient in a few parameters** (a calibration loss, the scores of a density): ForwardDiff. It is the
  fastest in most rows with a few inputs, at 1.4 to 5 prices, and compiles in about the time of the price. Enzyme
  is as fast or faster on the costlier methods and on smiles (2 to 4 prices in either mode), at the cost of a minute
  of compilation at the first call, and on the CPU only. Its
  advantage grows on the GPU, where the dual numbers ride in the kernels of the price: 1.1 to 1.4 prices.
- **A Jacobian with more outputs than inputs** (a smile in the parameters of the model, the step of a
  Levenberg–Marquardt calibration, see [Calibration](calibration.md)): ForwardDiff, which gets every output in one
  pass. A reverse-mode back-end needs one backward pass per strike.
- **A gradient in many inputs** (every strike of a smile, the points of a curve): Enzyme in reverse mode, whose
  backward pass costs about 3 to 4 times the function whatever the number of inputs (3.4 smiles for the 101 strikes),
  then Mooncake or Zygote, about 20 to 30 smiles here; ForwardDiff, which carries the
  inputs in chunks, already takes twice as long with 101 of them, and falls further behind as they grow. Zygote is the
  one that also runs on the GPU. Both compile for about a minute or two at the first call, which pays off over many
  gradients only.
- **Second derivatives:** `ForwardDiff.hessian`, or `Zygote.hessian` (forward over reverse) when the inputs are many.
  For one second derivative only (a gamma, one cross Greek), HyperDualNumbers in one pass of the pair of directions.
- **One directional derivative** (one Greek, a derivative along a direction): DualNumbers, one pass of a single dual
  number, cheaper than a full gradient.
- **Derivatives of high order:** TaylorSeries, `Taylor1` for one input to any order, `TaylorN` for all the mixed
  derivatives of several inputs in one pass. Its numbers are heap-allocated polynomials, so for first and
  second derivatives it is tens of times slower than ForwardDiff: use it for the orders the others do not reach.
- **ReverseDiff** is the slowest: every operation on a tracked number is recorded on a tape at each call. A compiled
  tape is 6 to 13 times faster, but it replays the operations recorded at one point, with every quantity that the
  package keeps constant for differentiation frozen at that point: the truncation range of COS and of the densities,
  and the grid of the Carr-Madan methods chosen from a tolerance. Reuse a compiled tape only near the point where it
  was recorded.

## Known limitations

- ReverseDiff.jl requires version 1.18 or later.
- On the GPU only ForwardDiff.jl and Zygote.jl are supported, see [GPU](@ref). Mooncake.jl differentiates the binary
  Lewis option there, and no other kernel: its CUDA rules do not take broadcasts that mix real and complex arrays, nor
  the `extrema` of a `CuArray`
  ([Mooncake.jl#1450](https://github.com/chalk-lab/Mooncake.jl/issues/1450),
  [#63](https://github.com/rcalxrc08/FinancialFFT.jl/issues/63)).
- Enzyme.jl differentiates every method and density on the CPU, in forward and reverse mode, through an extension
  on EnzymeCore that marks the constants of the methods inactive and gives the rules of the FFTs
  ([#62](https://github.com/rcalxrc08/FinancialFFT.jl/issues/62)). Pass the function to differentiate as `Const(f)`,
  with its inputs indexed, not splatted (`x -> f(x...)` makes Enzyme look for a shadow of `f`). On the GPU it does
  not work yet: reverse mode fails on broadcasts over `CuArray`s
  ([Enzyme.jl#3451](https://github.com/EnzymeAD/Enzyme.jl/issues/3451)), and forward mode needs runtime activity
  even with constant arrays ([Enzyme.jl#4017](https://github.com/EnzymeAD/Enzyme.jl/issues/4017)) and has no derivative of the CUDA `sincos` intrinsic of the complex exponential, which
  every method uses ([Enzyme#2552](https://github.com/EnzymeAD/Enzyme/issues/2552),
  [Enzyme#3307](https://github.com/EnzymeAD/Enzyme/issues/3307)).
