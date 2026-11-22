# Stage 05: pricing methods

_Dispatch, Carr-Madan, Carr-Madan-Lewis, the parameters from a tolerance, Lewis, COS, quadratures._ (Layer 3
reference.)

## Dispatch design

- Each method is a configurator struct subtyping `FinancialMonteCarlo.AbstractMethod` via `AbstractFFTMethod`
  (Carr-Madan, Carr-Madan-Lewis; use `fft`) or `AbstractIntegralMethod` (Lewis, COS; direct quadrature/cosine sums),
  `AbstractDensityMethod <: AbstractIntegralMethod` for the density methods (`src/01_hooks/abstracts.jl`).
- Pricing is `pricer(process, zero_rate, method, payoff)`, extending `FinancialMonteCarlo.pricer`, always with four
  arguments: the execution mode is a field of the method (FinancialMonteCarlo 0.6, #55), `mode` as a keyword of every
  public constructor (default `SerialMode()`), and the inner constructors take every field positionally, so that
  `FinancialMonteCarlo.with_mode(method, mode)` (`Accessors.@set`) rebuilds the method.
- The pricers and `density` read the mode with `FinancialMonteCarlo.execution_mode(method)` and move the payoff to the
  device of the mode with `FinancialMonteCarlo.to_mode` (a smile with host strikes runs in `CudaMode`, one with device
  strikes in `SerialMode`); the internal kernels keep a `mode` argument.
- Every method has one `pricer` for options and smiles (`FourierPayoff`): several strikes are priced through smiles,
  options with different maturities by broadcasting `pricer.`, and there is no `pricer` taking a vector of strikes or an
  array of options (#92 proposes surfaces in one call).
- The methods only call `integral` on the rate and on the dividend, each a number or a curve, which works on
  FinancialMonteCarlo 0.7-0.9 (0.9: curves are `SVector`s and two curves no longer multiply, FMC#79, #81).

## Quadratures and accuracy floors (#88)

Every quadrature has a comment naming the rule and its order: Simpson in Carr-Madan and Carr-Madan-Lewis (an even
number of nodes, the last panel not closed), Simpson on `M` intervals in the density pricers
(`density_integral_price`), the rectangle rule of Lewis, the trapezoidal rule behind the FFT density, the trapezoidal
rule in `log v` of `fft_parameters.jl`; COS has none. The floors of the convergence figure are not from the Fourier
quadrature: Simpson with the default `M = 2^10` limits the density pricers to ~5e-10 (O(M⁻⁴)), and the fixed
log-strike step `2π / A` the Carr-Madan methods (strike interpolation). Keep a comment on any new quadrature.
Quadrature weights come from PatternVectors.

## Carr-Madan (`carrmadan.jl`)

- Damped call transform, without intrinsic value: the undamped time value has a kink at the forward and decays like
  `1/v²`. The damping `α` is capped by `exponential_moment_bound`, and `A / 2^Npow` must be small with respect to it.
- The binary call is minus the strike derivative of the call: `α + iv` leaves the denominator, and the price is divided
  by `K`.
- Both Carr-Madan kernels transform `w .* exp.(ψ(…) .+ v_im .* c)` (#45): the factors that do not depend on the model
  (signs `(-1)^j`, Simpson weights, `dx/3`, the inverse of the denominator with one real division, `inv_denominator`)
  are floats even when the exponent is a dual, the drift is a phase in the exponent, and the real constant of the
  exponent (`-(1 + α) corr`, `correction / 2`) leaves the FFT for the scalar factor of the prices.
- Carr-Madan computes the weights in a loop of their own (`carr_madan_weights`, inside `@ignore_derivatives`), which
  measured faster. Measured on Merton against the previous commit: Carr-Madan 35% faster (price) and 19% (ForwardDiff).
- `FFTGrid(A, Npow)` and `FFTTolerance(ε, Npow_max)` (the `grid` field) are defined here.

## Carr-Madan-Lewis (`carrmadan_lewis.jl`)

Computes the weights in the broadcast, with the nodes `complex(one_half, x)` and a float `one_half`: a `Rational` added
to a complex number in a lazy argument of the exponent (`1 // 2 + x * im`) makes it 2.3 times slower (Lewis uses a float
`one_half` for the same reason). Keep rationals out of complex lazy arguments. The binary call is minus the strike
derivative of the Lewis formula: one kernel, with `lewis_denominator` and `carr_madan_lewis_call` selecting the payoff.

## Parameters from a tolerance (`fft_parameters.jl`, #41, suite `fft_parameters`)

- `CarrMadanMethod(ε; α, Npow_max)` and `CarrMadanLewisMethod(ε; Npow_max)` choose `A`, `Npow` at pricing time: the
  pricers call `fft_grid(method.grid, method, process, zero_rate, payoff)`, inside `@ignore_derivatives`, with `A_min`,
  `b_min` through `ignore_derivatives_mod` (ReverseDiff would track the Chernoff bound, Mooncake would differentiate
  `A`): the parameters are constants on every back-end (tested).
- Interpolation and truncation: `scale/π ∫ |ψ(v)| w(2πv/A) dv` with `w = min(L₆ θ⁶, 1 + Λ₆)` below `A` and 1 above (L₆,
  Λ₆ of the six-point stencil at t = 1/2), trapezoid in log v (PatternVectors weights 1, 2, …, 2, 1 times `du / 2`,
  #95: no grid of the `fft_parameters` suite moved), `≤ ε/2`, bisection for `A_min`; measured it binds the
  error, not the truncation, which this weight bounds too.
- Aliasing: Simpson's weights are the trapezoidal ones times `1 - (-1)^j/3`, so besides the Poisson copies at `2b` there
  are copies at `±b` (half period = half-width of the strike grid) with weight `-1/3`, almost independent of `k`
  (checked numerically); each side `≤ ε/4`, from closed forms (`G ≤ F0`, `≤ e^{-rT}`, `≤ K e^{-rT}`) or a Chernoff
  bound (`AliasingExponent`, minimised with `chernoff_optimum`, shared with the COS range).
- Then `Npow = ceil(log2(A_min b_min / π))`, `A = π 2^Npow / b_min`. Measured errors are 0.25-0.57 ε on six models.
- Open: #48 (Lewis and COS from a tolerance).

## Lewis (`lewis.jl`)

Evaluates the exponent on the nodes `v >= 0` only (the integrand is even: `ψ(1/2 - iv) = conj ψ(1/2 + iv)`), with weight
2, 1 for the node 0 of an odd `N` (`LewisMethodResult.weights`), and uses the Clenshaw recurrence for smiles only
(`lw_integrate` with the mode; a single option keeps the broadcast, as cheap). Outside `SerialMode`, on the GPU, a smile
is `exp(c_k / 2) Σ_j a_j cos(v_j c_k + b_j)` through `phased_cos_sum`, the kernel of COS: one `cos` per term and strike
(one strike per column through a lazy `transpose(x)`, not a `reshape`), everything real, `lewis_amplitudes_phases`
(#46), with the nodes `v_j = v_0 + j h` and coefficients from `lewis_coefficients`; smiles return vectors in every mode.
Below about `eps(S0) ≈ 1.4e-14` its error alternates with `N` on a Float64 rounding floor (one ulp of the integral times
`√(S0 K) / π`, subtracted from `S0 e^{-dT}`): the convergence figure draws that floor as a dashed line (#40). A Black
Scholes control variate would lower it (#99).

## COS (`cos.jl`)

- Range from `src/02_range/` (`range.md`). Sharing the range across a smile shares the payoff coefficients and the
  phases of the characteristic function, so a strike costs only the series, against 3.5-4.5 times as much with a range
  per strike; a smile that is wide with respect to the standard deviation needs a larger `N`.
- The series (`cos_phased_series`) folds the strike-independent phases into complex coefficients once and, in
  `SerialMode`, sums them per strike with a Clenshaw recurrence (`cos_clenshaw`, evaluated with `map` for ReverseDiff):
  one `sincos` per strike instead of one `cos` per term and strike, 5-9 times faster for smiles of 100-500 strikes, no
  gain on a single option, where the characteristic exponent dominates. Other modes (GPU) keep the N×K broadcast
  `cos(u x + phase)`: the recurrence is sequential in `k` and indexes the coefficients element-wise.
- The COS nodes are a comprehension `k * π / (b - a)`, not a float range (Enzyme, `autodiff.md`). #96: Float32 Heston is
  2.2x slower than Float64 on the host.
