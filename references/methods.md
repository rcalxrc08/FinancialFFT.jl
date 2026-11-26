# Stage 05: pricing methods

_Dispatch, Carr-Madan, Carr-Madan-Lewis, the parameters from a tolerance, Lewis, COS, quadratures._ (Layer 3
reference.)

## Dispatch design

- Each method is a configurator struct subtyping `FinancialMonteCarlo.AbstractMethod` via `AbstractFFTMethod`
  (Carr-Madan, Carr-Madan-Lewis; use `fft`) or `AbstractIntegralMethod` (Lewis, COS; direct quadrature/cosine sums),
  `AbstractDensityMethod <: AbstractIntegralMethod` for the density methods (`src/01_hooks/abstracts.jl`).
- Pricing is `pricer(process, underlying, zero_rate, method, payoff)`, extending `FinancialMonteCarlo.pricer`, always with five
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

Every quadrature has a comment naming the rule and its order: the trapezoidal rule in Carr-Madan and Carr-Madan-Lewis
(half weight at `v = 0`, Simpson until #104), Simpson on `M` intervals in the density pricers
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
  (signs `(-1)^j`, trapezoidal weights, `dx/2`, the inverse of the denominator with one real division, `inv_denominator`)
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
- Aliasing (#104): the trapezoidal rule (weight 1/2 at `v = 0`, the integrand is even) has only the Poisson copies at
  the multiples of `2b` (period of the strike grid), weight 1; the bound takes weight 2 on the copies at `±2b`
  (covering the rest) and `b_min = D_min / 2`. Simpson's weights, before, added copies at `±b` with weight `-1/3` and
  needed `b_min = D_min`: one more power of two, measured on six models (#104). Each side `≤ ε/4`, from closed forms (`G ≤ F0`, `≤ e^{-rT}`, `≤ K e^{-rT}`) or a Chernoff
  bound (`AliasingExponent`, minimised with `chernoff_optimum`, shared with the COS range).
- Then `Npow = ceil(log2(A_min b_min / π))`, `A = π 2^Npow / b_min`. Measured errors 0.13-0.31 ε on six models, both
  methods, T ∈ {0.1, 1}, ε ∈ {1e-6, 1e-9}, vanilla and binary (2026-11-26, #104; Simpson: 0.25-0.57 ε with one more
  power of two in most cases).
- `LewisMethod(ε; N_max = 2^20 + 1)` (#48): `QuadratureGrid(A, N)` / `QuadratureTolerance(ε, N_max)` in the `grid` field
  (`lewis.jl`), `quadrature_grid` resolves them (inactive for Enzyme). Truncation: the same modulus and scale as
  Carr-Madan-Lewis (`lewis_error_setup`, the Lewis integrand on `1/2 + iv`), `truncation_frequency_bound` (no interpolation: the tail
  sum beyond `A`, one reverse cumulative sum over the nodes instead of the bisection of Carr-Madan), `≤ ε/2`. Aliasing: the rule is trapezoidal, copies at the multiples of `L = 2π/h` with weight
  1 and none at half the period; `lewis_aliasing_distance` with weight 2 (the first copy covers the geometric rest),
  `ε/4` per side, closed form for vanilla, Chernoff (γ = 1/2) on one side for binaries. `N = 2 ceil(A_min L_min / 2π) + 1`
  (odd), `A = (N - 1) π / L_min`. Measured 2026-11-26: errors 0.23-0.6 ε on Black–Scholes and the five `model_cases`
  (T ∈ {0.1, 1, 2}, ε ∈ {1e-5 … 1e-9}, vanilla and binary smiles); a BS smile at ε = 1e-9 takes N = 547 (60 μs) against
  `LewisMethod(600, 20000)` (2.4 ms), choosing the grid 93 μs (270 μs with the bisection of Carr-Madan, 30 sums of 2048
  terms; the modulus itself 45 μs); the low-variance Heston of #48 gets A = 2278
  (Heston's exponent decays only exponentially in v, slowly with a small variance).
- `CosMethod(ε; N_max = 2^20)` (#48, #101): `CosTerms(N)` / `CosTolerance(ε, N_max)` in the `grid` field (`cos.jl`),
  `cos_range_and_terms` resolves the range of log(S_T/S0) and `N` (inactive for Enzyme). Series truncation: the terms are
  below `|φ(u_k)|` times `cos_payoff_envelope` (vanilla `2[min(1, uL)|1/u - e^h u/(1+u²)| + (e^h+e^a)/(1+u²)]`, binary
  `2 min(L, 1/u)`), their tail beyond `u_N` estimated with `truncation_frequency_bound` (scale `K_max e^{-rT}`, or
  `e^{-rT}`), `≤ ε/2`, `N = ceil(u* (b-a)/π)`, at least 8. Range: tail probability `ε / (8 P C)`, `C` the sum of the
  envelope over the terms, never above the `1e-14` floor (the derivatives, #64). Measured 2026-11-26: errors 0.03-0.19 ε
  (the envelope is looser than the Lewis modulus) on BS and the five `model_cases`, T ∈ {0.1, 1}, ε ∈ {1e-6, 1e-9},
  vanilla and binary, N = 46-214 at ε = 1e-9; the Kou DAX optimum of #101 gets N = 1456 at T = 1/6 (error 1.5e-10,
  where `CosMethod(512)` errs by 1e-2).
- Open: #48 for the densities.

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
