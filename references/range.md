# Stage 02: the Chernoff truncation range

_`src/02_range/extrema_computation_opt.jl`: the range `[a, b]` of COS and of the densities, `chernoff_optimum`,
`brent_root`, `exponential_moment_bound`._ (Layer 3 reference; `docs/src/methods.md`, section "Truncation range", and
its figure `chernoff` must stay in step with this file.)

- `[a, b]` comes from a Chernoff bound on the tails of the log-price, `t(s) = (T ψ_i(mul s) - log ε) / s` minimised over
  `s` (#38): `t'(s)` has the sign of `g(s) - L`, `g = s κ' - κ` (increasing for a convex exponent `κ`), `L = -log ε`,
  and one complex step of `κ` gives `g` (real part `κ`, imaginary part over `h` `κ'`; `chernoff_g`); `mul = 1` is the
  upper tail, so the endpoint is `driftT + mul t`, and only skewed models tell the two sides apart.
- The complex step is `eps(s)`, except for Heston (`sqrt(eps(s))`), whose imaginary parts cancel when `γ` is imaginary.
- The bracket starts at `s_min = 1` and at the Gaussian guess `s sqrt(L / g(s))` from it (exact for Black Scholes,
  `~1/√T`, beyond the root for heavier tails), moved to the next guess, at least doubled, while `g < L`, and halved
  towards the lower end where the exponent overflows (`g` not finite).
- Then Brent's method (`brent_root`) in the coordinates `u = log(s / (B - s))` (`log s` for `B = ∞`;
  `BoundedCoordinate`/`UnboundedCoordinate`, chosen once by dispatch) and `log(g - g0)` (computed as one log of
  `(g - g0) / (L - g0)`: 1-8% faster ranges, and accurate to eps near the root, where a difference of logs cancels),
  where `g` is a line both for a Gaussian and near a pole of the exponent at the bound `B` (on `g` itself Brent ended in
  bisection: `g ~ exp(s²)` in Merton, the NIG root 4e-6 below `B`), to a relative tolerance `sqrt(eps(FT))` on `s`: `t`
  is flat at its minimum, so the endpoints match a solve to 1e-14 to about 1e-13, and any `s` gives a valid, wider,
  bound. 4-26 evaluations of the exponent per range (both tails), against 102-115 for the bisection it replaced (range
  3-18 times faster). VibratoMonteCarlo#57 plans to reuse `brent_root` for its American bisection.
- The Heston `exponential_moment_bound` solves `T / τ(s) - 1 = 0` for the explosion time `τ` with the same
  `brent_root`, written with the rate `heston_explosion_rate` = `1 / τ` (zero where the moments never explode,
  continuous through `D = 0`: no `Inf`, and type stable in Float32, where returning `Inf` made it a `Union`; the inverse
  has no closed form) (0.7 µs against 4.3 µs for the bisection), stepping back to the finite side.
- The search never evaluates beyond `B (1 - sqrt(eps))`, with `B = exponential_moment_bound(process, mul[, T])`, where
  `E[exp(mul s X)]` stops being finite (Kou `λ₊`/`λ₋`, VG/NIG from the log/sqrt argument, Heston from the moment
  explosion time, hence the `T`; `Inf` otherwise): past it `characteristic_exponent_i` returns meaningless finite
  values. A new process with semi-heavy tails needs an `exponential_moment_bound` method.
- Tolerances: `scaled_tolerance(FT, 1e-14)`, `sqrt(eps(FT))` on the optimal `s` (also for the aliasing bound of
  `fft_parameters.jl`).
- The range is for `log(S_T / S0)`: on `y = log(S_T / K)` it moves by `log(S0 / K)`, a constant for AD, and a smile uses
  the union of the ranges of its strikes (the `extrema` of the strikes inside `@ignore_derivatives`).
- The search (`chernoff_optimum`) takes the exponent as a callable (`TailExponent(model, T, mul)` here) with
  `complex_step` and `exponential_moment_bound` methods, so other bounds reuse it (`AliasingExponent` of
  `fft_parameters.jl`).
- Still open from #38 (point 1, design to be approved): the search runs in the number type of the model (duals,
  TaylorN), whose derivative components are discarded; the proposal is one generic `value_model` rebuilding a process
  from the `v_value_mod` of its fields with `ConstructionBase.constructorof`, not a helper per model. #64: differentiate
  the range for COS (Greeks of order 2-4 up to 1000 times more accurate).
- The `oftype` in this file is on a constant path (FluxML/Zygote.jl#1751 does not bite).
