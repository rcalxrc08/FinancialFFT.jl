# Stage 04: characteristic exponents

_`src/04_charexp/charexp.jl`._ (Layer 3 reference; `docs/src/models.md` documents the exponents.)

- Per-process `characteristic_exponent_i(imv, process)` (scalar) and `characteristic_exponent_vi` (broadcast),
  implemented as real functions of an imaginary-argument input (`ψ(v) = ψ_i(v*im)`) to avoid complex arithmetic. A new
  process needs these methods (and `exponential_moment_bound` with semi-heavy tails, `range.md`).
- A broadcast of `characteristic_exponent_i` over a lazy argument (`v .* im .+ c`) goes to `characteristic_exponent_vi`
  through a `Base.broadcasted` overload, in a loop of its own: faster than fusing the scalar exponent into the
  surrounding broadcast (Lewis 0.68 against 0.95 ms, FFT density 1.60 against 1.93 ms on Merton without it), at the cost
  of one array; an array argument is broadcast element by element. Aqua excludes the ambiguity of this
  `Base.broadcasted` against `Dictionaries.DictionaryStyle` (`test/aqua_test.jl`).
- Its ChainRules `rrule` (Zygote) takes the value and the pullback of `characteristic_exponent_vi` from one
  `rrule_via_ad` in the forward pass (#77; recomputing it in the pullback cost 5-25% of a gradient, CPU and GPU, Float64
  and Float32).
- Methods call the three-argument `characteristic_exponent_i(imv, process, T)`, the log of `E[exp(imv X_T)]`, which
  defaults to `ψ_i * T`: a process that is not Lévy (Heston, nonlinear in `T`) defines the three-argument `_i`/`_vi`
  instead.
- Complex `sqrt` goes through `sqrt_mod` (Mooncake cannot differentiate Base's, `autodiff.md`).
- Open: #82 (LogNormalMixture exponent, ShiftedProcess by a strike shift), #90 (Heston from an arbitrary starting
  variance). `charexp_and_ders.jl` (untracked, git-excluded) is WIP for derivatives.
