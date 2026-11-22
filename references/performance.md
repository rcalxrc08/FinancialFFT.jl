# Performance

_Where the measurements live and how to measure._ (Layer 3 reference; `benchmark/` updates them.)

| Numbers | Where |
|:--|:--|
| BenchmarkTools suite: Black Scholes per method, models, densities, the Chernoff range in Float64/ForwardDiff/TaylorN | `benchmark/benchmarks.jl`, Benchmark workflow artifacts |
| AD: every back-end, every method, Heston tasks (gradient, Jacobian, gradient in the strikes, Hessian) as multiples of a price | `benchmark/ad.jl` → `docs/src/ad.md` "Choosing a back-end" |
| AD on the GPU (ForwardDiff, Zygote in `CudaMode`) | `benchmark/ad_gpu.jl` (needs CUDA in the environment) |
| Hermitian FFT density, FFT planning, ReverseDiff tape | `densities.md`, `fft_utils.md` |
| Chernoff range vs bisection | `range.md` |
| Carr-Madan weights, rationals in lazy complex arguments | `methods.md` |
| COS smile recurrence, Lewis | `methods.md` |

How to measure:

- One part of the suite per process (`benchmark/run.jl`); a hot-loop change is timed before and after (old commit in a
  `git worktree`) on an idle machine.
- Zygote: interpolated non-constant arguments, no `$(args)...` splat (FluxML/Zygote.jl#1752, fixed in Zygote 0.7.19:
  still the fair way to time it on older versions).
- Open: #96 (Float32 Heston COS 2.2x slower than Float64 on the host), #92 (surfaces launch-bound on the GPU).
