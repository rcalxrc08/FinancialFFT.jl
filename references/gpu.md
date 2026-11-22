# GPU and execution modes

_What runs on the device and what each back-end cannot do there._ (Layer 3 reference.)

- `FinancialMonteCarlo.AbstractMode` (`SerialMode`, `MultiThreading`, `AbstractCudaMode`; `BaseMode` before 0.6) is the
  `mode` field of every method (`methods.md`, "Dispatch design").
- CUDA extension: `grid_interpolate` of a number on the device; the grids go to the device through `adapt_array`,
  which is `FinancialMonteCarlo.to_mode`, whose CUDA extension makes `CuArray`s keeping the element type.
- Kernels must not index arrays element-wise (`CUDA.allowscalar(false)`). The rrule of `sum` of PatternVectors gives any
  vector a `PatternVector` cotangent, and ChainRules then computes `dot(Ȳ, B)` with a device `B`: PatternVectors does
  that `dot` with broadcasts (`autodiff.md`).
- Zygote cannot differentiate broadcasts whose intermediate results are tuples: pass real and imaginary parts as
  separate arrays (#42 proposes complex arrays now that Zygote handles tuple broadcasts).
- ReverseDiff cannot store tracked reals in GPU arrays.
- Mooncake's CUDA broadcast rules reject arrays mixing real and complex element types (every kernel but the binary
  Lewis integrand), `mapreduce` other than `+` (the `extrema` of the strikes of a COS smile, inside
  `@ignore_derivatives`, which Mooncake does not see) and a differentiable `Ref` argument (the FFT density): reported
  upstream as chalk-lab/Mooncake.jl#1450, tracked in #63, and recorded in the `gpu` suite as `mooncake_gpu` in
  `cases.jl` (a tolerance for the binary Lewis option, `Broken` elsewhere).
- Enzyme does not work on the GPU: reverse mode on broadcasts over `CuArray`s (EnzymeAD/Enzyme.jl#3451), forward mode
  needs runtime activity (EnzymeAD/Enzyme.jl#4017) and has no derivative of the CUDA `__nv_sincos` of complex `exp`
  (EnzymeAD/Enzyme#2552, #3307). With CUDA < 6.5 forward mode is also silently wrong, primal included, after a
  host-to-device copy such as `adapt_array` (`convert(CuPtr, ::Managed)` loses the shadow, JuliaGPU/CUDA.jl#3332; found
  in rcalxrc08/FinancialMonteCarlo.jl#68, not tested here), and CUDA ≥ 6.4.2 needed LLVM.jl 10, which Enzyme did not
  allow (EnzymeAD/Enzyme.jl#3741; Enzyme 0.13.221 lifted the cap, then CUDA 6.5.1 hits EnzymeAD/Enzyme.jl#4094).
- The `gpu` suite runs METHOD_CASES (prices, ForwardDiff, Zygote) in `CudaMode`, with smile strikes as `CuArray`s, and
  compares other models and smiles with the CPU. GPU timings: `benchmark/ad_gpu.jl`. #92: a surface is launch-bound per
  maturity on the GPU.
