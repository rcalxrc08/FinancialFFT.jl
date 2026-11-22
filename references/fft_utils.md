# Stage 03: shared utilities

_Grids and float types, the FFT functions, Clenshaw, interpolation, payoff conversions (`src/03_utils/utils.jl`)._
(Layer 3 reference.)

## `brfft_mod` and the FFT plans

- `brfft_mod`: FFTW's complex-to-real `brfft` (CUFFT on the device, FFTA for other floats), with methods in every AD
  extension (components for DualNumbers, HyperDualNumbers, TaylorSeries; one batched `brfft` for ForwardDiff; a tape
  instruction for ReverseDiff; a primitive for Mooncake; a ChainRules `rrule` for Zygote, since the rule of AbstractFFTs
  builds its weights on the CPU). The adjoint is `brfft_adjoint`, `rfft` with weights 1, 2, …, 2, 1 (a PatternVector;
  an `rfft` plan can be passed). #43 can reuse `brfft_mod` for the Carr-Madan methods.
- Planning a real FFTW transform costs 38-114 µs (2^10-2^18), as much as a transform of 2^14: the extensions that
  transform several components (DualNumbers, HyperDualNumbers, TaylorSeries) share one `plan_brfft` (FFTW's or FFTA's;
  TaylorSeries writes into one output buffer with `mul_mod!`, FFTW's `mul!` or a copy for FFTA, which has no `mul!` for
  real transforms, and reuses one input buffer), the forward rules of Mooncake one plan for primal and tangent, and
  ReverseDiff keeps its plans (`fft`/`bfft`, `brfft`/`rfft`) in the cache of the tape instruction, reused by a compiled
  tape; a single transform per call (Zygote, Mooncake's reverse rules, ForwardDiff's batch) has nothing to reuse. The
  ReverseDiff `brfft_mod` returns its `TrackedArray`, not a vector of tracked reals: the broadcasts after it are one
  instruction each (ReverseDiff 52-64% faster).
- The `fft` overloads of the AD back-ends take vectors: every FFT is of a vector.
- FFTA has no in-place plans: the `fft!` of the DualNumbers, HyperDualNumbers, ForwardDiff and TaylorSeries extensions
  call `plan_fft_mod!`/`fft_mod!`, FFTW's `plan_fft!` (CUFFT's on the device) for ComplexF32/ComplexF64 and FFTA's
  out-of-place plan copied into the argument (`InPlaceFFTPlan`) for other floats, so every back-end differentiates the
  Carr-Madan methods in BigFloat too (suite `types`).

## Floating-point types (suite `types`)

- Prices keep the float type of the inputs. `float_type(x...)` (the promoted type of the `v_value_mod` of the inputs,
  `float`ed: no arithmetic on the values) is the type of grids and ranges, and `to_float_type(FT, x)` converts method
  parameters (`A`, `α`) to it.
- Avoid Float64 literals and `pi * N` (an irrational times an integer is a Float64) in setup code; use the constants of
  IrrationalConstants (`twoπ`, `invπ`, imported in `src/FinancialFFT.jl`), which take the type of the other operand,
  and rationals for other constants (`fft_parameters.jl`), converted to the float type once outside loops.
- FFTW handles Float32/Float64 only: other floats (BigFloat, DoubleFloats) are transformed by FFTA (#50; a dependency,
  loaded with `import FFTA`), through the same AbstractFFTs functions and plans (`fft`, `brfft`, `rfft`, `plan_brfft`),
  on any length; FFTW's methods are more specific, so Float32/Float64 stay on FFTW and `CuArray`s on CUFFT. FFTA was
  chosen over GenericFFT.jl (actively maintained, faster `brfft`, errors of 1-3 eps against 3-27): against the radix-2
  FFT it replaced, `fft` 1.3-2.5 times and `brfft` 1.7-3.6 times faster on Float16, Double64, BigFloat (2^10, 2^14).
  AD element types still reach the `fft`/`brfft_mod` overloads of the extensions.

## Clenshaw recurrence

- `clenshaw_exp_sum` (complex coefficients as real/imaginary arrays, `Val(true)` for the real part only) and
  `clenshaw_cos_sum` (real coefficients) are shared by COS, Lewis smiles and `CosDensity` (its `density_value`, and its
  `density_array` in `SerialMode`).
- Near θ = 0 or π (`|cos θ| > 0.95`) they switch to Reinsch's form, on `d_k = b_k - σ b_{k+1}`: about 1.8 times slower
  per term (longer dependency chain), but on the Lewis coefficients, where `θ = h c` is close to 0 at the money, the
  plain recurrence errs by up to 1e-7 and Reinsch's by 1e-13.
- Both have ChainRules `rrule`s (#47): Zygote differentiated the element reads of their loops with a dense cotangent
  per read, O(N²) per call (26 s and 26 GiB for a COS smile of 101 strikes); the rules make one pass over `k`, with
  `cos kθ`, `sin kθ` by rotation refreshed every 32 terms, giving the coefficient cotangents `(ā + i b̄) e^{-ikθ}` and θ̄
  from the plain sums of `S' = Σ i k c_k e^{ikθ}` (no second recurrence). Only Zygote uses them. The pullback is real
  and fully `@muladd` (#72; `d = d - k * (…)`, since the macro does not fuse `-=`): measured 2026-11-22 (Julia 1.13.4,
  Windows laptop, old and new in one process) 14-26% faster on real series, 0-14% on complex ones, n = 257 and 2048.
  The `@muladd` of the Lewis binary coefficients measured 3% slower at n = 2048 and was not kept. The real series of
  `clenshaw_cos_sum` has its own pullback, `clenshaw_cos_sum_pullback`, without the imaginary terms (#74): 4-14% faster
  than the complex one with zero imaginary part (n = 65-2049, interleaved in one process). Open: #100 (the remaining
  real multiply-adds).

## Interpolation at the strikes

Both Carr-Madan methods interpolate the FFT output at the strikes with `grid_interpolate`: 6-point Lagrange
(`lagrange_quintic`, order `h⁶`, about 65 times more accurate than the cubic spline it replaced, #39) with gathers, whose
pullback is a scatter, so every back-end differentiates it, on the device too; a number on the device is interpolated as
a one-point array and summed (CUDA extension). The interpolant is only C⁰: strike derivatives jump at the nodes by about
1e-8.

## Payoff conversions

Everything but COS and the density pricers is priced as a call and converted with `call_to_put` (dispatching on
vanilla/binary/smile option types). COS and the density pricers expand the put and convert it with `put_to_option`: the
coefficients of a call grow like `exp(b)`, so a call needs the right tail under the share measure, which the range does
not cover and widening it would only trade for rounding errors.
