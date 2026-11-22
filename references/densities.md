# Stage 06: densities

_The density interface, `CosDensity`, `FFTDensity` and the density pricer._ (Layer 3 reference.)

## Interface (`density.jl`)

- `density(process, zero_rate, method, T)` is the interface, like `pricer`, for callers that need a density without a
  closed formula (Vibrato, likelihood ratio method): it returns an `AbstractDensity` of the log-return
  `log(S_T / S_0)`, independent of `S_0`, zero outside `[a, b]`.
- The interface defines `f(y)` (a number, on the CPU) and `density_values(f, y)` (an array) for every density, from two
  functions a density type implements without the mask: `density_value(f, y)` at a number and
  `density_array(f, y[, mode])` at an array (required since #78, no default map over `density_value`; the mode lets a pricer pick the CPU
  kernel). `mask_outside_range` (in `utils_density.jl`) zeroes them outside the range.
- A method that computes a density subtypes `AbstractDensityMethod` and adds a `density` method; with an `M` field it
  also prices through the one `pricer` of `utils_density.jl`, `density_integral_price`: Simpson's rule for the put on a
  log-moneyness grid ending at the strike (constant for differentiation: `ignore_derivatives_mod`), call by parity; a
  smile shares the density, so its prices are the ones of the single options. `density_integral_price` builds its
  nodes × strikes matrix through a lazy `transpose(x)`.
- `pricer` and `density` take four arguments: the five-argument forms with a trailing mode, deprecated since 0.4, were
  removed in 0.7 (#93).

## `CosDensityMethod(N, M)` (`cos_density.jl`)

Gives a `CosDensity` on the range of COS: `density_value` is a Clenshaw recurrence (`clenshaw_cos_sum`),
`density_array` broadcasts and a reduction only, with the coefficients as broadcast arguments (GPU), and in
`SerialMode` the recurrence per node (`map`), which `density_values` of a CPU density at an `Array` uses too
(`CosDensity{<:Array}`: 7-10 times faster than the N×K broadcast at 1000 points, no allocation but the result).

## `FFTDensityInverter(N, M)` (`fft_density.jl`)

- `DensityInverter`, which priced with a density in log-moneyness and needed `xmax`, was removed in its favour.
- Gives an `FFTDensity`: one FFT (`fft_density_values`) on the Chernoff range of COS, and cubic Lagrange interpolation on
  four nodes (`lagrange_cubic`), with gathers, so that every back-end differentiates it and it runs on the device; the
  range goes through `ignore_derivatives_mod`, a zero-derivative primitive in the Mooncake extension (Mooncake ignores
  `@ignore_derivatives`, and a grid moving with the parameters adds the derivative of the interpolation error to the
  scores).
- The FFT input is Hermitian (#44): the exponent is evaluated on the frequencies `k = 0, …, 2^(N-1)` only, on a constant
  grid `u_im = -i u_k` (the minus sign, since `brfft` has the `+i` sign), with the phase `(drift - corr - a) u_im`
  (anchoring the grid at `a` replaces the signs `(-1)^k`), and transformed by `brfft_mod` (`src/03_utils/utils.jl`).
  The Nyquist node enters through its real part. Negative values (ripples of slow-decaying cases, VG at short
  maturities) are floored at `eps`, no longer reflected by `abs`.
- Measured on Merton against the full transform (N = 14, 18): density and price 40-47% faster, every CPU back-end
  25-64% (Taylor1 the least, ReverseDiff the most), memory 43% less; on the GPU 17-32% at N = 18, noise at N ≤ 14
  (kernel launches dominate). A transform written as a half-length complex FFT (`fft` plus packing) was rejected: its
  preprocessing made it twice as slow as `brfft`.
- `fft_density_values` integrates the rate and the dividend separately (two `integral`s, #94), as every other method.
- It replaces the FFT density of VibratoMonteCarlo.jl (`lrm_aug.jl`), to be removed there after the next release
  (VibratoMonteCarlo issue #37 says how). Float16 overflows in the FFT density (not supported).
