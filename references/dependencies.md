# Dependencies

_Why each dependency exists, hard or weak, and its compat floor._ (Layer 3 reference.)

| Package | Hard/weak | Why | Compat floor and why |
|:--|:--|:--|:--|
| `FinancialMonteCarlo` | hard | processes, options, `pricer`, modes, `to_mode`, curves | 0.7, 0.8, 0.9 (0.9: `SVector` curves, only `integral` is called) |
| `FFTW` | hard | Float32/Float64 FFTs | 1 |
| `FFTA` | hard | FFTs of other floats (#50) | 0.3 |
| `PatternVectors` | hard (LocalRegistry) | quadrature weights | 0.1.9 (GPU `dot`, unthunked cotangents, out-of-line error) |
| `ChainRulesCore` | hard | Zygote rules, `@ignore_derivatives` | 1 |
| `MuladdMacro` | hard | `@muladd` | 0.2 |
| `IrrationalConstants` | hard | `twoπ`, `invπ` | 0.2 |
| `Adapt` | hard | device arrays | 3, 4 |
| `CUDA` | weak | `grid_interpolate` of a number on the device | 5, 6 |
| `DualNumbers`, `HyperDualNumbers`, `ForwardDiff`, `TaylorSeries` | weak | `fft` overloads, hooks | ForwardDiff 0.10/1; TaylorSeries 0.15-0.23 |
| `ReverseDiff` | weak | tape instructions | 1.18 (1.17 overflows the stack on complex broadcasts) |
| `Mooncake` | weak | FFT primitives, `sqrt_mod` overlay | 0.5 |
| `EnzymeCore` | weak | inactive marks, FFT rules | 0.8 |

Upstream workarounds (cite issue and version; recheck on every upstream release, remove when fixed):

| Workaround | Upstream issue | Affected versions | Last checked |
|:--|:--|:--|:--|
| comprehension grids instead of float ranges | EnzymeAD/Enzyme.jl#4016 | Enzyme 0.13 | 2026-11-21 |
| Float32 Heston wrong under Enzyme (documented, not fixed) | EnzymeAD/Enzyme.jl#4108 | Enzyme 0.13.221 | 2026-11-21 |
| `mooncake_gpu` `Broken` cases (Carr-Madan, `FFTDensityInverter`) | #63 (chalk-lab/Mooncake.jl#1450 fixed: Lewis, COS, `CosDensityMethod` pass) | Mooncake 0.5.72 | 2026-11-22 |
| `sqrt_mod` overlay for Mooncake | Mooncake bitcasts Base's complex `sqrt` | Mooncake 0.5.72 | 2026-11-21 |
| phases folded into complex `exp` for TaylorN | JuliaDiff/TaylorSeries.jl#445 (no `cis`) | TaylorSeries 0.23 | 2026-11-19 |
| Enzyme not supported on the GPU | EnzymeAD/Enzyme.jl#3451, #4017, #2552, #3307, #4094; JuliaGPU/CUDA.jl#3332 | Enzyme 0.13, CUDA 6.5 | 2026-11-21 |
| `promote`/`convert` instead of `oftype` on differentiated paths | FluxML/Zygote.jl#1751 (#1752-#1754 fixed in Zygote 0.7.19) | Zygote 0.7.19 | 2026-11-22 |
