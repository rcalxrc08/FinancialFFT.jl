# Public API

_The exported names._ (Layer 3 reference.) Processes, options, rates and modes come from FinancialMonteCarlo.

| Name | Signature | Stage | Notes |
|:--|:--|:--|:--|
| `pricer` | `pricer(process, underlying, zero_rate, method, payoff)` | 05, 06 | extends `FinancialMonteCarlo.pricer`; options and smiles; the five-argument form was removed in 0.7 (#93) |
| `CarrMadanMethod` | `CarrMadanMethod(A, Npow, α = 0.75; mode)`, `CarrMadanMethod(ε; α = 0.75, Npow_max = 23, mode)` | 05 | FFT, Simpson, quintic interpolation |
| `CarrMadanLewisMethod` | `CarrMadanLewisMethod(A, Npow; mode)`, `CarrMadanLewisMethod(ε; Npow_max = 23, mode)` | 05 | FFT of the Lewis integrand |
| `LewisMethod` | `LewisMethod(A, N; mode)`, `LewisMethod(ε::AbstractFloat; N_max = 2^20 + 1, mode)` | 05 | direct quadrature |
| `CosMethod` | `CosMethod(N; mode)` | 05 | Fang–Oosterlee, Chernoff range |
| `density` | `density(process, underlying, zero_rate, method, T)` | 06 | an `AbstractDensity` of `log(S_T / S_0)` |
| `CosDensityMethod` | `CosDensityMethod(N = 128, M = 2^10; mode)` | 06 | `CosDensity` |
| `FFTDensityInverter` | `FFTDensityInverter(N = 14, M = 2^10; mode)` | 06 | `FFTDensity` |

Not exported, used by other packages: `characteristic_exponent_i`, `characteristic_exponent_vi` (VibratoMonteCarlo,
`etc/logo.jl`), `brent_root`, `chernoff_optimum`, `exponential_moment_bound`, `density_value`, `density_values`.
