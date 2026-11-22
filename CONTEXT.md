# CONTEXT.md: task routing (Layer 1)

Find the task, open the contract it points to, and load only the inputs listed there.

| Task | Contract |
|:--|:--|
| Method type hierarchy, the scalar AD hooks (`real_mod`, `v_value_mod`, `sqrt_mod`, …) | `src/01_hooks/CONTEXT.md` |
| The Chernoff truncation range, `brent_root`, `exponential_moment_bound` | `src/02_range/CONTEXT.md` |
| Grids and float types, `brfft_mod` and FFT plans, Clenshaw sums, strike interpolation, payoff conversions | `src/03_utils/CONTEXT.md` |
| Characteristic exponents of the processes, a new process | `src/04_charexp/CONTEXT.md` |
| Carr-Madan, Carr-Madan-Lewis, parameters from a tolerance, Lewis, COS, `pricer` | `src/05_methods/CONTEXT.md` |
| `density`, `CosDensity`, `FFTDensity`, the density pricer | `src/06_densities/CONTEXT.md` |
| AD back-end extensions, CUDA | `ext/CONTEXT.md` |
| Suites, `cases.jl`, `Broken` cases, the GPU suite | `test/CONTEXT.md` |
| Documentation pages, figures, the calibration example | `docs/CONTEXT.md` |
| Benchmarks and AD tables | `benchmark/CONTEXT.md` |
| The logo | `etc/CONTEXT.md` |
| Commands, CI, releases, sub-environments, untracked files | `references/workflows.md` |
| What is being worked on, known issues | `state/STATUS.md` |
| Whether a change fits the package's goals; why a rule exists | `references/mission.md`, `references/decisions.md` |

Shared references (Layer 3), each loaded only when a contract lists it: `public_api.md`, `conventions.md`,
`autodiff.md`, `gpu.md`, `dependencies.md`, `workflows.md`, `testing.md`, `performance.md`, `docs_design.md`, and per
stage `range.md`, `fft_utils.md`, `charexp.md`, `methods.md`, `densities.md`.

The stages communicate through Julia values: 01 defines the method hierarchy and hooks; 02 returns the range `[a, b]`
of a process at `T`; 03 provides grids, transforms and sums; 04 the exponent `ψ_i(imv, process, T)`; 05 returns prices
from `pricer(process, zero_rate, method, payoff)`; 06 returns an `AbstractDensity` from `density(process, zero_rate,
method, T)` and prices through it.
