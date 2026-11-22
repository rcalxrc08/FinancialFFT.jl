# Decisions

_Architecture decisions, oldest first. Append; a reversed decision gets a new entry that says "supersedes D-NNN"._
(Layer 3 reference.) Read before a refactor: a structure that looks accidental may be an entry here.

## D-001 An extension of FinancialMonteCarlo

- **Decision:** reuse FinancialMonteCarlo's processes, options, rates and `pricer`; add only methods.
- **Why:** one call for Monte Carlo and Fourier prices; no duplicated model types.
- **Consequences:** `mission.md`; compat on FinancialMonteCarlo follows its releases.
- **Status:** active

## D-002 (#55) The execution mode is a field of the method

- **Decision:** `pricer` always takes four arguments; `mode` is a keyword of every constructor; `with_mode` rebuilds.
- **Why:** FinancialMonteCarlo 0.6 moved modes into methods; one dispatch path for CPU and GPU.
- **Consequences:** the five-argument forms are deprecated (#93).
- **Status:** active

## D-003 (#38) Chernoff truncation range with Brent's method

- **Decision:** the COS and density range from a Chernoff bound, solved by `brent_root` in transformed coordinates.
- **Why:** 4-26 exponent evaluations against 102-115 for bisection; range 3-18 times faster, accurate to 1e-13.
- **Consequences:** `range.md`; a new process with semi-heavy tails needs `exponential_moment_bound`.
- **Status:** active

## D-004 (#39) Quintic Lagrange interpolation at the strikes

- **Decision:** `grid_interpolate` with 6-point Lagrange and gathers.
- **Why:** 65 times more accurate than the cubic spline; differentiable by every back-end, on the device.
- **Consequences:** strike derivatives jump by ~1e-8 at the nodes (C⁰).
- **Status:** active

## D-005 (#44) Hermitian FFT density through `brfft_mod`

- **Decision:** the FFT density evaluates half the spectrum and transforms with `brfft`.
- **Why:** 40-47% faster, 43% less memory; a half-length complex FFT was twice as slow.
- **Consequences:** `brfft_mod` methods in every AD extension; `densities.md`, `fft_utils.md`.
- **Status:** active

## D-006 (#50) FFTA for non-IEEE floats

- **Decision:** FFTA (not GenericFFT.jl, not the own radix-2) for BigFloat, Float16, DoubleFloats.
- **Why:** faster `brfft`, errors of 1-3 eps against 3-27; 1.3-3.6 times faster than the radix-2 it replaced.
- **Consequences:** `InPlaceFFTPlan`; `fft_utils.md`.
- **Status:** active

## D-007 (#62) Enzyme through EnzymeCore only

- **Decision:** Enzyme support lives in an EnzymeCore extension (inactive marks, FFT rules); no Enzyme code in the core;
  no float ranges anywhere.
- **Why:** EnzymeCore is light; float ranges fail (EnzymeAD/Enzyme.jl#4016, FinancialCalibrator.jl#41).
- **Consequences:** constant setup in named functions; comprehension grids.
- **Status:** active

## D-008 (#47, #77) Zygote rules for Clenshaw and the vectorised exponent

- **Decision:** ChainRules rules for the Clenshaw sums and `characteristic_exponent_vi` (value and pullback in one
  `rrule_via_ad`).
- **Why:** Zygote was O(N²) on the loops (26 s, 26 GiB on a COS smile); recomputing the exponent cost 5-25%.
- **Consequences:** `fft_utils.md`, `charexp.md`.
- **Status:** active

## D-009 (2026-11-22) Layered context (ICM)

- **Decision:** the repository context follows the Interpretable Context Methodology: `CLAUDE.md` (L0), `CONTEXT.md`
  routing (L1), stage contracts (L2), `references/` (L3), `state/` (L4); `test/context.jl` keeps it in sync (#97).
- **Why:** the 5306-word `CLAUDE.md` loaded everything for every task; the layout worked in FinancialCalibrator and
  VibratoMonteCarlo.
- **Consequences:** `src/methods/` split into stage folders; every source file belongs to a contract.
- **Status:** active
