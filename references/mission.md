# Mission

_Why the package exists and how to tell whether it succeeds._ (Layer 3 reference: stable, changes only with a decision
in `decisions.md`.)

## Purpose

FinancialFFT.jl prices European options (vanilla, binary, smiles) and computes densities of the log-return with
Fourier methods (Carr-Madan, Carr-Madan-Lewis, Lewis, COS, FFT and COS densities) for the processes of
FinancialMonteCarlo.jl, through the same `pricer(process, zero_rate, method, payoff)` call, differentiable by every AD
back-end and runnable on the GPU.

## Goals

- One `pricer` per method for options and smiles; densities for the likelihood-ratio and vibrato methods of
  VibratoMonteCarlo and for FinancialCalibrator.
- Prices and Greeks of any order through DualNumbers, HyperDualNumbers, ForwardDiff, TaylorSeries, ReverseDiff, Zygote,
  Mooncake and Enzyme (CPU).
- Parameters from a tolerance (Carr-Madan methods; #48 for Lewis and COS); ranges from a Chernoff bound.
- Float type of the inputs kept, BigFloat included.

## Constraints

- An extension of FinancialMonteCarlo: no own processes or option types; the only exports are `pricer`, `density` and
  the method structs.
- CPU and CUDA from the same code (execution mode as a field of the method).
- PatternVectors from the private registry.

## Success criteria

- The suites pass on CI (1.11, 1.12) and the `gpu` suite locally; `Broken` cases in `cases.jl` stay documented.
- Errors against FinancialToolbox and `LewisMethod` within the tolerances of `cases.jl`.

## Non-goals

- Path-dependent options by Monte Carlo (FinancialMonteCarlo); calibration (FinancialCalibrator); closed forms
  (FinancialToolbox). Barriers for Lévy models are a proposal (#91).
