# Testing

_The suites, the case tables, known failures._ (Layer 3 reference.)

- `test/runtests.jl` loads the shared `utils.jl` and `cases.jl`, runs `context.jl`, then `include`s `<suite>_test.jl` for
  every name in its `SUITES` list (or only those passed as `test_args`). A new suite file must be added to `SUITES`;
  other files in `test/` are not run. The testsets report only at the end, so `runtests.jl` logs `Running
  <suite>_test.jl` and `Finished <suite>_test.jl in N s` (flushed): a hung suite shows up by name in a local log or on
  CI.
- A full local run takes about 50 minutes (`gradients`, `enzyme` and `other_models` over 10 minutes each; #81 tracks the
  slow suite).
- `gpu_test.jl` is kept out of `SUITES` because CI has no GPU; for the same reason CUDA is not a test dependency, and
  `runtests.jl` adds it to the test environment only when the suite is selected.
- `cases.jl` (`METHOD_CASES`) declares, per method, the supported payoffs and the tolerance for each number type / AD
  back-end; `black_scholes` and `gradients` loop over it against `blsprice`/`blsbin` from FinancialToolbox.
  `other_models` compares the other methods with `LewisMethod` on Merton, Kou, VG, NIG and Heston (`model_cases`).
- Known failures are recorded as `Broken(toll)` (see `utils.jl`), which becomes `@test_broken`: the suite reports
  "Unexpected Pass" as soon as one starts working, so update `cases.jl` when fixing one.
- `aqua` runs `Aqua.test_all` (the one excluded ambiguity, `Base.broadcasted` of the vectorised characteristic exponent
  against `Dictionaries.DictionaryStyle`, is explained in `aqua_test.jl`); `types` also checks with `@inferred` that
  every pricer and density is type stable, and that prices keep the float type (BigFloat through FFTA).
- `fft_parameters` checks the tolerance-chosen grids and that the parameters are constants on every back-end.
- Coverage gaps: #78 (CosDensity broadcast only on the GPU, the unused default `density_array`).
- Reference values: FinancialToolbox closed forms for Black–Scholes; `LewisMethod` for the other models.
