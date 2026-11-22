# Conventions

_Coding standards that hold in every stage._ (Layer 3 reference.)

- Prices keep the float type of the inputs: grids and ranges in `float_type(x...)`, parameters through
  `to_float_type`; no Float64 literals, no `pi * N`; IrrationalConstants and rationals (`fft_utils.md`).
- Keep rationals out of complex lazy arguments (2.3x slower, `methods.md`).
- Numeric kernels call the hooks (`real_mod`, `imag_mod`, `sincos_mod`, `sqrt_mod`, `v_value_mod`, `adapt_array`, …),
  never `real`/`imag`/raw arrays (`autodiff.md`).
- Constants for AD inside `@ignore_derivatives ignore_derivatives_mod.(…)`; constant setup in named functions Enzyme
  can mark inactive; no float ranges on any path (Enzyme).
- No element-wise indexing of arrays that may live on the device; one strike per column through a lazy `transpose`.
- Pricing paths are type stable (`@inferred` in the `types` suite; Mooncake fails on small `Union`s).
- Every quadrature has a comment naming the rule and its order (#88).
- `@muladd` for every `a * b + c`, real numbers only (complex kernels stay as they are); `@muladd` does not fuse `-=`/`+=`,
  write `d = d - k * x`; keep a fusion only if it does not measure slower (#72; open: #100).
- Validation by `ArgumentError`/`DomainError`, not `@assert` (#89 open).
- Reuse before writing: FinancialMonteCarlo, FinancialToolbox, PatternVectors, then established libraries.
- Formatting: JuliaFormatter with `.JuliaFormatter.toml` (margin=1200, effectively no line wrapping).
