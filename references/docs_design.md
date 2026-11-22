# Documentation design

_The site, its figures and the shared design._ (Layer 3 reference.)

- `docs/make.jl` (Documenter) first runs `docs/figures.jl`, which computes the figures with the package and saves each
  as SVG in a light and a dark variant in `docs/src/assets/figures/` (git-ignored); `assets/figures.css` shows the one
  of the reader's theme. Figure palette (`THEMES`): blue, teal, amber, cyan, green, darker steps in the light theme.
- `calibration.md` calibrates Heston to four smiles (Levenberg–Marquardt on implied volatilities from `pricer` and
  FinancialToolbox's `blsimpv`, Jacobian and Hessian by ForwardDiff, identifiability from the singular values, implicit
  sensitivities to the quotes checked against recalibrating with a bump of 1e-2: smaller bumps hit the rounding floor
  of the recalibrations); its figure `calibration` comes from `calibration_data()` in `figures.jl`, which runs the
  page's own `@example calibration` blocks up to `model_vols` (no copy of the code; keep that variable and the block
  structure).
- `references.md` lists the papers cited by the pages. `methods.md` explains the Chernoff range (section "Truncation
  range": the bound, the moment bound, the tolerance, the search; figure `chernoff`, the bound `t(s)` of the lower tail
  per model with the optimum of `chernoff_optimum`): keep it in step with `src/02_range/extrema_computation_opt.jl`.
  `models.md` documents the characteristic exponents. Plots exports a `density` too: qualify it as
  `FinancialFFT.density` there.
- The design is shared by the docs of all the rcalxrc08 packages (identical copies, edit them together):
  `assets/style.css` (Dark Quant palette), `dark.js` (dark theme by default), `figures.css`, `site.css` (page headers
  `fc-page-head` with kicker, the first paragraph as lede, chips and an `fc-toc` strip of section links; numbered
  sections; figure, maths, table, docstring and footer cards) and `home.css` (the landing page: hero, stat tiles, feature
  cards, flow diagram, results behind radio-button tabs, quick start steps).
- The hero is `hero_figure` at the end of `figures.jl`: the densities of the six showcase models by
  `FFTDensityInverter`, as ridges (no volatility surface: that is FinancialCalibrator's hero). The sidebar groups the
  pages into Guides and Reference. `index.md` is the landing page; its GPU section is the target of `[GPU](@ref)` in
  `ad.md` and `calibration.md`.
- `ad.md`, "Choosing a back-end": the tables printed by `benchmark/ad.jl` and `benchmark/ad_gpu.jl` (#61).
- Markdown tables need `|:--|` columns. #49: a page on the Dual Spectral Theorem (LewisMethod + TaylorSeries).
- The logo and favicon live in `etc/` (drawn by `etc/logo.jl`).
