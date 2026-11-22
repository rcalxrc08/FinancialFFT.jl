# Documentation

The Documenter site (`DeployPage.yml`).

## Inputs

| Load | Why |
|:--|:--|
| `make.jl` | runs `figures.jl` first, then the pages (Guides and Reference) |
| `figures.jl` | figures in light and dark SVG, `calibration_data()`, `hero_figure`, the `chernoff` figure |
| `references/docs_design.md` | the shared design, the calibration example, what must stay in step with the code |
| `references/public_api.md` | every exported name is documented |
| `references/workflows.md` | how to build the docs |

## Process

- `calibration_data()` runs the page's own `@example calibration` blocks: keep the variable `model_vols` and the block
  structure.
- `methods.md` "Truncation range" follows `src/02_range/extrema_computation_opt.jl`; `ad.md` tables come from
  `benchmark/ad.jl` and `benchmark/ad_gpu.jl`.
- The design files are identical copies across the rcalxrc08 packages: edit them together.

## Outputs

A site that builds with no warnings.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| Docs build in a separate process: `julia --project=docs docs/make.jl` | builds, no warnings |
