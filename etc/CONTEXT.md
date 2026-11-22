# etc: logo

## Inputs

| Load | Why |
|:--|:--|
| `logo.jl` | draws `logo.png` and `favicon.ico` (needs Luxor, ForwardDiff, FinancialFFT, FinancialMonteCarlo, FinancialToolbox) |
| `references/docs_design.md` | the logo and favicon are copied into the docs assets |

## Process

- `julia etc/logo.jl` in an environment with the packages above; commit the PNG and ICO.

## Outputs

`logo.png`, `favicon.ico`.
