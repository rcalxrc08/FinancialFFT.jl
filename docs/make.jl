using Documenter, FinancialFFT, Literate, FinancialMonteCarlo
EXAMPLE = joinpath(@__DIR__, "..", "examples", "getting_started.jl")
OUTPUT = joinpath(@__DIR__, "src")
Literate.markdown(EXAMPLE, OUTPUT; documenter = true)
# makedocs(format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico"]), sitename = "FinancialFFT.jl", modules = [FinancialFFT], pages = ["index.md", "types.md", "stochproc.md", "parallel_vr.md", "payoffs.md", "metrics.md", "multivariate.md", "intdiffeq.md", "extends.md"])
makedocs(format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico"]), sitename = "FinancialFFT.jl", modules = [FinancialFFT], pages = ["index.md", "types.md", "stochproc.md", "metrics.md"])
get(ENV, "CI", nothing) == "true" ? deploydocs(repo = "https://gitlab.com/rcalxrc08/FinancialFFT.jl.git") : nothing