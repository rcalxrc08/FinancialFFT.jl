using Documenter, FinancialFFT, FinancialMonteCarlo
makedocs(
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico"], repolink = "https://gitlab.com/rcalxrc08/FinancialFFT.jl"),
    repo = "https://gitlab.com/rcalxrc08/FinancialFFT.jl.git",
    sitename = "FinancialFFT.jl",
    modules = [FinancialFFT],
    pages = ["index.md", "methods.md", "payoffs.md", "models.md", "ad.md", "api.md"],
)
get(ENV, "CI", nothing) == "true" ? deploydocs(repo = "https://gitlab.com/rcalxrc08/FinancialFFT.jl.git") : nothing
