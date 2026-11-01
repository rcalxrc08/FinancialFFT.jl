using Documenter, FinancialFFT, FinancialMonteCarlo
makedocs(
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico"], edit_link = "master"),
    repo = Remotes.GitHub("rcalxrc08", "FinancialFFT.jl"),
    sitename = "FinancialFFT.jl",
    modules = [FinancialFFT],
    pages = ["index.md", "methods.md", "payoffs.md", "models.md", "ad.md", "api.md"],
)
