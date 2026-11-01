using Documenter, FinancialFFT, FinancialMonteCarlo
# Documenter takes the logo and the favicon from the assets: etc/ holds the only copies.
for asset in ("logo.png", "favicon.ico")
    cp(joinpath(@__DIR__, "..", "etc", asset), joinpath(@__DIR__, "src", "assets", asset), force = true)
end
makedocs(
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico"], edit_link = "master"),
    repo = Remotes.GitHub("rcalxrc08", "FinancialFFT.jl"),
    sitename = "FinancialFFT.jl",
    modules = [FinancialFFT],
    pages = ["index.md", "methods.md", "payoffs.md", "models.md", "ad.md", "api.md"],
)
