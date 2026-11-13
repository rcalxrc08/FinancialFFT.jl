using Documenter, FinancialFFT, FinancialMonteCarlo
# Documenter takes the logo and the favicon from the assets: etc/ holds the only copies, and git keeps no empty folder.
assets = mkpath(joinpath(@__DIR__, "src", "assets"))
for asset in ("logo.png", "favicon.ico")
    cp(joinpath(@__DIR__, "..", "etc", asset), joinpath(assets, asset), force = true)
end
# The figures are computed with the package, in a light and a dark variant (see figures.jl and assets/figures.css), in
# a module of their own: Plots exports a density, which would make the one of FinancialFFT ambiguous in Main.
module DocumentationFigures
include(joinpath(@__DIR__, "figures.jl"))
end
makedocs(format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico", "assets/figures.css", "assets/style.css", "assets/site.css", "assets/home.css", "assets/dark.js"], edit_link = "master"), repo = Remotes.GitHub("rcalxrc08", "FinancialFFT.jl"), sitename = "FinancialFFT.jl", modules = [FinancialFFT], pages = ["Home" => "index.md", "Guides" => ["Methods" => "methods.md", "Payoffs" => "payoffs.md", "Models and curves" => "models.md", "Automatic differentiation" => "ad.md", "Calibration" => "calibration.md"], "Reference" => ["API" => "api.md", "References" => "references.md"]])
