# Figures of the documentation, computed with the package at build time and saved as SVG in src/assets/figures, in a
# light and a dark variant: the stylesheet assets/figures.css shows the one of the theme of the reader.
using FinancialFFT, FinancialMonteCarlo, FinancialToolbox, ForwardDiff, Plots
# GR draws without a display on the CI runner
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
gr()

const FIGURES = mkpath(joinpath(@__DIR__, "src", "assets", "figures"))

# Categorical slots in fixed order (validated for colour vision deficiencies on adjacent pairs, in both modes), and
# the ink of the text, of the axes and of the grid.
const THEMES = (light = (series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], text = "#1f2328", muted = "#5b6470", axis = "#b9b4ab", grid = "#e6e3de"), dark = (series = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"], text = "#e6e8eb", muted = "#9aa3ad", axis = "#4a525c", grid = "#2a2f36"))
# The secondary encoding of the series: the light slots 3-5 are below 3:1 contrast on the surface.
const MARKERS = [:circle, :rect, :utriangle, :diamond, :dtriangle, :star5]

function base_plot(theme; kwargs...)
    return plot(; size = (760, 430), background_color = :transparent, foreground_color_text = theme.muted, foreground_color_guide = theme.muted, foreground_color_axis = theme.axis, foreground_color_border = theme.axis, foreground_color_legend = nothing, legend_background_color = :transparent, legendfontcolor = theme.text, titlefontcolor = theme.text, gridcolor = theme.grid, gridalpha = 1, gridlinewidth = 1, framestyle = :axes, fontfamily = "sans-serif", titlefontsize = 12, guidefontsize = 10, tickfontsize = 9, legendfontsize = 9, titlelocation = :left, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm, kwargs...)
end

function save_themes(draw, name)
    for (theme_name, theme) in pairs(THEMES)
        savefig(draw(theme), joinpath(FIGURES, "$name-$theme_name.svg"))
    end
end

const S0, r, d = 100.0, 0.02, 0.01
const zero_rate = ZeroRate(r)
underlying() = Underlying(S0, d)
# Parameters chosen for visibly different shapes, not calibrated to a market.
showcase_models() = ["Black Scholes" => BlackScholesProcess(0.2, underlying()), "Merton" => MertonProcess(0.15, 0.5, -0.1, 0.15, underlying()), "Kou" => KouProcess(0.15, 1.0, 0.3, 20.0, 10.0, underlying()), "Variance Gamma" => VarianceGammaProcess(0.2, -0.15, 0.2, underlying()), "NIG" => NormalInverseGaussianProcess(0.15, -0.3, 0.5, underlying()), "Heston" => HestonProcess(0.4, 0.2, 0.0, 1.5, -0.7, 0.04, underlying())]

# Implied volatility smiles: one CosMethod smile per model, inverted with blsimpv.
function smiles_figure(theme)
    T = 1.0
    strikes = collect(range(60.0, 150.0, length = 91))
    p = base_plot(theme; title = "Implied volatility at T = 1, from one smile per model", xlabel = "strike", ylabel = "implied volatility", legend = :outerright, size = (900, 430))
    for (j, (name, model)) in enumerate(showcase_models())
        calls = strikes .>= S0
        prices = pricer(model, zero_rate, CosMethod(2^10), FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, calls))
        vols = blsimpv.(S0, strikes, r, T, prices, d, calls)
        # markers every 15 strikes only: the line carries the data, the markers the identity
        plot!(p, strikes, vols; label = nothing, color = theme.series[j], linewidth = 2)
        scatter!(p, strikes[1:15:end], vols[1:15:end]; label = name, color = theme.series[j], marker = MARKERS[j], markersize = 5, markerstrokewidth = 0)
    end
    plot!(p; ylims = (0.12, 0.36))
    return p
end

# Absolute error against the formula of Black and Scholes, as a function of the number of evaluations of the
# characteristic exponent.
function convergence_figure(theme)
    model = BlackScholesProcess(0.2, underlying())
    T, K = 1.0, 105.0
    option = EuropeanOption(T, K)
    exact = blsprice(S0, K, r, T, 0.2, d)
    floor_error = 1e-16
    error(method) = max(abs(pricer(model, zero_rate, method, option) - exact), floor_error)
    families = [
        # LewisMethod evaluates the exponent on the nodes in [0, A] only
        "LewisMethod(100, N)" => [(n ÷ 2, LewisMethod(100.0, n)) for n in 2 .^ (4:14)],
        "CosMethod(N)" => [(n + 1, CosMethod(n)) for n in 2 .^ (3:12)],
        "CarrMadanLewisMethod(600, n)" => [(2^n, CarrMadanLewisMethod(600.0, n)) for n = 8:20],
        "CarrMadanMethod(400, n)" => [(2^n, CarrMadanMethod(400.0, n)) for n = 8:20],
        "FFTDensityInverter(n, 2¹⁰)" => [(2^n, FFTDensityInverter(n)) for n = 6:18],
        "CosDensityMethod(N, 2¹⁰)" => [(n + 1, CosDensityMethod(n)) for n in 2 .^ (3:10)]
    ]
    p = base_plot(theme; title = "Error of a call (K = 105, T = 1) against Black and Scholes", xlabel = "evaluations of the characteristic exponent", ylabel = "absolute error", xscale = :log10, yscale = :log10, legend = :outerright, size = (900, 430), yticks = 10.0 .^ (-16:2:0), xticks = 10.0 .^ (1:6), ylims = (1e-16, 1e1))
    for (j, (name, family)) in enumerate(families)
        x = first.(family)
        y = [error(method) for (_, method) in family]
        plot!(p, x, y; label = name, color = theme.series[j], linewidth = 2, marker = MARKERS[j], markersize = 5, markerstrokewidth = 0)
    end
    return p
end

# Densities of the log-return with FFTDensityInverter, on a logarithmic scale that shows the tails.
function densities_figure(theme)
    T = 1.0
    y = range(-1.2, 0.8, length = 801)
    p = base_plot(theme; title = "Density of log(S_T / S_0) at T = 1, FFTDensityInverter()", xlabel = "log-return", ylabel = "density", yscale = :log10, ylims = (1e-6, 10), yticks = 10.0 .^ (-6:1), legend = :outerright, size = (900, 430))
    for (j, (name, model)) in enumerate(showcase_models())
        f = FinancialFFT.density(model, zero_rate, FFTDensityInverter(), T)
        values = f.(y)
        plot!(p, y, values; label = nothing, color = theme.series[j], linewidth = 2)
        scatter!(p, y[41:80:end], values[41:80:end]; label = name, color = theme.series[j], marker = MARKERS[j], markersize = 5, markerstrokewidth = 0)
    end
    return p
end

# Scores of the volatility, ∂σ log f(y), from ForwardDiff through FFTDensityInverter: the weights of the likelihood
# ratio method. Dashed, the Black Scholes score through CosDensityMethod.
function scores_figure(theme)
    T = 1.0
    y = range(-0.8, 0.6, length = 401)
    models = ["Black Scholes" => σ -> BlackScholesProcess(σ, underlying()), "Merton" => σ -> MertonProcess(σ, 0.5, -0.1, 0.15, underlying()), "Heston (σ₀)" => σ -> HestonProcess(0.4, σ, 0.0, 1.5, -0.7, 0.04, underlying())]
    σ = 0.2
    p = base_plot(theme; title = "Score ∂σ log f(y) at T = 1, ForwardDiff through FFTDensityInverter()", xlabel = "log-return", ylabel = "score", legend = :outerright, size = (900, 430))
    for (j, (name, model)) in enumerate(models)
        scores = ForwardDiff.derivative(s -> log.(FinancialFFT.density_values(FinancialFFT.density(model(s), zero_rate, FFTDensityInverter(), T), collect(y))), σ)
        plot!(p, y, scores; label = nothing, color = theme.series[j], linewidth = 2)
        scatter!(p, y[21:40:end], scores[21:40:end]; label = name, color = theme.series[j], marker = MARKERS[j], markersize = 5, markerstrokewidth = 0)
    end
    # the same score from the cosine expansion of the density
    model = last(first(models))
    cos_scores = ForwardDiff.derivative(s -> log.(FinancialFFT.density_values(FinancialFFT.density(model(s), zero_rate, CosDensityMethod(), T), collect(y))), σ)
    plot!(p, y, cos_scores; label = "Black Scholes, CosDensityMethod", color = theme.muted, linewidth = 1.5, linestyle = :dash)
    return p
end

# The Chernoff bound of the lower tail of the log-return, t(s) = (κ(s) - log ε) / s for the exponent κ(s) = log E[exp(-s X)]:
# every s gives a valid distance, the range uses the minimum (marked, as the search of the package finds it). The curves
# stop at the exponential moment bound, where κ, and t, become infinite.
function chernoff_figure(theme)
    T, log_ε = 1.0, log(1e-14)
    p = base_plot(theme; title = "Chernoff bound of the lower tail at T = 1, ε = 1e-14", xlabel = "s", ylabel = "t(s), distance below the drift", xscale = :log10, xlims = (1, 100), xticks = ([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"]), ylims = (0, 14), legend = :outerright, size = (900, 430))
    for (j, (name, model)) in enumerate(showcase_models())
        exponent = FinancialFFT.TailExponent(model, T, -1)
        s_end = min(FinancialFFT.exponential_moment_bound(exponent) * (1 - 1e-9), 100.0)
        s = exp.(range(0.0, log(s_end), length = 600))
        plot!(p, s, FinancialFFT.compute_ts.(s, Ref(exponent), log_ε); label = nothing, color = theme.series[j], linewidth = 2)
        s_opt = FinancialFFT.chernoff_optimum(exponent, log_ε, 1.0, sqrt(eps()))
        scatter!(p, [s_opt], [FinancialFFT.compute_ts(s_opt, exponent, log_ε)]; label = name, color = theme.series[j], marker = MARKERS[j], markersize = 6, markerstrokewidth = 0)
    end
    return p
end

save_themes(smiles_figure, "smiles")
save_themes(convergence_figure, "convergence")
save_themes(densities_figure, "densities")
save_themes(scores_figure, "scores")
save_themes(chernoff_figure, "chernoff")
