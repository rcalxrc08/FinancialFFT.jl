# The benchmark suite, a BenchmarkTools group, run by benchmark/run.jl (locally and in the Benchmark workflow).
#   Black Scholes / method / payoff / back-end    prices and gradients in (S0, r, σ), a single option and a smile
#   Models / model / method                        prices of a single option
#   Densities / method / task                      setup and evaluation at 1000 points
#   Chernoff range / model / number type           the truncation range of COS and of the densities
# It is built in parts, `suite(part)` for each of `PARTS`: the Black Scholes group of each method, the models and the
# densities. run.jl runs each part in its own process, since the code compiled for the gradients (Zygote, Mooncake) is
# never freed: in one process the whole suite needs about 7 GB, more than a CI runner has.
# A method that the package does not define is skipped, so that the suite also runs on older commits (the base of a
# pull request).
using BenchmarkTools, FinancialFFT, FinancialMonteCarlo, ForwardDiff, TaylorSeries, Zygote, Mooncake

# The group at the path keys, created when missing.
group!(group, keys...) = foldl((g, k) -> haskey(g, k) ? g[k] : (g[k] = BenchmarkGroup()), keys; init = group)

const S0, r, d, T, σ = 100.0, 0.02, 0.01, 1.0, 0.2
const Ks = collect(range(60.0, 160.0, length = 101))

method_constructors = [
    :LewisMethod => () -> LewisMethod(600.0, 20000),
    :CosMethod => () -> CosMethod(2^12),
    :CarrMadanLewisMethod => () -> CarrMadanLewisMethod(600.0, 16),
    :CarrMadanMethod => () -> CarrMadanMethod(400.0, 18),
    :FFTDensityInverter => () -> FFTDensityInverter(),
    :CosDensityMethod => () -> CosDensityMethod(),
]
const METHODS = [string(name) => constructor() for (name, constructor) in method_constructors if isdefined(FinancialFFT, name)]
const PARTS = vcat(["Black Scholes / " * name for (name, _) in METHODS], ["Models", "Densities", "Chernoff range"])
const PAYOFFS = ["option" => EuropeanOption(T, 105.0), "smile of 101" => FinancialMonteCarlo.EuropeanOptionSmile(T, Ks, Ks .>= S0)]

# The price, summed over the strikes of a smile, as a function of (S0, r, σ).
function add_black_scholes!(group, method, payoff)
    price = x -> sum(pricer(BlackScholesProcess(x[3], Underlying(x[1], d)), ZeroRate(x[2]), method, payoff))
    x = [S0, r, σ]
    group["price"] = @benchmarkable $price($x)
    group["ForwardDiff"] = @benchmarkable ForwardDiff.gradient($price, $x)
    group["Zygote"] = @benchmarkable Zygote.gradient($price, $x)
    cache = Mooncake.prepare_gradient_cache(price, x)
    group["Mooncake"] = @benchmarkable Mooncake.value_and_gradient!!($cache, $price, $x)
    return group
end

const MODELS = [
    "Merton" => MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(S0, d)),
    "Kou" => KouProcess(0.2, 5.0, 0.3, 30.0, 20.0, Underlying(S0, d)),
    "VarianceGamma" => VarianceGammaProcess(0.2, 0.03, 0.01, Underlying(S0, d)),
    "NormalInverseGaussian" => NormalInverseGaussianProcess(0.2, 0.01, 0.01, Underlying(S0, d)),
    "Heston" => HestonProcess(0.3, 0.2, 0.0, 1.5, -0.7, 0.04, Underlying(S0, d)),
]

# The parameters of the models of the "Chernoff range" part, and the models from them: Float64 values, ForwardDiff duals
# (a gradient in every parameter) or TaylorN (order 2 in every parameter, the Hessian).
const RANGE_MODELS = [
    "Merton" => ((σ, λ, μ, δ, S) -> MertonProcess(σ, λ, μ, δ, Underlying(S, d)), [0.2, 5.0, 0.03, 0.02, S0]),
    "Heston" => ((σ, σ₀, κ, θ, S) -> HestonProcess(σ, σ₀, 0.0, κ, -0.7, θ, Underlying(S, d)), [0.3, 0.2, 1.5, 0.04, S0]),
]
function with_derivatives(x, number_type)
    number_type == "Float64" && return x
    n = length(x)
    number_type == "ForwardDiff" && return [ForwardDiff.Dual{Nothing}(x[i], ntuple(j -> float(i == j), n)...) for i in eachindex(x)]
    # TaylorSeries 0.23 renamed set_variables to variables!
    variables = isdefined(TaylorSeries, :variables!) ? TaylorSeries.variables!("x", numvars = n, order = 2) : TaylorSeries.set_variables("x", numvars = n, order = 2)
    return x .+ variables
end

# The part of the suite named part (one of PARTS), at its place in the whole suite.
function suite(part)
    SUITE = BenchmarkGroup()
    if startswith(part, "Black Scholes / ")
        method_name = chopprefix(part, "Black Scholes / ")
        method = Dict(METHODS)[method_name]
        for (payoff_name, payoff) in PAYOFFS
            add_black_scholes!(group!(SUITE, "Black Scholes", method_name, payoff_name), method, payoff)
        end
    elseif part == "Models"
        for (model_name, model) in MODELS, (method_name, method) in METHODS
            group!(SUITE, "Models", model_name)[method_name] = @benchmarkable pricer($model, $(ZeroRate(r)), $method, $(EuropeanOption(T, 105.0)))
        end
    elseif part == "Densities"
        for (method_name, method) in METHODS
            method isa FinancialFFT.AbstractDensityMethod || continue
            model = BlackScholesProcess(σ, Underlying(S0, d))
            f = density(model, ZeroRate(r), method, T)
            y = collect(range(-1.0, 1.0, length = 1000))
            group = group!(SUITE, "Densities", method_name)
            group["setup"] = @benchmarkable density($model, $(ZeroRate(r)), $method, $T)
            group["f(y), 1000 points"] = @benchmarkable map($f, $y)
            group["density_values, 1000 points"] = @benchmarkable FinancialFFT.density_values($f, $y)
        end
    elseif part == "Chernoff range"
        for (model_name, (model_of, x)) in RANGE_MODELS, number_type in ("Float64", "ForwardDiff", "TaylorN")
            model = model_of(with_derivatives(x, number_type)...)
            group!(SUITE, "Chernoff range", model_name)[number_type] = @benchmarkable FinancialFFT.compute_chernorff_limits($model, $(r * T), $(d * T), $T)
        end
    else
        throw(ArgumentError("unknown part $part, not in $PARTS"))
    end
    return SUITE
end
