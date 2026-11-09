# Timings of the AD back-ends, as multiples of the time of one price, for the tables of docs/src/ad.md (issue #61):
#     julia --project=benchmark benchmark/ad.jl
# Two parts:
#   - every method: the gradient of a Black–Scholes price in (S0, r, σ);
#   - Heston with CosMethod, by task: the gradient of a price and of the sum of a smile in the 7 inputs
#     (S0, r, σ, σ₀, κ, ρ, θ), the Jacobian of the smile (101 × 7), the gradient of the sum of the smile in its 101
#     strikes, the Hessian of a price (7 × 7).
# DualNumbers takes one pass per input, HyperDualNumbers one per pair of inputs (its ε₁ε₂ part), TaylorN one pass for
# all of them (order 1 for first derivatives, 2 for the Hessian). Mooncake and the compiled ReverseDiff tape are prepared
# once, outside the timing. The ratios hold across machines better than the times, which are printed too.
# AD_BACKENDS=Enzyme (a substring of the names) measures only the matching back-ends and leaves the other cells empty.
using BenchmarkTools, FinancialFFT, FinancialMonteCarlo, ForwardDiff, ReverseDiff, Zygote, Mooncake, DualNumbers, HyperDualNumbers, TaylorSeries, Enzyme

selected(name) = (only = get(ENV, "AD_BACKENDS", ""); isempty(only) || occursin(only, name))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0
const T, d = 1.0, 0.01
const Ks = collect(range(60.0, 160.0, length = 101))
const option = EuropeanOption(T, 105.0)
const smile_payoff = FinancialMonteCarlo.EuropeanOptionSmile(T, Ks, Ks .>= 100.0)

# TaylorSeries 0.23 renamed set_variables to variables!
taylor_variables(n, order) = isdefined(TaylorSeries, :variables!) ? TaylorSeries.variables!("x", numvars = n, order = order) : TaylorSeries.set_variables("x", numvars = n, order = order)
unit(n, i) = [j == i ? 1.0 : 0.0 for j = 1:n]

# The gradient of a scalar function f of a vector x, back-end by back-end: each entry is a function of (f, x) that
# prepares what can be prepared and returns the thunk to time.
const GRADIENTS = [
    "DualNumbers" => (f, x) -> () -> [DualNumbers.epsilon(f(DualNumbers.Dual.(x, unit(length(x), i)))) for i in eachindex(x)],
    "HyperDualNumbers" => (f, x) -> () -> [f(HyperDualNumbers.Hyper.(x, unit(length(x), i), 0.0, 0.0)).epsilon1 for i in eachindex(x)],
    "TaylorN" => (f, x) -> (v = taylor_variables(length(x), 1); () -> TaylorSeries.gradient(f(x .+ v))),
    "ForwardDiff" => (f, x) -> (config = ForwardDiff.GradientConfig(f, x); () -> ForwardDiff.gradient(f, x, config)),
    "ReverseDiff" => (f, x) -> () -> ReverseDiff.gradient(f, x),
    "ReverseDiff, compiled tape" => (f, x) -> (tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, x)); out = similar(x); () -> ReverseDiff.gradient!(out, tape, x)),
    "Zygote" => (f, x) -> () -> Zygote.gradient(f, x),
    "Mooncake" => (f, x) -> (cache = Mooncake.prepare_gradient_cache(f, x); () -> Mooncake.value_and_gradient!!(cache, f, x)),
    "Enzyme, forward" => (f, x) -> () -> Enzyme.gradient(Enzyme.Forward, Const(f), x),
    "Enzyme, reverse" => (f, x) -> () -> Enzyme.gradient(Enzyme.Reverse, Const(f), x),
]

# The time of thunk(), or nothing if it fails.
function measure(thunk)
    try
        thunk()
        return @belapsed $thunk()
    catch err
        @warn "failed" exception = (err, catch_backtrace())
        return nothing
    end
end
ratio(t, t0) = isnothing(t) ? "fails" : string(round(t / t0; sigdigits = 2))

const METHODS = ["LewisMethod(600, 20000)" => LewisMethod(600.0, 20000), "CosMethod(2^12)" => CosMethod(2^12), "CarrMadanLewisMethod(600, 16)" => CarrMadanLewisMethod(600.0, 16), "CarrMadanMethod(400, 18)" => CarrMadanMethod(400.0, 18), "CosDensityMethod()" => CosDensityMethod(), "FFTDensityInverter()" => FFTDensityInverter()]

function methods_table()
    names = first.(GRADIENTS)
    println("| Method | price | ", join(names, " | "), " |")
    println("|:--|--:|", repeat("--:|", length(names)))
    for (method_name, method) in METHODS
        price(x) = pricer(BlackScholesProcess(x[3], Underlying(x[1], d)), ZeroRate(x[2]), method, option)
        x = [100.0, 0.02, 0.2]
        t0 = @belapsed $price($x)
        cells = [selected(name) ? ratio(measure(gradient(price, x)), t0) : "" for (name, gradient) in GRADIENTS]
        println("| `", method_name, "` | ", BenchmarkTools.prettytime(t0 * 1e9), " | ", join(cells, " | "), " |")
        flush(stdout)
    end
end

# Heston with CosMethod: x = (S0, r, σ, σ₀, κ, ρ, θ).
const COS = CosMethod(2^10)
heston(x) = HestonProcess(x[3], x[4], zero(x[3]), x[5], x[6], x[7], Underlying(x[1], d))
const X = [100.0, 0.02, 0.3, 0.2, 1.5, -0.7, 0.04]
heston_price(x) = pricer(heston(x), ZeroRate(x[2]), COS, option)
heston_smile(x) = pricer(heston(x), ZeroRate(x[2]), COS, smile_payoff)
heston_smile_sum(x) = sum(heston_smile(x))
smile_of_strikes(K) = sum(pricer(heston(X), ZeroRate(X[2]), COS, FinancialMonteCarlo.EuropeanOptionSmile(T, K, Ks .>= 100.0)))

const JACOBIANS = [
    "DualNumbers" => (f, x) -> () -> reduce(hcat, [DualNumbers.epsilon.(f(DualNumbers.Dual.(x, unit(length(x), i)))) for i in eachindex(x)]),
    "TaylorN" => (f, x) -> (v = taylor_variables(length(x), 1); () -> reduce(vcat, transpose.(TaylorSeries.gradient.(f(x .+ v))))),
    "ForwardDiff" => (f, x) -> () -> ForwardDiff.jacobian(f, x),
    "ReverseDiff" => (f, x) -> () -> ReverseDiff.jacobian(f, x),
    "Zygote" => (f, x) -> () -> Zygote.jacobian(f, x),
    "Enzyme, forward" => (f, x) -> () -> Enzyme.jacobian(Enzyme.Forward, Const(f), x),
    "Enzyme, reverse" => (f, x) -> () -> Enzyme.jacobian(Enzyme.Reverse, Const(f), x),
]
const HESSIANS = [
    "HyperDualNumbers" => (f, x) -> () -> [f(HyperDualNumbers.Hyper.(x, unit(length(x), i), unit(length(x), j), 0.0)).epsilon12 for i in eachindex(x), j in eachindex(x) if i <= j],
    "TaylorN" => (f, x) -> (v = taylor_variables(length(x), 2); () -> TaylorSeries.hessian(f(x .+ v))),
    "ForwardDiff" => (f, x) -> () -> ForwardDiff.hessian(f, x),
    "ReverseDiff" => (f, x) -> () -> ReverseDiff.hessian(f, x),
    "Zygote" => (f, x) -> () -> Zygote.hessian(f, x),
]

function tasks_table()
    t_price = @belapsed heston_price($X)
    t_smile = @belapsed heston_smile($X)
    tasks = [
        "gradient of a price, 7 inputs" => (heston_price, X, GRADIENTS, t_price),
        "gradient of the sum of a smile of 101 strikes, 7 inputs" => (heston_smile_sum, X, GRADIENTS, t_smile),
        "Jacobian of a smile, 101 × 7" => (heston_smile, X, JACOBIANS, t_smile),
        "gradient of the sum of a smile in its 101 strikes" => (smile_of_strikes, Ks, GRADIENTS, t_smile),
        "Hessian of a price, 7 × 7" => (heston_price, X, HESSIANS, t_price),
    ]
    names = unique(vcat(first.(GRADIENTS), first.(JACOBIANS), first.(HESSIANS)))
    println("price: ", BenchmarkTools.prettytime(t_price * 1e9), ", smile: ", BenchmarkTools.prettytime(t_smile * 1e9))
    println("| Task | ", join(names, " | "), " |")
    println("|:--|", repeat("--:|", length(names)))
    for (task, (f, x, backends, t0)) in tasks
        available = Dict(backends)
        # TaylorN in 101 variables: 102 coefficients per number, not a sensible choice; DualNumbers and HyperDualNumbers
        # take one pass per strike
        cells = [haskey(available, name) && selected(name) && !(name == "TaylorN" && length(x) > 10) ? ratio(measure(available[name](f, x)), t0) : "" for name in names]
        println("| ", task, " | ", join(cells, " | "), " |")
        flush(stdout)
    end
end

println("## Gradient of a Black–Scholes price in (S0, r, σ), times one price\n")
methods_table()
println("\n## Heston, CosMethod(2^10), times one price (or one smile for the smile tasks)\n")
tasks_table()
