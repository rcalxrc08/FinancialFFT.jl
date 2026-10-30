# Capabilities and tolerances of each method against the Black Scholes closed formulas.
# A payoff missing from `payoffs` is not supported by the method (no `pricer` method for it).
# `Broken(...)` marks combinations that run but are known to fail: see the manual, "Automatic differentiation".

# `dual` is either one tolerance or a NamedTuple with one tolerance per Black Scholes input (see BS_ARG_NAMES).
tolerances(; price, dual = price, hyper = dual, taylor = hyper, forwarddiff = dual, reversediff = dual, zygote = dual, mooncake = dual) = (; price, dual, hyper, taylor, forwarddiff, reversediff, zygote, mooncake)
dual_tolerance(toll, name) = toll isa NamedTuple ? toll[name] : toll

# The TaylorSeries check includes the second derivative in σ, hence its looser tolerances.
const lewis_tolerances = tolerances(price = 1e-10, taylor = 1e-9)
const cos_tolerances = tolerances(price = 1e-10, taylor = 1e-9, reversediff = 1e-9, zygote = 1e-9)
const cml_tolerances = tolerances(price = 1e-5, taylor = 1e-4, zygote = Broken())
const cml_binary_tolerances = tolerances(price = Broken(1e-5), zygote = Broken())
const cm_tolerances = tolerances(price = 1e-5, hyper = 1e-4, zygote = Broken(1e-5))
const density_tolerances = tolerances(price = 1e-7, taylor = Broken(1e-6))
const density_binary_tolerances = tolerances(price = 1e-4, taylor = Broken(1e-4))

const METHOD_CASES = [
    (name = "LewisMethod", method = LewisMethod(600.0, 20000), payoffs = Dict(kind => lewis_tolerances for kind in PAYOFF_KINDS)),
    (name = "CosMethod", method = CosMethod(2^12), payoffs = Dict(kind => cos_tolerances for kind in PAYOFF_KINDS)),
    (name = "CarrMadanLewisMethod", method = CarrMadanLewisMethod(600.0, 16), payoffs = Dict(:european => cml_tolerances, :european_smile => cml_tolerances, :binary => cml_binary_tolerances, :binary_smile => cml_binary_tolerances)),
    (name = "CarrMadanMethod", method = CarrMadanMethod(400.0, 18), payoffs = Dict(:european => cm_tolerances)),
    (name = "DensityInverter", method = FinancialFFT.DensityInverter(18), payoffs = Dict(:european => density_tolerances, :binary => density_binary_tolerances)),
]

# Non Black Scholes models, with the parameters used across the package examples.
model_cases(S0, d) = [
    "Merton" => MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(S0, d)),
    "Kou" => KouProcess(0.2, 5.0, 0.3, 30.0, 20.0, Underlying(S0, d)),
    "VarianceGamma" => VarianceGammaProcess(0.2, 0.03, 0.01, Underlying(S0, d)),
    "NormalInverseGaussian" => NormalInverseGaussianProcess(0.2, 0.01, 0.01, Underlying(S0, d)),
    "Heston" => HestonProcess(0.3, 0.2, 0.0, 1.5, -0.7, 0.04, Underlying(S0, d)),
]
