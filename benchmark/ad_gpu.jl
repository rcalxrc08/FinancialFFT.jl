# Timings of the AD back-ends on the GPU (ForwardDiff and Zygote, the two supported there), as multiples of one
# price, for the table of docs/src/ad.md (issue #61). Needs a CUDA GPU, and CUDA.jl in the environment (it is not a
# dependency of the benchmark project, since CI has no GPU: add it to a copy), so it is not run by CI:
#     julia --project=benchmark benchmark/ad_gpu.jl
# Heston with CosMethod(2^10) in CudaMode, inputs (S0, r, σ, σ₀, κ, ρ, θ): the gradient of a price and of the sum of a
# smile of 101 strikes on the device, and the Jacobian of the smile.
using CUDA, BenchmarkTools, FinancialFFT, FinancialMonteCarlo, ForwardDiff, Zygote

CUDA.allowscalar(false)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0
const T, d = 1.0, 0.01
const Ks = collect(range(60.0, 160.0, length = 101))
const COS = CosMethod(2^10; mode = FinancialMonteCarlo.CudaMode())
heston(x) = HestonProcess(x[3], x[4], zero(x[3]), x[5], x[6], x[7], Underlying(x[1], d))
const X = [100.0, 0.02, 0.3, 0.2, 1.5, -0.7, 0.04]
const smile_payoff = FinancialMonteCarlo.EuropeanOptionSmile(T, CuArray(Ks), CuArray(Ks .>= 100.0))
heston_price(x) = pricer(heston(x), ZeroRate(x[2]), COS, EuropeanOption(T, 105.0))
heston_smile(x) = pricer(heston(x), ZeroRate(x[2]), COS, smile_payoff)
heston_smile_sum(x) = sum(heston_smile(x))

timed(thunk) = (thunk(); @belapsed CUDA.@sync $thunk())
ratio(t, t0) = string(round(t / t0; sigdigits = 2))
t_price = timed(() -> heston_price(X))
t_smile = timed(() -> heston_smile(X))
println("price: ", BenchmarkTools.prettytime(t_price * 1e9), ", smile: ", BenchmarkTools.prettytime(t_smile * 1e9))
println("| Task (GPU) | ForwardDiff | Zygote |")
println("|:--|--:|--:|")
for (task, f, t0, forward, reverse) in (("gradient of a price, 7 inputs", heston_price, t_price, ForwardDiff.gradient, Zygote.gradient), ("gradient of the sum of a smile of 101 strikes, 7 inputs", heston_smile_sum, t_smile, ForwardDiff.gradient, Zygote.gradient), ("Jacobian of a smile, 101 × 7", heston_smile, t_smile, ForwardDiff.jacobian, Zygote.jacobian))
    println("| ", task, " | ", ratio(timed(() -> forward(f, X)), t0), " | ", ratio(timed(() -> reverse(f, X)), t0), " |")
end
