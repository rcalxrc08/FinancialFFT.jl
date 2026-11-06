using FinancialFFT, ChainRulesCore, PatternVectors

function FK(k_adj, drift_rn, mcProcess, T, bma)
    k_im = k_adj * im
    FK = FinancialFFT.exp_mod(FinancialFFT.characteristic_exponent_i(k_im, mcProcess) * T + k_im * drift_rn)
    return FK / bma
end
struct CosMethodDensityResult{num_1, num_2}
    FK::num_1
    k_vec::num_2
    function CosMethodDensityResult(a::num_1, b::num_2) where {num_1, num_2}
        return new{num_1, num_2}(a, b)
    end
end

Base.broadcastable(x::CosMethodDensityResult) = Ref(x)

function finalize_cos_method_impl(x, k_vec, FKK)
    return sum(@. FKK * cos(k_vec * x))
end

function finalize_cos_method(x_val, log_S0_K_a, FK_res)
    x = log_S0_K_a + x_val
    return finalize_cos_method_impl(x, FK_res.k_vec, FK_res.FK)
end

function finalize_cos_method_vec(x_val::AbstractArray, log_S0_K_a, FK_res)
    return @. finalize_cos_method(x_val, log_S0_K_a, FK_res)
end
function finalize_cos_method_vec(x_val, log_S0_K_a, FK_res)
    return finalize_cos_method(x_val, log_S0_K_a, FK_res)
end

function cos_method_density(x_val, mcProcess, r, T, N, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode(), St = mcProcess.underlying.S0)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = ChainRulesCore.@ignore_derivatives FinancialFFT.compute_chernorff_limits(mcProcess, rT, dT, T)
    bma = ChainRulesCore.@ignore_derivatives b - a
    drift_rn = rT - dT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    drift_adj = drift_rn - Float32(a)
    one_int = ChainRulesCore.@ignore_derivatives UInt8(1)
    two_int = ChainRulesCore.@ignore_derivatives UInt8(2)
    av_vec = ChainRulesCore.@ignore_derivatives PatternVector(N + 1, InitialValuePattern(one_int, two_int))
    pi_adj = ChainRulesCore.@ignore_derivatives pi / bma
    k_vec = ChainRulesCore.@ignore_derivatives FinancialFFT.adapt_array(collect(0:N) .* pi_adj, mode)
    FKK = @. av_vec * FK(k_vec, drift_adj, mcProcess, T, Float32(bma))

    FK_res = CosMethodDensityResult(FKK, k_vec)
    log_S0_K_a = -log(St) - a
    @show FK_res
    @show typeof(bma)
    @show typeof(FKK)
    return finalize_cos_method_vec(x_val, log_S0_K_a, FK_res)
end

function pricer_density_cos(mcProcess, r, opt, N, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = FinancialFFT.compute_chernorff_limits(mcProcess, rT, dT, T)
    log_S0 = log(S0)
    x_min = log_S0 + a
    x_max = log_S0 + b
    dx = (x_max - x_min) / N
    xvals = ChainRulesCore.@ignore_derivatives FinancialFFT.adapt_array(collect(x_min:dx:x_max), mode)
    # @show length(xvals)
    density_x = cos_method_density(xvals, mcProcess, r, T, N, mode)
    K = opt.K
    payoff = @. max(exp(xvals) - K, 0.0)
    price = sum(@. payoff * density_x * dx)
    return price * exp(-rT)
end

function log_density_default2(x)
    return (x^2 + log(2 * π)) / (-2)
end
function log_density2(x_in, mu, sigma)
    x = (x_in - mu) / sigma
    return log_density_default2(x) - log(sigma)
end
function Density_bs(x_val, mcProcess, r, T)
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    drift_rn = log(S0) + rT - dT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    std_dev = mcProcess.σ * sqrt(T)
    return exp(log_density2(x_val, drift_rn, std_dev))
end

using FinancialMonteCarlo, PatternVectors
epss = 1e-14
S0 = 100.0
r = 0.02
d = 0.01
T = 1.1
sigma = 0.2
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
# Model = BlackScholesProcess(sigma, Underlying(S0, d));
Model = MertonProcess(sigma, lam, mu1, sigma1, FinancialMonteCarlo.Underlying(S0, d))

x_val = 0.0
K = 1.0
N = 2^10
T = 1.0
rf = ZeroRate(r)
@show cos_method_density(x_val, Model, rf, T, N);
x_vals = zeros(Float64, 100)
@show cos_method_density(x_vals, Model, rf, T, N);

using BenchmarkTools
# @btime cos_method_density($x_val, $Model, $rf, $T, $N);
# @btime cos_method_density($x_vals, $Model, $rf, $T, $N);

rf = ZeroRate(Float32(r))
Model2 = BlackScholesProcess(Float32(sigma), Underlying(Float32(S0), Float32(d)));
@show pricer_density_cos(Model2, rf, EuropeanOption(Float32(T), Float32(100.0)), N)
@show pricer_density_cos(Model2, rf, EuropeanOption(Float32(T), Float32(100.0)), N, new_mode)
@show blsprice(S0, 100.0, r, T, sigma, d)