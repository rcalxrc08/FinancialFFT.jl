# Option Parameters
using AlternateVectors, MuladdMacro, ChainRulesCore, FinancialMonteCarlo, FinancialToolbox, FinancialFFT
struct CosMethod{num_1 <: Integer} <: FinancialFFT.AbstractIntegralMethod
    N::num_1
    function CosMethod(N::num_1) where {num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        return new{num_1}(N)
    end
end

struct CosMethodResult{num_1, num_2, num_3}
    u::num_1
    v_char_exp::num_2
    uk_adj::num_3
    function CosMethodResult(a::num_1, b::num_2, c::num_3) where {num_1, num_2, num_3}
        return new{num_1, num_2, num_3}(a, b, c)
    end
end

Base.broadcastable(x::CosMethodResult) = Ref(x)

function chi_vectorized(u_el, adj, sincos_1, sincos_2, exp_d)
    sin_uu_adj_1, cos_uu_adj_1 = sincos_1
    sin_uu_adj_2, cos_uu_adj_2 = sincos_2
    res = (cos_uu_adj_1 * exp_d - cos_uu_adj_2 + u_el * (sin_uu_adj_1 * exp_d - sin_uu_adj_2)) / (1 + u_el^2)
    return (1 + !adj) * (res - (sin_uu_adj_1 - sin_uu_adj_2) / (adj + u_el))
end

function uk_call(u, a, b, z)
    bma = b - a
    adjuster = ChainRulesCore.@ignore_derivatives AlternatePaddedVector(true, false, false, false, length(u))
    inv_bma = inv(bma)
    exp_u = exp(b)
    adjj = @. (1 + !adjuster) * b * adjuster
    res = @. inv_bma * (chi_vectorized(u, adjuster, sincos(u * bma), sincos(-u * a), exp_u) - adjj) * exp(FinancialFFT.real_mod(z)), FinancialFFT.imag_mod(z)
    return res
end

using FinancialFFT

function compute_call_price_cos_method_vec(u, x, v_char_exp, uk_adj)
    z = u * x + v_char_exp
    return uk_adj * cos(z)
end

function compute_call_price_cos_method(x, cal_res, opt::EuropeanOption)
    return opt.K * sum(@. compute_call_price_cos_method_vec(cal_res.u, x, cal_res.v_char_exp, cal_res.uk_adj))
end

function compute_call_price_cos_method_bin_vec(u, x, v_char_exp, uk_adj)
    z = u * x + v_char_exp
    sinz, cosz = sincos(z)
    return uk_adj * (cosz + u * sinz)
end

function compute_call_price_cos_method(x, cal_res, ::BinaryEuropeanOption)
    return -sum(@. compute_call_price_cos_method_bin_vec(cal_res.u, x, cal_res.v_char_exp, cal_res.uk_adj))
end

function compute_call_discounted_price_cos_method(S0, driftT_adj, cal_res, opt, df)
    K = opt.K
    x = log(S0 / K) + driftT_adj
    return compute_call_price_cos_method(x, cal_res, opt) * df
end

function compute_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
    price_call = compute_call_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
    return FinancialFFT.call_to_put(price_call, S0_adj, df, opt)
end
function finalize_cos_method(S0_adj, opt, driftT_adj, df, cal_res)
    return compute_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
end
function finalize_cos_method(S0_adj, opt::Array, driftT_adj, df, cal_res)
    return @. compute_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
end

function compute_limits(x::FinancialMonteCarlo.BlackScholesProcess, rT, dT)
    muT = rT - dT
    sigma = x.σ
    cum1 = muT - FinancialFFT.characteristic_exponent_i(1, x) * T
    cum2 = sigma^2 * T
    cum4 = 0
    L = 9
    a = cum1 - L * (sqrt(cum2 + sqrt(cum4)))
    b = cum1 + L * (sqrt(cum2 + sqrt(cum4)))
    return a, b
end
using TaylorSeries
function compute_limits_full(x::FinancialMonteCarlo.BaseProcess, rT, dT)
    muT = rT - dT
    dv = Taylor1(Bool, 3)
    v = 1 + dv
    moments = FinancialFFT.characteristic_exponent_i(v, x) * T
    moment_1 = @views moments[0]
    der_1 = derivative(moments, 1)
    der_3 = derivative(der_1, 2)
    moment_2 = @views der_1[0]
    moment_4 = @views der_3[0]
    cum1 = muT - moment_1
    cum2 = moment_2
    cum4 = moment_4
    L = 9
    a = cum1 - L * (sqrt(cum2 + sqrt(cum4)))
    b = cum1 + L * (sqrt(cum2 + sqrt(cum4)))
    return a, b
end
function compute_limits_full_2(x::FinancialMonteCarlo.BaseProcess, rT, dT)
    muT = rT - dT
    s = 10
    epss = 1e-4
    xtr = inv(s) * (muT + T * characteristic_exponent_i(s, x) - log(epss))
    return -xtr, xtr
end
"""
Documentation CosMethod Method
"""
function cos_method_pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt, T, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    N = method.N
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = ChainRulesCore.@ignore_derivatives compute_limits_full(mcProcess, rT, dT)
    bma = b - a
    u = ChainRulesCore.@ignore_derivatives FinancialFFT.adapt_array(collect((0:N) * (pi / bma)), mode)
    driftT = rT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    v_char_exp = @. FinancialFFT.characteristic_exponent_i(u * im, mcProcess) * T
    uk, v_char_exp_re = uk_call(u, a, b, v_char_exp)
    cal_res = CosMethodResult(u, v_char_exp_re, uk)
    driftT_adj = driftT - a
    df = exp(-rT)
    return finalize_cos_method(S0 * exp(-dT), opt, driftT_adj, df, cal_res)
end

function pricer3(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, abstractPayoffs::Array{U}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = @views abstractPayoffs[index_same_t]
        prices_method = cos_method_pricer(mcProcess, zero_rate, method, payoffs, T, mode)
        @views prices[index_same_t] = prices_method
    end

    return prices
end

function pricer3(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    prices_method = cos_method_pricer(mcProcess, r, method, opt, opt.T, mode)
    return prices_method
end

S0 = 100.0
r = 0.01
d = 0.0
T = 1.1
sigma = 0.2
K = 120.0
N = 2^13
Model = BlackScholesProcess(sigma, Underlying(S0, d));
opt = EuropeanOption(T, K)
method = CosMethod(N)
z_r = ZeroRate(r)
@show blsprice(S0, K, r, T, sigma, d)
@show blsbin(S0, K, r, T, sigma, d)
@show cos_method_pricer(Model, z_r, method, opt, T)
@show cos_method_pricer(Model, z_r, method, BinaryEuropeanOption(T, K), T)
using BenchmarkTools
@btime cos_method_pricer($Model, $z_r, $method, $opt, $T)
KK = collect(100.0 * (1:20) / 10.0)
opts = EuropeanOption.(T, KK)
@btime cos_method_pricer($Model, $z_r, $method, $opts, $T)
using DualNumbers
@show cos_method_pricer(Model, z_r, method, EuropeanOption(T, dual(K, 1.0)), T)
using CUDA
cuda_mode = FinancialMonteCarlo.CudaMode()
CUDA.allowscalar(false)
@btime cos_method_pricer($Model, $z_r, $method, $opts, $T, $cuda_mode)

function blsprice_cos(S0, K, r, T, sigma, d)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = EuropeanOption(T, K)
    N = 2^13
    method = CosMethod(N)
    z_r = ZeroRate(r)
    return cos_method_pricer(Model, z_r, method, opt, T)
end
function blsprice_cos_cu(S0, K, r, T, sigma, d)
    Model = BlackScholesProcess(sigma, Underlying(S0, d))
    opt = EuropeanOption(T, K)
    N = 2^13
    method = CosMethod(N)
    cuda_mode = FinancialMonteCarlo.CudaMode()
    z_r = ZeroRate(r)
    return cos_method_pricer(Model, z_r, method, opt, T, cuda_mode)
end
function blsprice_cos_cu(x)
    return blsprice_cos_cu(x...)
end
function blsprice_cos(x)
    return blsprice_cos(x...)
end
# using Zygote
# @show "Zygote"
# @btime Zygote.gradient(blsprice_cos, $S0, $K, $r, $T, $sigma, $d);
# @btime Zygote.gradient(blsprice_cos_cu, $S0, $K, $r, $T, $sigma, $d);

# using ForwardDiff
# @show "ForwardDiff"
# inputs_fwd_diff = [S0, K, r, T, sigma, d]
# @btime ForwardDiff.gradient(blsprice_cos, $inputs_fwd_diff);
# @btime ForwardDiff.gradient(blsprice_cos_cu, $inputs_fwd_diff);