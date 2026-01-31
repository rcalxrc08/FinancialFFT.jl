# Option Parameters

struct CosMethod{num_1 <: Integer} <: FinancialFFT.AbstractIntegralMethod
    N::num_1
    function CosMethod(N::num_1) where {num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        return new{num_1}(N)
    end
end
export CosMethod
struct CosMethodResult{num_1, num_2, num_3}
    u::num_1
    v_char_exp::num_2
    uk_adj::num_3
    function CosMethodResult(a::num_1, b::num_2, c::num_3) where {num_1, num_2, num_3}
        return new{num_1, num_2, num_3}(a, b, c)
    end
end

Base.broadcastable(x::CosMethodResult) = Ref(x)

function chi_vectorized(u_el, adj, bma, a, exp_u)
    sin_uu_adj_1, cos_uu_adj_1 = sincos(u_el * bma)
    sin_uu_adj_2, cos_uu_adj_2 = sincos(u_el * a)
    res = (cos_uu_adj_1 * exp_u - cos_uu_adj_2 + u_el * (sin_uu_adj_1 * exp_u + sin_uu_adj_2)) / (1 + u_el^2)
    return (1 + !adj) * (res - (sin_uu_adj_1 + sin_uu_adj_2) / (adj + u_el))
end

function uk_call(u, a, b, z)
    bma = b - a
    adjuster = ChainRulesCore.@ignore_derivatives AlternatePaddedVector(true, false, false, false, length(u))
    inv_bma = inv(bma)
    exp_u = exp(b)
    adjj = @. (1 + !adjuster) * b * adjuster
    res = @. inv_bma * (chi_vectorized(u, adjuster, bma, a, exp_u) - adjj) * exp(FinancialFFT.real_mod(z)), FinancialFFT.imag_mod(z)
    return res
end

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

function compute_call_price_cos_method_vec_smile(u, x, v_char_exp, uk_adj)
    return @. (uk_adj * cos(u * x' + v_char_exp))'
end

function compute_call_price_cos_method_smile(x, cal_res, opt::FinancialMonteCarlo.EuropeanOptionSmile, df)
    t_1 = @views sum(compute_call_price_cos_method_vec_smile(cal_res.u, x, cal_res.v_char_exp, cal_res.uk_adj), dims = 2)[:]
    return @. opt.K * t_1 * df
end

function compute_call_discounted_price_cos_method(S0, driftT_adj, cal_res, opt::FinancialMonteCarlo.EuropeanOptionSmile, df)
    x = @. log(S0 / opt.K) + driftT_adj
    return compute_call_price_cos_method_smile(x, cal_res, opt, df)
end

function compute_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
    price_call = compute_call_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
    return FinancialFFT.call_to_put(price_call, S0_adj, df, opt)
end
function finalize_cos_method(S0_adj, opt, driftT_adj, df, cal_res)
    return compute_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
end
function finalize_cos_method(S0_adj, opt::AbstractArray, driftT_adj, df, cal_res)
    return @. compute_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
end

function compute_chernorff_limits(Model::FinancialMonteCarlo.BaseProcess, rT, dT, T)
    drift = rT - dT
    epss = 1e-12
    s_min = 1.0
    s_max = 50.0
    x_toll_bisection = 1e-14
    compute_chernoff_extrema_bisection(Model, T, epss, drift, s_min, s_max, x_toll_bisection)
end
"""
Documentation CosMethod Method
"""
function cos_method_pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt, T, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    N = method.N
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = ChainRulesCore.@ignore_derivatives compute_chernorff_limits(mcProcess, rT, dT, T)
    bma = b - a
    u = ChainRulesCore.@ignore_derivatives FinancialFFT.adapt_array(collect((0:N) * (pi / bma)), mode)
    driftT = rT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    v_char_exp = @. FinancialFFT.characteristic_exponent_i(u * im, mcProcess) * T
    uk, v_char_exp_im = uk_call(u, a, b, v_char_exp)
    cal_res = CosMethodResult(u, v_char_exp_im, uk)
    driftT_adj = driftT - a
    df = exp(-rT)
    return finalize_cos_method(S0 * exp(-dT), opt, driftT_adj, df, cal_res)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, abstractPayoffs::Array{U}, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = @views abstractPayoffs[index_same_t]
        prices_method = cos_method_pricer(mcProcess, zero_rate, method, payoffs, T, mode)
        @views prices[index_same_t] .= prices_method
    end

    return prices
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    prices_method = cos_method_pricer(mcProcess, r, method, opt, opt.T, mode)
    return prices_method
end
