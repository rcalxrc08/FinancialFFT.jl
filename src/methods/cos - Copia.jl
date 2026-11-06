
function compute_call_price_cos_method_vec_smile(u, x, v_char_exp, uk_adj)
    return @. uk_adj * cos(u * x' + v_char_exp)
end

function compute_call_price_cos_method_smile(x, cal_res, K)
    t_1 = compute_call_price_cos_method_vec_smile(cal_res.u, x, cal_res.v_char_exp, cal_res.uk_adj)
    res = K .* sum(t_1, dims = 1)'
    return res
end

function compute_call_discounted_price_cos_method(S0, driftT_adj, cal_res, opt::FinancialMonteCarlo.EuropeanOptionSmile, df)
    K = opt.K
    x = @. log(S0 / K) + driftT_adj
    return compute_call_price_cos_method_smile(x, cal_res, K) .* df
end

function finalize_cos_method(S0_adj, opt::FinancialMonteCarlo.EuropeanOptionSmile, driftT_adj, df, cal_res)
    price_call = compute_call_discounted_price_cos_method(S0_adj, driftT_adj, cal_res, opt, df)
    return FinancialFFT.call_to_put(price_call, S0_adj, df, opt)
end

function cos_method_pricer2(mcProcess::FinancialMonteCarlo.BaseProcess, r::FinancialMonteCarlo.AbstractZeroRateCurve, method::CosMethod, opt, T, mode::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    N = method.N
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = FinancialFFT.compute_chernorff_limits(mcProcess, rT, dT, T)
    bma = b - a
    u = FinancialFFT.adapt_array(collect((0:N) * (pi / bma)), mode)
    driftT = rT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    v_char_exp = @. FinancialFFT.characteristic_exponent_i(u * im, mcProcess) * T
    uk, v_char_exp_im = FinancialFFT.uk_call(u, a, b, v_char_exp)
    cal_res = FinancialFFT.CosMethodResult(u, v_char_exp_im, uk)
    driftT_adj = driftT - a
    df = exp(-rT)
    return finalize_cos_method(S0 * exp(-dT), opt, driftT_adj, df, cal_res)
end
