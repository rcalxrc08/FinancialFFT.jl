real_mod(x) = real(x)
imag_mod(x) = imag(x)

function evaluate_integrand_lewis_v2_scalar_binary(v, corr_adj, char_exp_v)
    term = (1 // 2 + v * im) * corr_adj + char_exp_v
    exp_re = exp(FinancialFFT.real_mod(term))
    term_im = FinancialFFT.imag_mod(term)
    sin_im, cos_im = sincos(term_im)
    return 2 * exp_re * (cos_im + sin_im * 2v) / (1 + (2v)^2)
end

function evaluate_integrand_lewis_v22(v, corr_adj, char_exp_v)
    return @. evaluate_integrand_lewis_v2_scalar_binary(v, corr_adj, char_exp_v)
end

function lw_integrate(char_exp_v, v_im_adj, K, dT, S0, rT_corr)
    corr_adj = log(S0 / K) + rT_corr + dT
    total_sum_lewis = sum(evaluate_integrand_lewis_v22(v_im_adj, corr_adj, char_exp_v))
    return total_sum_lewis
end