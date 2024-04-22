
real_mod(x) = real(x)
imag_mod(x) = imag(x)
adapt_array(x, _) = x
function adapt_itp(itp, _)
    return itp
end
function call_to_put(C, S0_adj, df, opt::EuropeanOption)
    K = opt.K
    P = C - S0_adj + K * df
    return ifelse(opt.isCall, C, P)
end

function call_to_put(C, _, df, opt::BinaryEuropeanOption)
    iscall = ChainRulesCore.@ignore_derivatives ifelse(opt.isCall, 1, 0)
    res = iscall * C + (1 - iscall) * (df - C)
    return res
end