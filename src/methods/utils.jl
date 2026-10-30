
real_mod(x) = real(x)
imag_mod(x) = imag(x)
# Mooncake cannot differentiate the complex sqrt of Base: its extension overlays this hook.
sqrt_mod(x) = sqrt(x)
adapt_array(x, _) = x
# Quantities that are constants for differentiation (truncation ranges, grids). ReverseDiff does not honor
# ChainRulesCore.@ignore_derivatives, so its extension strips the tracking here.
ignore_derivatives_mod(x) = x
# Moves an interpolation to the device of the points where it is evaluated (the CUDA extension handles CuArrays).
function adapt_itp(itp, _)
    return itp
end
include("smile.jl")
function call_to_put(C, S0_adj, df, opt::EuropeanOption)
    adj = opt.K * df - S0_adj
    return ifelse(opt.isCall, C, C + adj)
end

function call_to_put(C, S0_adj, df, opt::EuropeanOptionSmile)
    return @. ifelse(opt.isCall, C, C - S0_adj + opt.K * df)
end

function call_to_put(C, _, df, opt::BinaryEuropeanOption)
    return ifelse(opt.isCall, C, df - C)
end

function call_to_put(C, _, df, opt::BinaryEuropeanOptionSmile)
    return @. ifelse(opt.isCall, C, df - C)
end
