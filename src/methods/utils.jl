
real_mod(x) = real(x)
imag_mod(x) = imag(x)
adapt_array(x, _) = x
using ChainRulesCore
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

function midpoint_definite_integral(f, xmin, xmax, N)
    x = range(xmin, length = N, stop = xmax)
    dx = (xmax - xmin) / (N - 1)
    sum_ = zero(typeof(f(xmin))) * dx * 0
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)
    @views weights_[1] = 1
    @views weights_[N] = 1
    for (x_, w) in zip(x, weights_)
        x_ != 0 && !isnan(f(x_)) && (sum_ += f(x_) * w * dx / 3)
    end
    return sum_
end

include("AlternateVectors.jl")
include("alternate_padded.jl")