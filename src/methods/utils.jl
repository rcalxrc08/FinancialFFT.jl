
real_mod(x) = real(x)
function call_to_put(C, underlying, zero_rate, opt::EuropeanOption)
    S = underlying.S0
    T = opt.T
    d = FinancialMonteCarlo.integral(underlying.d, T) / T
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    K = opt.K
    return opt.isCall ? C : (C - S * exp(-d * T) + K * exp(-r * T))
end

function call_to_put(C, underlying, zero_rate, opt::BinaryEuropeanOption)
    T = opt.T
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    return opt.isCall ? C : (exp(-r * T) - C)
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
