using DualNumbers, HyperDualNumbers

function real_mod(x::Hyper)
    return hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))
end
real_mod(x) = real(x)
function call_to_put(C, underlying, zero_rate, opt)
    S = underlying.S0
    T = opt.T
    d = FinancialMonteCarlo.integral(underlying.d, T) / T
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    K = opt.K
    return opt.isCall ? C : (C - S * exp(-d * T) + K * exp(-r * T))
end
#Base.hash(x::Dual) = Base.hash(Base.hash("r") - Base.hash(x.value) - Base.hash("i") - Base.hash(x.epsilon))
function integral_1(f, xmin, xmax, N)
    x = range(xmin, length = N, stop = xmax)
    dx = (xmax - xmin) / (N - 1)
    sum_ = f(xmin) * dx * 0
    for x_ in x
        x_ != 0 && (sum_ += f(x_) * dx)
    end
    return sum_
end