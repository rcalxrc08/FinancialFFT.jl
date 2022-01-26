#TODO: CHANGE NAME
function pricer_cu(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    T = abstractPayoff.T
    K = abstractPayoff.K
    A = method.A
    N = method.N
    S0 = mcProcess.underlying.S0
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    d = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T) / T
    cf = FinancialFFT.CharactheristicFunction(mcProcess, T)
    corr = FinancialFFT.CharactheristicExponent(-1im, mcProcess, T)
    CharFunc(v) = cf(v) * exp(-v * 1im * corr)
    x__ = log(S0 / K) + (r - d) * T
    func_(z) = real_mod(exp(-z * 1im * x__) * CharFunc(-z - 1im * 0.5) / (z^2 + 0.25))
    int_1 = midpoint_definite_integral_cu(func_, -A, A, N)
    price = S0 * (1 - exp(-x__ / 2) * int_1 / (2 * pi)) * exp(-d * T)
    return call_to_put(price, mcProcess.underlying, zero_rate, abstractPayoff)
end