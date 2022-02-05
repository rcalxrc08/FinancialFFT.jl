"""
Struct for Lewis Integration Method

		bsProcess=LewisMethod(A,N)

Where:\n
		A	=	volatility of the process.
		N	=	volatility of the process.
"""
mutable struct LewisMethod{num <: Number, num_1 <: Integer} <: AbstractIntegralMethod
    A::num
    N::num_1
    function LewisMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        @assert(A > 0.0, "A must be positive")
        @assert(N > 2, "N must be greater than 2")
        return new{num, num_1}(A, N)
    end
end

export LewisMethod;

"""
Documentation Lewis Method
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
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
    mod = log(S0 / K) + (r - d) * T
    func_(z) = real_mod(exp(-z * 1im * mod) * CharFunc(-z - 1im * 0.5) / (z^2 + 0.25))
    int_1 = midpoint_definite_integral(func_, -A, A, N)
    price = S0 * (1 - exp(-mod / 2) * int_1 / (2 * pi)) * exp(-d * T)
    return call_to_put(price, mcProcess.underlying, zero_rate, abstractPayoff)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoffs::Array{U}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        prices[index_same_t] .= [pricer(mcProcess, zero_rate, method, payoff) for payoff in payoffs]
    end

    return prices
end