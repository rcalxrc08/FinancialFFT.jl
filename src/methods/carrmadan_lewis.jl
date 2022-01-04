"""
Struct for Lewis Integration Method

		bsProcess=CarrMadanLewisMethod(σ::num1) where {num1 <: Number}

Where:\n
		σ	=	volatility of the process.
"""
mutable struct CarrMadanLewisMethod{num <: Number, num_1 <: Integer} <: FinancialMonteCarlo.AbstractMethod
    A::num
    N::num_1
    function CarrMadanLewisMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        if A <= 0.0
            error("A must be positive")
        elseif N <= 2
            error("N must be greater than 2")
        else
            return new{num, num_1}(A, N)
        end
    end
end

export CarrMadanLewisMethod;

function CarrMadanLewisPricer(mcProcess::FinancialMonteCarlo.BaseProcess, K::Number, r::Number, T::Number, N::Integer, A::Real, d::Number = 0.0)
    S0 = mcProcess.underlying.S0
    CharExp(v) = FinancialFFT.CharactheristicExponent(v, mcProcess)
    EspChar(v) = CharExp(v) - v * 1im * CharExp(-1im)
    CharFunc(v) = exp(T * EspChar(v))
    k = log(K / S0)
    func_(z) = real(exp(z * 1im * (r * T - k)) * (CharFunc(z - 1im) - 1) / (1im * z - z^2))
    zt_k = integral_1(func_, -A, A, N) / (2 * pi)
    price = zt_k + max(1 - exp(k - r * T), 0)
    return price * S0
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, spotData::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, abstractPayoffs_::Array{U}) where {U <: FinancialMonteCarlo.AbstractPayoff}
    r = spotData.r
    d = mcProcess.underlying.d
    A = method.A
    N = method.N

    f1(::T1) where {T1} = (T1 <: EuropeanOption)
    abstractPayoffs = filter(f1, abstractPayoffs_)

    TT = unique([opt.T for opt in abstractPayoffs])
    prices = Array{Number}(undef, length(abstractPayoffs_))

    for T in TT
        index_same_t = findall(op -> (op.T == T && f1(op)), abstractPayoffs_)
        payoffs = abstractPayoffs_[index_same_t]
        strikes = [opt.K for opt in payoffs]
        r_tmp = FinancialMonteCarlo.integral(r, T) / T
        d_tmp = FinancialMonteCarlo.integral(d, T) / T
        prices[index_same_t] .= CarrMadanLewisPricer.(mcProcess, strikes, r_tmp, T, N, A, d_tmp)
    end

    length(abstractPayoffs) < length(abstractPayoffs_) ? (return prices) : (return prices * 1.0)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, spotData::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    r = spotData.r
    r_tmp = FinancialMonteCarlo.integral(r, abstractPayoff.T) / abstractPayoff.T
    d = mcProcess.underlying.d
    A = method.A
    N = method.N

    return CarrMadanLewisPricer(mcProcess, abstractPayoff.K, r_tmp, abstractPayoff.T, N, A, d)
end
