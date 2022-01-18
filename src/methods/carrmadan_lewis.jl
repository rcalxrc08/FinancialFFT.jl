"""
Struct for Lewis Integration Method

		bsProcess=CarrMadanLewisMethod(σ::num1) where {num1 <: Number}

Where:\n
		σ	=	volatility of the process.
"""
mutable struct CarrMadanLewisMethod{num <: Number, num_1 <: Integer} <: FinancialMonteCarlo.AbstractMethod
    A::num
    Npow::num_1
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

function CarrMadanLewisPricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, r::Number, T::Number, method) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    d = FinancialMonteCarlo.dividend(mcProcess)
    A = method.A

    CharExp(v) = CharactheristicExponent(v, mcProcess)
    EspChar(v) = CharExp(v) + (r - d - CharExp(-1im)) * v * 1im
    CharFunc(v) = exp(T * EspChar(v))
    dx = A / N
    x = collect(0:(N-1)) * dx            # the final value A is excluded
    x[1] = 1e-22
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)
    @views weights_[1] = 1
    @views weights_[N] = 1

    dk = 2 * pi / A
    b = N * dk / 2
    ks = -b .+ dk * (0:(N-1))
    integrand = @. exp(-1im * b * x) * CharFunc(x - 0.5 * 1im) / (x^2 + 0.25) * weights_ * dx / 3
    fft!(integrand)
    integral_value = @. real_mod(integrand) / pi

    spline_cub = CubicSplineInterpolation(ks, integral_value)
    prices = @. S0 - sqrt(S0 * StrikeVec) * exp(-(r - d) * T) * spline_cub(log(StrikeVec / S0))
    return prices * exp(-d * T)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, spotData::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, abstractPayoffs_::Array{U}) where {U <: FinancialMonteCarlo.AbstractPayoff}
    r = spotData.r
    d = mcProcess.underlying.d

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
        prices[index_same_t] .= CarrMadanLewisPricer(mcProcess, strikes, r_tmp, T, method)
    end

    length(abstractPayoffs) < length(abstractPayoffs_) ? (return prices) : (return prices * 1.0)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, spotData::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanLewisMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    r = spotData.r
    r_tmp = FinancialMonteCarlo.integral(r, abstractPayoff.T) / abstractPayoff.T

    return first(CarrMadanLewisPricer(mcProcess, [abstractPayoff.K], r_tmp, abstractPayoff.T, method))
end
