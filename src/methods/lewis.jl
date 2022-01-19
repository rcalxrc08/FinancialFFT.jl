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
        if A <= 0.0
            error("A must be positive")
        elseif N <= 2
            error("N must be greater than 2")
        else
            return new{num, num_1}(A, N)
        end
    end
end

function real_mod(x::Hyper{ComplexF64})
    return hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))
end

export LewisMethod;

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, K::Number, r::Number, T::Number, method::LewisMethod)
    A = method.A
    N = method.N
    S0 = mcProcess.underlying.S0
    d = FinancialMonteCarlo.dividend(mcProcess)
    CharExp(v) = FinancialFFT.CharactheristicExponent(v, mcProcess)
    EspChar(v) = CharExp(v) - v * 1im * CharExp(-1im)
    CharFunc(v) = exp(T * EspChar(v))
    x__ = log(S0 / K) + (r - d) * T
    func_(z) = real_mod(exp(-z * 1im * x__) * CharFunc(-z - 1im * 0.5) / (z^2 + 0.25))
    int_1 = integral_1(func_, -A, A, N)
    price = S0 * (1 - exp(-x__ / 2) * int_1 / (2 * pi)) * exp(-d * T)
    return price
end

function integral_1(f, xmin, xmax, N)
    x = range(xmin, length = N, stop = xmax)
    dx = (xmax - xmin) / (N - 1)
    sum_ = f(xmin) * dx * 0
    for x_ in x
        x_ != 0 && (sum_ += f(x_) * dx)
    end
    return sum_
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoffs_::Array{U}) where {U <: FinancialMonteCarlo.AbstractPayoff}
    r = zero_rate.r
    d = mcProcess.underlying.d

    f1(::T1) where {T1} = (T1 <: EuropeanOption)
    abstractPayoffs = filter(f1, abstractPayoffs_)

    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs_)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs_))

    for T in TT
        index_same_t = findall(op -> (op.T == T && f1(op)), abstractPayoffs_)
        payoffs = abstractPayoffs_[index_same_t]
        strikes = [opt.K for opt in payoffs]
        r_tmp = FinancialMonteCarlo.integral(r, T) / T
        d_tmp = FinancialMonteCarlo.integral(d, T) / T
        model2 = deepcopy(mcProcess)
        model2.underlying = Underlying(mcProcess.underlying.S0, d_tmp)
        prices[index_same_t] .= pricer.(model2, strikes, r_tmp, T, method)
    end

    length(abstractPayoffs) < length(abstractPayoffs_) ? (return prices) : (return prices * 1.0)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::LewisMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    r = zero_rate.r
    r_tmp = FinancialMonteCarlo.integral(r, abstractPayoff.T) / abstractPayoff.T
    return pricer(mcProcess, abstractPayoff.K, r_tmp, abstractPayoff.T, method)
end
