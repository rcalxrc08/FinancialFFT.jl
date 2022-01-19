"""
Struct for Carr Madan Method

		bsProcess=CarrMadanMethod(σ::num1) where {num1 <: Number}

Where:\n
		A	    = 	volatility of the process.
		Npow	=	volatility of the process.
"""
mutable struct CarrMadanMethod{num <: Number, num_1 <: Integer} <: AbstractFFTMethod
    A::num
    Npow::num_1
    alpha::Float64
    function CarrMadanMethod(A::num, Npow::num_1, alfa = 0.0) where {num <: Number, num_1 <: Integer}
        if A <= 0.0
            error("A must be positive")
        elseif Npow <= 2
            error("Npow must be greater than 2")
        else
            return new{num, num_1}(A, Npow, alfa)
        end
    end
end
export CarrMadanMethod;

using Interpolations, FFTW;
"""
Pricing European Options through Fast Fourier Transform Method (Carr Madan)

		VectorOfPrice=CarrMadanPricer(mcProcess::FinancialMonteCarlo.BaseProcess,S0::Number,StrikeVec::Array{U},r::Number,T::Number,Npow::Integer,A::Real,d::Number=0.0) where {U <: Number}

Where:\n
		S0 = Spot price.
		StrikeVec = Vector of Strike of the Option to price.
		r= zero rate with tenor T.
		T= tenor of the options.
		Npow= Integer Parameter for the FFT. Represent the log2 of the number of nodes.
		A= Real Parameter of the FFT. Represent a maximum for the axis.

		VectorOfPrice= Price of the European Options with Strike equals to StrikeVec, tenor T and the market prices a risk free rate of r.
"""
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, r::Number, T::Number, method::CarrMadanMethod) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    d = FinancialMonteCarlo.dividend(mcProcess)
    CharExp(v) = CharactheristicExponent(v, mcProcess)
    EspChar(v) = CharExp(v) - v * 1im * CharExp(-1im)
    A = method.A
    #v-> compute integral as a summation
    dx = A / N
    real_vec = 0:(N-1)
    v = collect(real_vec) * dx            # the final value A is excluded
    v[1] = 1e-312
    CharFunc(v) = exp(T * EspChar(v))
    integrand_f(v) = exp(1im * (r - d) * v * T) * (CharFunc(v - 1im) - 1) / (1im * v - v^2)
    # Option Price
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)# Simpson weights
    @views weights_[1] = 1
    @views weights_[end] = 1
    dk = 2 * pi / A
    b = N * dk / 2
    complex_vec_z_k = @. integrand_f(v) * dx * weights_ / 3 * exp(1im * b * v) #* exp(1im * pi * real_vec)
    # integrand = @. exp(-1im * b * x) * CharFunc(x - 0.5 * 1im) / (x^2 + 0.25) * weights_ * dx / 3
    fft!(complex_vec_z_k)
    z_T = @. real_mod(complex_vec_z_k) / pi

    ks = -b .+ dk * real_vec
    spline_cub = CubicSplineInterpolation(ks, z_T)
    k_interp = @. log(StrikeVec / S0)
    C = @. S0 * spline_cub(k_interp) + max(S0 - StrikeVec * exp(-(r - d) * T), 0)
    # P = S * real( fft(h)/pi + max(1-exp(k-r*T),0)); # prices
    return C * exp(-d * T)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractFFTMethod, abstractPayoffs_::Array{U}) where {U <: FinancialMonteCarlo.AbstractPayoff}
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
        prices[index_same_t] = pricer(model2, strikes, r_tmp, T, method)
    end

    length(abstractPayoffs) < length(abstractPayoffs_) ? (return prices) : (return prices * 1.0)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractFFTMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption)
    r = zero_rate.r
    r_tmp = FinancialMonteCarlo.integral(r, abstractPayoff.T) / abstractPayoff.T

    return first(pricer(mcProcess, [abstractPayoff.K], r_tmp, abstractPayoff.T, method))
end
