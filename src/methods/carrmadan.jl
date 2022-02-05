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
    function CarrMadanMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        @assert(A > 0.0, "A must be positive")
        @assert(N > 2, "N must be greater than 2")
        @assert(N < 24, "N will cause overflow")
        return new{num, num_1}(A, N)
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
function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, StrikeVec::Array{U, 1}, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, T::Number, method::CarrMadanMethod, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: Number}
    Npow = method.Npow
    N = 2^Npow
    S0 = mcProcess.underlying.S0
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    d = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T) / T
    A = method.A
    #* v-> compute integral as a summation
    dx = A / N
    real_vec = 0:(N-1)
    v = collect(real_vec) * dx            # the final value A is excluded
    v[1] = 1e-312
    cf = FinancialFFT.CharactheristicFunction(mcProcess, T)
    corr = FinancialFFT.CharactheristicExponent(-1im, mcProcess, T)
    CharFunc(v) = cf(v) * exp(-v * 1im * corr)
    integrand_f(v) = exp(1im * (r - d) * v * T) * (CharFunc(v - 1im) - 1) / (1im * v - v^2)
    # Option Price
    weights_ = @. 3 + (-1)^((0:(N-1)) + 1)# Simpson weights
    @views weights_[1] = 1
    @views weights_[end] = 1
    dk = 2 * pi / A
    b = N * dk / 2
    complex_vec_z_k = @. integrand_f(v) * dx * weights_ / 3 * exp(1im * b * v)
    fft!(complex_vec_z_k)
    z_T = @. real_mod(complex_vec_z_k) / pi

    ks = -b .+ dk * real_vec
    spline_cub = CubicSplineInterpolation(ks, z_T)
    k_interp = @. log(StrikeVec / S0)
    C = @. S0 * spline_cub(k_interp) + max(S0 - StrikeVec * exp(-(r - d) * T), 0)
    return C * exp(-d * T)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractFFTMethod, abstractPayoffs::Array{U}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        strikes = [opt.K for opt in payoffs]
        prices_call = pricer(mcProcess, strikes, zero_rate, T, method)
        prices_final = [call_to_put(prices_call[i], mcProcess.underlying, zero_rate, payoffs[i]) for i = 1:length(payoffs)]
        prices[index_same_t] = prices_final
    end

    return prices
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractFFTMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    return first(pricer(mcProcess, [abstractPayoff.K], zero_rate, abstractPayoff.T, method))
end
