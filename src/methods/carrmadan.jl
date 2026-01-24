"""
Struct for Carr Madan Method

		bsProcess=CarrMadanMethod(σ::num1) where {num1 <: Number}

Where:\n
		A	    = 	volatility of the process.
		Npow	=	volatility of the process.
"""
struct CarrMadanMethod{num <: Number, num_1 <: Integer} <: AbstractFFTMethod
    A::num
    Npow::num_1
    function CarrMadanMethod(A::num, N::num_1) where {num <: Number, num_1 <: Integer}
        ChainRulesCore.@ignore_derivatives @assert(A > 0.0, "A must be positive")
        ChainRulesCore.@ignore_derivatives @assert(N > 2, "N must be greater than 2")
        ChainRulesCore.@ignore_derivatives @assert(N < 24, "N will cause overflow")
        return new{num, num_1}(A, N)
    end
end
export CarrMadanMethod;

using DataInterpolations, FFTW;
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
    rT = FinancialMonteCarlo.integral(zero_rate.r, T)
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    A = method.A
    #* v-> compute integral as a summation
    dx = A / N
    real_vec = 0:(N-1)
    eps_ = 1e-320
    v = collect(real_vec) * dx            # the final value A is excluded
    v[1] = eps_
    # ChainRulesCore.@ignore_derivatives v[1] = 1e-22 #wrong
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    # v_adj = @. v - im
    # v_adj = @. -im * (v * im - im * im)
    # v_adj = @. im * (v_im + 1) * Int8(-1)
    # res = @. FinancialFFT.characteristic_exponent(im * (v_im + 1) * Int8(-1), mcProcess) * T
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives AlternateVector(one_adj, minus_one_adj, N)
    two_adj = ChainRulesCore.@ignore_derivatives Int8(2)
    four_adj = ChainRulesCore.@ignore_derivatives Int8(4)
    weights_ = ChainRulesCore.@ignore_derivatives @. ifelse(iseven(real_vec), two_adj, four_adj)
    # weights_ = ChainRulesCore.@ignore_derivatives @. 3 + (-1)^((0:(N-1)) + 1)# Simpson weights
    ChainRulesCore.@ignore_derivatives @views weights_[1] = 1
    ChainRulesCore.@ignore_derivatives @views weights_[end] = 1
    dk = 2 / A * pi
    b = pi * N / A
    drift_rd = rT - dT
    # corr_im = im * corr
    # res_1 = @. exp(v_im * (rT - dT + b))
    # res_2 = @. (exp(res - (v_im + 1) * corr) - 1)
    # res_2 = @. (exp(res - (v_im + 1) * corr) - 1)
    # v_im = @. v * im
    # @show one_minus_one[1:5]
    # @show exp.(im * b .* v)[1:5]
    v_im = @. v * im
    alfa = 0.0
    complex_vec_z_k = @. one_minus_one * cis(drift_rd * v) * (exp(FinancialFFT.characteristic_exponent_i(-(v_im + (1 + alfa)), mcProcess) * T - v_im * corr - corr) - 1) / ((v_im + alfa) * (alfa + 1 + v_im)) * dx * weights_ / 3
    # complex_vec_z_k = @. exp(v_im * (drift_rd + b)) * (exp(FinancialFFT.characteristic_exponent_i(-(v_im + 1), mcProcess) * T - (v_im + 1) * corr) - 1) / (v_im * (v_im + 1)) * (dx * weights_ / 3)
    # complex_vec_z_k = @. exp(v_im * (r - d) * T + v_im * b) * (exp(FinancialFFT.characteristic_exponent_i(-(v_im + 1), mcProcess) * T - (-im*v_im - 1im) * 1im * corr) - 1) / (v_im + v_im^2) * dx * weights_ / 3
    y = fft(complex_vec_z_k)
    z_T = @. real_mod(y) / pi

    ks = -b .+ dk * real_vec
    spline_cub = ChainRulesCore.@ignore_derivatives CubicSplineInterpolation(ks, z_T) #wrong
    k_interp = @. log(StrikeVec / S0)
    spl_res = ChainRulesCore.@ignore_derivatives spline_cub(k_interp) #wrong
    C = @. S0 * spl_res + max(S0 - StrikeVec * exp(-drift_rd), 0)
    return C * exp(-dT)
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::CarrMadanMethod, abstractPayoffs::Array{U}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode()) where {U <: FinancialMonteCarlo.EuropeanOption}
    TT = unique([opt.T for opt in abstractPayoffs])
    zero_typed = FinancialMonteCarlo.predict_output_type_zero(mcProcess, zero_rate, abstractPayoffs)
    prices = Array{typeof(zero_typed)}(undef, length(abstractPayoffs))

    for T in TT
        index_same_t = findall(op -> (op.T == T), abstractPayoffs)
        payoffs = abstractPayoffs[index_same_t]
        strikes = [opt.K for opt in payoffs]
        prices_call = pricer(mcProcess, strikes, zero_rate, T, method)
        dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
        prices_final = [call_to_put(prices_call[i], mcProcess.underlying.S0 * exp(dT), exp(-FinancialMonteCarlo.integral(zero_rate.r, T)), payoffs[i]) for i in eachindex(payoffs)]
        prices[index_same_t] = prices_final
    end

    return prices
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AbstractFFTMethod, abstractPayoff::FinancialMonteCarlo.EuropeanOption, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    res = first(pricer(mcProcess, [abstractPayoff.K], zero_rate, abstractPayoff.T, method))
    dT = -FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), abstractPayoff.T)
    return call_to_put(res, mcProcess.underlying.S0 * exp(dT), exp(-FinancialMonteCarlo.integral(zero_rate.r, abstractPayoff.T)), abstractPayoff)
end
