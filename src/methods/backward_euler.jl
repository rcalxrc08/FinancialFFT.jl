using LinearAlgebra, SparseArrays
"""
Struct for Lewis Integration Method

		bsProcess=BackwardEuler(A,N)

Where:\n
		A	=	volatility of the process.
		N	=	volatility of the process.
"""
mutable struct BackwardEuler{num <: Integer, num_1 <: Integer} <: AbstractIntegralMethod
    M::num
    N::num_1
    function BackwardEuler(M::num, N::num_1) where {num <: Integer, num_1 <: Integer}
        @assert(M > 2, "A must be positive")
        @assert(N > 2, "N must be greater than 2")
        return new{num, num_1}(M, N)
    end
end

export BackwardEuler;

function pricer(mcProcess::BlackScholesProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::BackwardEuler, abstractPayoff::FinancialMonteCarlo.EuropeanPayoff)
    T = abstractPayoff.T
    K = abstractPayoff.K
    N = method.N
    M = method.M
    S0 = mcProcess.underlying.S0
    r = FinancialMonteCarlo.integral(zero_rate.r, T) / T
    d = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T) / T
    ## log-price transformation
    # 1-Grid
    # payout(ST::numtype_, euPayoff::EuropeanOption)
    phi(ST) = FinancialMonteCarlo.payout(ST, abstractPayoff)
    σ = mcProcess.σ
    rd = r - d
    rdsigma2_2 = (rd - σ^2 / 2)
    num = typeof(S0 * rdsigma2_2 * T)
    Smin = S0 * exp(rdsigma2_2 * T - 6 * σ * sqrt(T))
    Smax = S0 * exp(rdsigma2_2 * T + 6 * σ * sqrt(T))
    x = range(log(Smin / S0), length = N + 1, stop = log(Smax / S0))
    dx = x[2] - x[1]
    dt = T / M

    # CallOrPut = isCall ? 1 : -1
    # 2-Terminal Condition & Linear System Matrix
    @views ST::Array{num} = S0 .* exp.(x[2:end-1])
    # V::Array{num} = (max.(CallOrPut .* (ST .- K), 0.0))
    V::Array{num} = @. phi(ST)
    a = -rdsigma2_2 / (2 * dx) + σ^2 / (2 * dx^2)
    b = -1 / dt - σ^2 / dx^2 - rd
    c = rdsigma2_2 / (2 * dx) + σ^2 / (2 * dx^2)
    #Supp = ones(N-1);
    #A=Tridiagonal(Supp[1:end-1].*a,Supp.*b,Supp[1:end-1].*c);
    A = Tridiagonal([a for i = 1:N-2], [b for i = 1:N-1], [c for i = 1:N-2])
    MAT = lu(A)
    # 3-Backward-in-time Loop
    BC = spzeros(num, N - 1)
    for j = M-1:-1:0
        if abstractPayoff.isCall
            # Vmax = Smax - K * exp(-rd * (T - j * dt))
            Vmax = phi(Smax)
            BC[end] = -c * Vmax
        else
            #Vmin=K*exp(-rd*(T-j*dt))-Smin;
            # Vmin = K * exp(-rd * (T - j * dt)) - Smin
            Vmax = phi(Smin)
            BC[1] = -a * Vmin
        end
        ###################
        V = MAT \ (-V ./ dt + BC)
    end
    PriceVec = V

    # 4-Plot and Price
    #priceInterpolator=Dierckx.Spline1D(S0*exp.(x[2:end-1]),exp(-d*T)*PriceVec);
    spline_cub = CubicSplineInterpolation(x[2:end-1], exp(-d * T) .* PriceVec)
    #priceInterpolator = interpolate((ST,), exp(-d*T).*PriceVec, Gridded(Linear()))

    VectorOfPrice = spline_cub(log(S0 / S0))
    #VectorOfPrice=priceInterpolator(S0);
    return VectorOfPrice
end

function pricer(mcProcess::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::BackwardEuler, abstractPayoffs::Array{U}) where {U <: FinancialMonteCarlo.EuropeanOption}
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