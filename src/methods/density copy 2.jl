using FFTW, Interpolations
function characteristic_function_to_density(mcProcess, T, N, A, drift_rn)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess)
    n = 2^N
    eps_typed = ChainRulesCore.@ignore_derivatives eps(0.0)
    vec_lin = ChainRulesCore.@ignore_derivatives eps_typed .+ (0:(n-1))            # Indices
    vec = ChainRulesCore.@ignore_derivatives collect(vec_lin * pi)          # Indices
    dt = pi / A # Step size, frequency space
    c = -n / 2 * dt          # Evaluate the characteristic function on [c,d]
    drift = drift_rn - corr * T
    V = FinancialFFT.characteristic_exponent_vi(im * (c .+ vec / A), mcProcess) * T
    drift_im = drift * im
    V2 = (c .+ vec) * drift_im
    Y = @. exp(V + V2)
    Y_res = fft(Y)
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives [ifelse(isodd(i), one_adj, minus_one_adj) for i = 1:length(vec_lin)]
    density_vals = @. one_minus_one * FinancialFFT.real_mod(Y_res) / (2 * A)
    #TODO: eps should be typed
    eps_mod = ChainRulesCore.@ignore_derivatives eps(eltype(density_vals))
    density_vals_res = @. max(density_vals, eps_mod)
    # dx = 2 * A / n           # Step size, for the density
    # x = -A .+ vec_lin * 2 * A / n         # Grid, for the density
    x = A * (-1 .+ vec_lin * (2 / n))         # Grid, for the density
    dx = ChainRulesCore.@ignore_derivatives x.step
    almost_one = sum(density_vals_res) * dx
    return (x, density_vals_res / almost_one)
end

function density_new(mcProcess, T, r, N = 18, xmax = 10.0, St = mcProcess.underlying.S0)
    drift_rn = FinancialMonteCarlo.integral(r - mcProcess.underlying.d, T) + log(St)
    return characteristic_function_to_density(mcProcess, T, N, xmax, drift_rn)
end
function characteristic_function_to_density_n(mcProcess, N, A, r, T, St)
    n = ChainRulesCore.@ignore_derivatives 2^N
    vec_lin = ChainRulesCore.@ignore_derivatives 0:(n-1)
    eps_mod = eps(0.0)#wrong
    vec = ChainRulesCore.@ignore_derivatives collect((-n / 2 + eps_mod) .+ vec_lin)            # Indices
    drift_rd = FinancialMonteCarlo.integral(r - mcProcess.underlying.d, T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives @. ifelse(iseven(vec_lin), one_adj, minus_one_adj)
    drift_rf = drift_rd + log(St) - corr
    dt = pi / A # Step size, frequency space
    dt_im = dt * im
    t_im = vec * dt_im
    # V1 = characteristic_exponent_vi(t_im, mcProcess) * T
    Y = @. exp(characteristic_exponent_i(t_im, mcProcess) * T + drift_rf * t_im) * one_minus_one
    Y_r = fft(Y)
    density_vals = @. one_minus_one * FinancialFFT.real_mod(Y_r) / (2 * A)
    #TODO: eps should be typed
    density_vals_adj = @. max(density_vals, eps_mod)
    dx = 2 * A / n           # Step size, for the density
    x = -A .+ vec_lin * dx         # Grid, for the density
    almost_one = sum(density_vals_adj) * dx
    ChainRulesCore.@ignore_derivatives @assert !isnan(almost_one) "Returned result is NaN"
    return (x, density_vals_adj / almost_one)
end

function density(mcProcess, T, r, N = 18, xmax = 10.0, St = mcProcess.underlying.S0)
    return characteristic_function_to_density_n(mcProcess, N, xmax, r, T, St)
end

function characteristic_function_to_density_y(mcProcess, N, A, r, T, St, K = 1)
    n = ChainRulesCore.@ignore_derivatives 2^N
    vec_lin = ChainRulesCore.@ignore_derivatives 0:(n-1)
    eps_mod = eps(0.0)#wrong
    vec = ChainRulesCore.@ignore_derivatives collect((-n / 2 + eps_mod) .+ vec_lin)            # Indices
    drift_rd = FinancialMonteCarlo.integral(r - mcProcess.underlying.d, T)
    corr = FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    one_adj = ChainRulesCore.@ignore_derivatives Int8(1)
    minus_one_adj = ChainRulesCore.@ignore_derivatives Int8(-1)
    one_minus_one = ChainRulesCore.@ignore_derivatives @. ifelse(iseven(vec_lin), one_adj, minus_one_adj)
    drift_rf = drift_rd + log(St) - log(K) - corr
    dt = pi / A # Step size, frequency space
    dt_im = dt * im
    t_im = vec * dt_im
    # V1 = characteristic_exponent_vi(t_im, mcProcess) * T
    Y = @. exp(characteristic_exponent_i(t_im, mcProcess) * T + drift_rf * t_im) * one_minus_one
    Y_r = fft(Y)
    density_vals = @. one_minus_one * FinancialFFT.real_mod(Y_r) / (2 * A)
    #TODO: eps should be typed
    density_vals_adj = @. max(density_vals, eps_mod)
    dx = 2 * A / n           # Step size, for the density
    almost_one = sum(density_vals_adj) * dx
    ChainRulesCore.@ignore_derivatives @assert !isnan(almost_one) "Returned result is NaN"
    return density_vals_adj / almost_one
end

function density_y(mcProcess, T, r, N = 18, xmax = 10.0, St = mcProcess.underlying.S0, K = 1)
    return characteristic_function_to_density_y(mcProcess, N, xmax, r, T, St, K)
end
function characteristic_function_to_density_x(N, A)
    n = ChainRulesCore.@ignore_derivatives 2^N
    vec_lin = ChainRulesCore.@ignore_derivatives 0:(n-1)
    x_vec = ChainRulesCore.@ignore_derivatives collect(-1 .+ vec_lin * 2 / n)
    x = A * x_vec         # Grid, for the density
    return x
end

function density_x(N, xmax)
    return characteristic_function_to_density_x(N, xmax)
end

function payout_adimensional_1(ST::numtype_, euPayoff::EuropeanOption) where {numtype_ <: Number}
    iscall = ChainRulesCore.@ignore_derivatives ifelse(euPayoff.isCall, Int8(1), Int8(-1))
    zero_typed = ChainRulesCore.@ignore_derivatives zero(ST) * euPayoff.K
    return max(iscall * (ST - euPayoff.K), zero_typed)
end

function payout_adimensional_2(ST, euPayoff::EuropeanOption)
    iscall = ChainRulesCore.@ignore_derivatives ifelse(euPayoff.isCall, Int8(1), Int8(-1))
    zero_typed = ChainRulesCore.@ignore_derivatives zero(eltype(ST))
    K = euPayoff.K
    return @. K * max(iscall * (ST / K - 1), zero_typed)
end

function payout_adimensional_3(ST, euPayoff::EuropeanOption)
    iscall = ChainRulesCore.@ignore_derivatives ifelse(euPayoff.isCall, Int8(1), Int8(-1))
    zero_typed = ChainRulesCore.@ignore_derivatives zero(eltype(ST))
    return @. max(iscall * (ST - 1), zero_typed)
end

function pricer_from_density(mcProcess, T, r, opt, N = 18, xmax = 12.0)
    #K = 1
    K = opt.K
    f_x = density_y(mcProcess, T, r, N, xmax, mcProcess.underlying.S0, K)
    df = exp(-FinancialMonteCarlo.integral(r, T))
    x = ChainRulesCore.@ignore_derivatives exp.(density_x(N, xmax))
    p = K * payout_adimensional_3(x, opt)
    V = @. p * f_x
    n = ChainRulesCore.@ignore_derivatives 2^N
    dx = 2 * xmax / n           # Step size, for the density
    price = sum(V) * dx * df
    return price
end
