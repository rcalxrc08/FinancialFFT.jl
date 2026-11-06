using FinancialFFT

function charexpBSM(s, sigma)
    sigma_2 = sigma^2
    return -sigma_2 / 2 * s^2
end

exp_mod(x) = exp(FinancialFFT.real_mod(x)) * cos(FinancialFFT.imag_mod(x))

function FK(k_adj, a, drift_rn, sigma, T)
    FK = exp_mod(charexpBSM(k_adj, sigma) * T + im * k_adj * (drift_rn * T - a))
    return FK
end

function Density(S0, K, a, b, r, d, sigma, T, N)
    x = log(S0 / K)
    bma = b - a
    sigma_2 = sigma^2
    drift_rn = (r - d) - sigma_2 / 2
    sum_ = FK(0, a, drift_rn, sigma, T) / 2
    pi_adj = pi / bma
    adj_x = (x - a) * pi / bma
    for k = 1:N
        sum_ += cos(k * adj_x) * FK(k * pi_adj, a, drift_rn, sigma, T)
    end
    return 2 / bma * sum_
end

function Density2(x_val, S0, K, a, b, r, d, sigma, T, N)
    x = log(S0 / K) + x_val
    bma = b - a
    sigma_2 = sigma^2
    drift_rn = (r - d) - sigma_2 / 2
    sum_ = FK(0, a, drift_rn, sigma, T) / 2
    pi_adj = pi / bma
    adj_x = (x - a) * pi / bma
    for k = 1:N
        sum_ += cos(k * adj_x) * FK(k * pi_adj, a, drift_rn, sigma, T)
    end
    return 2 / bma * sum_
end
function Density3(x_val, mcProcess, K, r, T, N)
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = FinancialFFT.compute_chernorff_limits(mcProcess, rT, dT, T)
    x = log(S0 / K) + x_val
    bma = b - a
    drift_rn = rT - dT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    drift_adj = drift_rn - a
    sum_ = FK2(0, drift_adj, mcProcess, T) / 2
    pi_adj = pi / bma
    adj_x = (x - a) * pi_adj
    for k = 1:N
        sum_ += cos(k * adj_x) * FK2(k * pi_adj, drift_adj, mcProcess, T)
    end
    return 2 / bma * sum_
end

function FK2(k_adj, drift_rn, mcProcess, T)
    k_im = k_adj * im
    FK = exp_mod(FinancialFFT.characteristic_exponent_i(k_im, mcProcess) * T + k_im * drift_rn)
    return FK
end

function Density4(x_val, mcProcess, K, r, T, N)
    S0 = mcProcess.underlying.S0
    dT = FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(mcProcess), T)
    rT = FinancialMonteCarlo.integral(r.r, T)
    a, b = FinancialFFT.compute_chernorff_limits(mcProcess, rT, dT, T)
    bma = b - a
    drift_rn = rT - dT - FinancialFFT.characteristic_exponent_i(1, mcProcess) * T
    drift_adj = drift_rn - a
    p = AlternatePaddedVector(1, 2, 2, 2, N + 1)
    pi_adj = pi / bma
    k_vec = collect(0:N) .* pi_adj
    FKK = @. p * FK2(k_vec, drift_adj, mcProcess, T)
    x = log(S0 / K) + x_val
    adj_x = (x - a)
    sum_ = sum(@. FKK * cos(k_vec * adj_x))
    return sum_ / bma
end

function Density(S0, K, a, b, r, d, sigma, T, N)
    x = log(S0 / K)
    bma = b - a
    sigma_2 = sigma^2
    drift_rn = (r - d) - sigma_2 / 2
    pi_adj = pi / bma
    adj_x = (x - a)
    k_vec = collect(0:N) .* pi_adj
    p = AlternatePaddedVector(1, 2, 2, 2, N + 1)
    result = sum(@. p * cos(k_vec * adj_x) * FK(k_vec, a, drift_rn, sigma, T)) / bma
    return result
end

function calibrate_f_k(S0, K, a, b, r, d, sigma, T, N)
    bma = b - a
    sigma_2 = sigma^2
    drift_rn = (r - d) - sigma_2 / 2
    pi_adj = pi / bma
    k_vec = collect(0:N) .* pi_adj
    FKK = @. FK(k_vec, a, drift_rn, sigma, T)
    x = log(S0 / K)
    adj_x = (x - a)
    return FKK
end