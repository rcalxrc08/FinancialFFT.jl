# Option Parameters
using PatternVectors, MuladdMacro
power(x, n) = x^n
multiply(a, b) = a * b
divide(a, b) = a / b

function CHI_scalar(uu_el, sincos_1, sincos_2, exp_c, exp_d)
    sin_uu_adj_1, cos_uu_adj_1 = sincos_1
    sin_uu_adj_2, cos_uu_adj_2 = sincos_2
    return @muladd (cos_uu_adj_1 * exp_d - cos_uu_adj_2 * exp_c + uu_el * (sin_uu_adj_1 * exp_d - sin_uu_adj_2 * exp_c)) / (1 + uu_el^2)
end
# functioninition of Chi(c,d) [Equation 22 Fang(2008)]
function CHI(uu, a, c, d)
    return @. CHI_scalar(uu, sincos(uu * (d - a)), sincos(uu * (c - a)), exp(c), exp(d))
end
# functionintion of Psi (c,d) [Equation 23 Fang(2008)]
function PSI(uu, a, c, d)
    adjuster = PatternVector(length(uu), InitialValuePattern(1, 0))
    uu_adj = @. uu + adjuster
    psi = @. (sin(uu_adj * (d - a)) - sin(uu_adj * (c - a))) / uu_adj
    psi_1 = sin(d - a) - sin(c - a)
    psi_0 = d - c - psi_1
    adjuster2 = PatternVector(length(uu), InitialValuePattern(psi_0, zero(psi_0)))
    return psi .+ adjuster2
end
# functionintion of U_k_vec [Equation 29 Fang(2008)]
function UK(u, a, b, iscall)
    lower = ifelse(!iscall, a, 0)
    upper = ifelse(!iscall, 0, b)
    sign_el = ifelse(iscall, 1, -1)
    bma = b - a
    res1 = CHI(u, a, lower, upper)
    res2 = PSI(u, a, lower, upper)
    return @. 2 / bma * (res1 - res2) * sign_el
end

function psi_minus_i(sigma)
    return 0.5 * sigma^2
end
function charexpBSM(u_im, sigma, T)
    adj_σ = sigma^2 / 2
    val_ = @. u_im^2 * adj_σ * T
    return val_
end

using FinancialFFT
# functioninition of Option Value [Equation 30 Fang(2008)]
function OptionValueCOS1_scalar(S0, K, a, u_im, CFBS, uk, adjuster)
    x = log(S0 / K)
    return sum(@. FinancialFFT.exp_mod(u_im * (x - a) + CFBS) * uk * adjuster)
end

# functioninition of Option Value [Equation 30 Fang(2008)]
function OptionValueCOS1_Vec(S0, K, r, d, T, sigma, N, a, b, iscall)
    k_vec = collect(0:N)
    bma = b - a
    u = @. k_vec * pi / bma
    r_d = r - d - psi_minus_i(sigma)
    u_im = @. u * im
    CFBS = @. charexpBSM(u_im, sigma, T) + r_d * u_im * T
    adjuster = PatternVector(length(k_vec), InitialValuePattern(1 // 2, 1 // 1))
    uk = UK(u, a, b, iscall)
    V_COS = [OptionValueCOS1_scalar(S0, K_el, a, u_im, CFBS, uk, adjuster) for K_el in K]
    return @. K * V_COS * exp(-r * T)
end