# All of the current implemented characteristic exponents are implemented as a real function with a complex input.
#This allows us to compute directly the drift without the use of complex numbers.
# Namely characteristic_exponent(v)=characteristic_exponent_i(v*im)
function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    adj_σ = σ^2 / 2
    val_ = imv^2 * adj_σ
    return val_
end
function characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    adj_σ = σ^2 / 2
    val_ = @. imv^2 * adj_σ
    return val_
end

function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2 / 2
    adj_σ₀ = sigma1^2 / 2
    val = adj_σ * imv^2 + lam * (exp(imv * (adj_σ₀ * imv + mu1)) - 1)
    return val
end

function characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2 / 2
    adj_σ₀ = sigma1^2 / 2
    val = @. adj_σ * imv^2 + lam * (exp(imv * (adj_σ₀ * imv + mu1)) - 1)
    return val
end

function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.KouProcess)
    σ = mcProcess.σ
    lamp = mcProcess.λ₊
    lamm = mcProcess.λ₋
    p = mcProcess.p
    λ = mcProcess.λ

    adj_σ = σ^2 / 2
    one_minus_p = 1 - p
    val_ = imv * (adj_σ * imv + λ * (p / (lamp - imv) - one_minus_p / (lamm + imv)))
    return val_
end

function characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.KouProcess)
    σ = mcProcess.σ
    lamp = mcProcess.λ₊
    lamm = mcProcess.λ₋
    p = mcProcess.p
    λ = mcProcess.λ

    adj_σ = σ^2 / 2
    one_minus_p = 1 - p
    val_ = @. imv * (adj_σ * imv + λ * (p / (lamp - imv) - one_minus_p / (lamm + imv)))
    return val_
end

function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.VarianceGammaProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ / 2
    adj_θ = θ * κ
    val_ = -inv(κ) * log(1 - imv * (imv * adj_σ + adj_θ))

    return val_
end
function characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.VarianceGammaProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ / 2
    adj_θ = θ * κ
    val_ = @. -inv(κ) * log(1 - imv * (imv * adj_σ + adj_θ))

    return val_
end

function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ
    adj_θ = 2 * θ * κ
    val_ = (1 - sqrt(1 - imv * (imv * adj_σ + adj_θ))) / κ

    return val_
end
function characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ
    adj_θ = 2 * θ * κ
    val_ = @. (1 - sqrt(1 - imv * (imv * adj_σ + adj_θ))) / κ

    return val_
end

# Log of E[exp(imv * X_T)], where X_T is the driftless log price. It is linear in T for a Lévy process, which only needs the
# methods above; a process that is not Lévy defines these ones instead.
characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.BaseProcess, T) = characteristic_exponent_i(imv, mcProcess) * T
characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.BaseProcess, T) = characteristic_exponent_vi(imv, mcProcess) .* T

# Formulation of Albrecher et al. ("The little Heston trap"), where the principal branches of sqrt and log are continuous.
# It is even in γ, and exp(-γ * T) does not overflow.
function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.HestonProcess, T)
    σ = mcProcess.σ
    σ² = σ^2
    v0 = mcProcess.σ₀^2
    κ_s = mcProcess.κ + mcProcess.λ
    adj_κθ = mcProcess.κ * mcProcess.θ / σ²
    ρσ = mcProcess.ρ * σ
    β = κ_s - ρσ * imv
    α = imv * (imv - 1)
    γ = sqrt_mod(β * β - σ² * α)
    exp_γT = exp(-γ * T)
    den = γ + β + (γ - β) * exp_γT
    val_ = adj_κθ * ((β - γ) * T - 2 * log(den / (2 * γ))) + v0 * α * (1 - exp_γT) / den
    return val_
end
# γ is imaginary for a large real imv, the exponent is still real as long as E[exp(imv * X_T)] is finite.
function characteristic_exponent_i(imv::Real, mcProcess::FinancialMonteCarlo.HestonProcess, T)
    val_ = real_mod(characteristic_exponent_i(complex(imv), mcProcess, T))
    # S is a martingale: this also covers γ = 0, when κ_s = ρσ
    isone(imv) && return zero(val_)
    return val_
end
function characteristic_exponent_vi(imv, mcProcess::FinancialMonteCarlo.HestonProcess, T)
    σ = mcProcess.σ
    σ² = σ^2
    v0 = mcProcess.σ₀^2
    κ_s = mcProcess.κ + mcProcess.λ
    adj_κθ = mcProcess.κ * mcProcess.θ / σ²
    ρσ = mcProcess.ρ * σ
    β = @. κ_s - ρσ * imv
    α = @. imv * (imv - 1)
    γ = @. sqrt_mod(β * β - σ² * α)
    exp_γT = @. exp(-γ * T)
    den = @. γ + β + (γ - β) * exp_γT
    val_ = @. adj_κθ * ((β - γ) * T - 2 * log(den / (2 * γ))) + v0 * α * (1 - exp_γT) / den
    return val_
end

# function characteristic_exponent(v::num_, mcProcess::proc) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess}
#     return characteristic_exponent_i(im * v, mcProcess)
# end
# function characteristic_exponent_v(v, mcProcess::proc) where {proc <: FinancialMonteCarlo.AbstractMonteCarloProcess}
#     return characteristic_exponent_vi(im .* v, mcProcess)
# end
# function characteristic_exponent(v::num_, mcProcess::proc, T::num2) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
#     ce = characteristic_exponent(v, mcProcess)
#     cf = ce * T
#     return cf
# end

# function Base.broadcasted(::S, ::typeof(characteristic_exponent), v::Array, mcProcess) where {S <: Base.Broadcast.BroadcastStyle}
#     return characteristic_exponent_v(v, mcProcess[])
# end
# function Base.broadcasted(::S, ::typeof(characteristic_exponent), v::V, mcProcess) where {S <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.Broadcasted}
#     return characteristic_exponent_v(v, mcProcess[])
# end
function Base.broadcasted(::S, ::typeof(characteristic_exponent_i), v::V, mcProcess, T) where {S <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.Broadcasted}
    return characteristic_exponent_vi(v, mcProcess[], T[])
end
using ChainRulesCore: rrule_via_ad
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.broadcasted), ::typeof(characteristic_exponent_i), v, model, T)
    res = characteristic_exponent_vi(v, model, T)
    function update_pullback(slice)
        _, pb_fwd = ChainRulesCore.rrule_via_ad(config, characteristic_exponent_vi, v, model, T)
        _, der_v, der_mc, der_T = pb_fwd(slice)
        return NoTangent(), NoTangent(), der_v, der_mc, der_T
    end
    return res, update_pullback
end
# function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.broadcasted), ::typeof(characteristic_exponent), v, model)
#     res = characteristic_exponent_v(v, model)
#     function update_pullback(slice)
#         _, pb_fwd = ChainRulesCore.rrule_via_ad(config, characteristic_exponent_v, v, model)
#         _, der_v, der_mc = pb_fwd(slice)
#         return NoTangent(), NoTangent(), der_v, der_mc
#     end
#     return res, update_pullback
# end
