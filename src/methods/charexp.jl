# All of the current implemented characteristic exponents are implemented as a real function with a complex input.
#This allows us to compute directly the drift without the use of complex numbers.
# Namely CharactheristicExponent(v)=CharactheristicExponent_i(v*im)
function CharactheristicExponent_i(im_v, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    adj_σ = σ^2 / 2
    val_ = im_v^2 * adj_σ
    return val_
end
function CharactheristicExponent_vi(im_v, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    # adj_σ*v^2
    adj_σ = σ^2 / 2
    val_ = @. im_v^2 * adj_σ
    return val_
end

function CharactheristicExponent_i(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2 / 2
    adj_σ₀ = sigma1^2 / 2
    val = adj_σ * imv^2 + lam * (exp(imv * (adj_σ₀ * imv + mu1)) - 1)
    return val
end

function CharactheristicExponent_vi(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2 / 2
    adj_σ₀ = sigma1^2 / 2
    # @show typeof(adj_σ₀)
    # @show typeof(imv)
    val = @. adj_σ * imv^2 + lam * (exp(imv * (adj_σ₀ * imv + mu1)) - 1)
    return val
end

function CharactheristicExponent_i(imv, mcProcess::FinancialMonteCarlo.KouProcess)
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

function CharactheristicExponent_vi(imv, mcProcess::FinancialMonteCarlo.KouProcess)
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

function CharactheristicExponent_i(imv, mcProcess::FinancialMonteCarlo.VarianceGammaProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ / 2
    adj_θ = θ * κ
    val_ = -inv(κ) * log(1 - imv * (imv * adj_σ + adj_θ))

    return val_
end
function CharactheristicExponent_vi(imv, mcProcess::FinancialMonteCarlo.VarianceGammaProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ / 2
    adj_θ = θ * κ
    val_ = @. -inv(κ) * log(1 - imv * (imv * adj_σ + adj_θ))

    return val_
end

function CharactheristicExponent_i(imv, mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ
    adj_θ = 2 * θ * κ
    val_ = (1 - sqrt(1 - imv * (imv * adj_σ + adj_θ))) / κ

    return val_
end
function CharactheristicExponent_vi(imv, mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ
    adj_θ = 2 * θ * κ
    val_ = @. (1 - sqrt(1 - imv * (imv * adj_σ + adj_θ))) / κ

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::proc) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess}
    return CharactheristicExponent_i(im * v, mcProcess)
end
function CharactheristicExponent_v(v, mcProcess::proc) where {proc <: FinancialMonteCarlo.AbstractMonteCarloProcess}
    return CharactheristicExponent_vi(im .* v, mcProcess)
end
function CharactheristicExponent(v::num_, mcProcess::proc, T::num2) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
    ce = CharactheristicExponent(v, mcProcess)
    cf = ce * T
    return cf
end

function CharactheristicFunction(v, mcProcess::proc, T::num2) where {proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
    ce = CharactheristicExponent(v, mcProcess)
    return exp(ce * T)
end
function Base.broadcasted(::S, ::typeof(CharactheristicExponent), v::Array, mcProcess) where {S <: Base.Broadcast.BroadcastStyle}
    return CharactheristicExponent_v(v, mcProcess[])
end
function Base.broadcasted(::S, ::typeof(CharactheristicExponent), v::V, mcProcess) where {S <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.Broadcasted}
    return CharactheristicExponent_v(v, mcProcess[])
end
function Base.broadcasted(::S, ::typeof(CharactheristicExponent_i), v::Array, mcProcess) where {S <: Base.Broadcast.BroadcastStyle}
    return CharactheristicExponent_vi(v, mcProcess[])
end
function Base.broadcasted(::S, ::typeof(CharactheristicExponent_i), v::V, mcProcess) where {S <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.Broadcasted}
    return CharactheristicExponent_vi(v, mcProcess[])
end
using ChainRulesCore: rrule_via_ad
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.broadcasted), ::typeof(CharactheristicExponent_i), v, model)
    res = CharactheristicExponent_vi(v, model)
    function update_pullback(slice)
        _, pb_fwd = ChainRulesCore.rrule_via_ad(config, CharactheristicExponent_vi, v, model)
        _, der_v, der_mc = pb_fwd(slice)
        return NoTangent(), NoTangent(), der_v, der_mc
    end
    return res, update_pullback
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.broadcasted), ::typeof(CharactheristicExponent), v, model)
    res = CharactheristicExponent_v(v, model)
    function update_pullback(slice)
        _, pb_fwd = ChainRulesCore.rrule_via_ad(config, CharactheristicExponent_v, v, model)
        _, der_v, der_mc = pb_fwd(slice)
        return NoTangent(), NoTangent(), der_v, der_mc
    end
    return res, update_pullback
end
