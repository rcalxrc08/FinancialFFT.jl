# All of the current implemented characteristic exponents are implemented as a real function with a complex input.
#This allows us to compute directly the drift without the use of complex numbers.
# Namely characteristic_exponent(v)=characteristic_exponent_i(v*im)
function characteristic_exponent_i_bs(imv, adj_σ)
    return imv^2 * adj_σ
end
function characteristic_exponent_i(imv, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    adj_σ = σ^2 / 2
    return imv^2 * adj_σ
end
function characteristic_exponent_i_and_der(imv, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    adj_σ = σ^2
    adj_σ_2 = adj_σ / 2
    return imv^2 * adj_σ_2, imv * adj_σ
end
function characteristic_exponent_i_and_ders(imv, mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ
    adj_σ = σ^2
    adj_σ_2 = adj_σ / 2
    return imv^2 * adj_σ_2, imv * adj_σ, adj_σ
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

function characteristic_exponent_i22(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2 / 2 * imv^2
    adj_σ₀ = sigma1^2 / 2
    y1 = adj_σ₀ * imv + mu1
    y2 = imv * y1
    y3 = exp(y2) - 1
    val = adj_σ + lam * y3
    return val
end

function characteristic_exponent_i22_der(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    der_adj_σ = σ^2 * imv
    adj_σ = der_adj_σ / 2 * imv
    adj_σ₀ = sigma1^2 / 2
    y1 = adj_σ₀ * imv + mu1
    y2 = imv * y1
    der_y2 = y1 + imv * adj_σ₀
    y3 = exp(y2) - 1
    der_y3 = der_y2 * (y3 + 1)
    val = adj_σ + lam * y3
    der_val = der_adj_σ + lam * der_y3
    return val, der_val
end

function characteristic_exponent_i22_ders(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    der_adj_σ_2 = σ^2
    der_adj_σ = der_adj_σ2 * imv
    adj_σ₀ = sigma1^2 / 2
    y1 = adj_σ₀ * imv + mu1
    der_y1 = adj_σ₀
    y2 = imv * y1
    der_y2 = y1 + imv * adj_σ₀
    der_y2_2 = der_y1 + adj_σ₀
    y3 = exp(y2) - 1
    der_y3 = der_y2 * (y3 + 1)
    der_y3_2 = der_y2_2 * (y3 + 1) + der_y2 * der_y3
    der_val = der_adj_σ + lam * der_y3
    der_val_2 = der_adj_σ_2 + lam * der_y3_2
    return der_val_2
end

function characteristic_exponent_i_and_der(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2
    adj_σ_2 = adj_σ / 2
    adj_σ₀ = sigma1^2
    adj_σ₀_2 = adj_σ₀ / 2
    res_exp = exp(imv * (adj_σ₀_2 * imv + mu1))
    der_res_exp = res_exp * (adj_σ₀ * imv + mu1)
    val = adj_σ_2 * imv^2 + lam * (res_exp - 1)
    der = adj_σ * imv + lam * der_res_exp
    return val, der
end
function characteristic_exponent_i_and_ders(imv, mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σ_jump
    mu1 = mcProcess.μ_jump
    lam = mcProcess.λ
    adj_σ = σ^2
    adj_σ_2 = adj_σ / 2
    adj_σ₀ = sigma1^2
    adj_σ₀_2 = adj_σ₀ / 2
    res_exp = exp(imv * (adj_σ₀_2 * imv + mu1))
    muladd_res = adj_σ₀ * imv + mu1
    der_res_exp = res_exp * muladd_res
    der_res_exp_2 = der_res_exp * muladd_res + res_exp * adj_σ₀
    val = adj_σ_2 * imv^2 + lam * (res_exp - 1)
    der = adj_σ * imv + lam * der_res_exp
    der_2 = adj_σ + lam * der_res_exp_2
    return val, der, der_2
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
function tmp___()
    imv2 = imv^2
    imv3 = imv2 * imv
    imv4 = imv3 * imv
    imv5 = imv4 * imv
    lamp2 = lamp * lamp
    lamm2 = lamm * lamm
    σ_2 = σ^2
    prod_lam = lamm * lamp
    lambda_p = p * λ
    (-imv2 * lamm * λ + 2 * imv * prod_lam * λ - lamm * lamp2 * λ + imv2 * lamm * lambda_p + imv2 * lamp * lambda_p + lamm2 * lamp * lambda_p + lamm * lamp2 * lambda_p + imv5 * σ_2 + 2 * imv4 * lamm * σ_2 - 2 * imv4 * lamp * σ_2 + imv3 * lamm2 * σ_2 - 4 * imv3 * prod_lam * σ_2 + imv3 * lamp2 * σ_2 - 2 * imv2 * lamm2 * lamp * σ_2 + 2 * imv2 * lamm * lamp2 * σ_2 + imv * lamm2 * lamp2 * σ_2) / (((-imv + lamp)^2) * ((imv + lamm)^2))
    (imv * (lamm2 * lamp2 * σ_2 + 2 * prod_lam * λ) + imv2 * (lamm * lambda_p - lamm * λ + lamp * lambda_p - 2 * lamm2 * lamp * σ_2 + 2 * lamm * lamp2 * σ_2) - lamm * lamp2 * λ + lamm2 * lamp * lambda_p + lamm * lamp2 * lambda_p + imv5 * σ_2 + 2 * imv4 * lamm * σ_2 - 2 * imv4 * lamp * σ_2 + imv3 * lamm2 * σ_2 - 4 * imv3 * prod_lam * σ_2 + imv3 * lamp2 * σ_2) / (((-imv + lamp)^2) * ((imv + lamm)^2))
    term_1 = 0
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

function characteristic_exponent_i2_der(imv, mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ
    adj_σ = σ^2 * κ
    adj_θ = 2 * θ * κ
    y1 = imv * adj_σ + adj_θ
    der_y1 = adj_σ
    y2 = 1 - imv * y1
    der_y2 = -(y1 + imv * der_y1)
    y_tmp = sqrt(y2)
    y3 = 1 - y_tmp
    der_y3 = -inv(y_tmp * 2) * der_y2
    val_ = y3 / κ
    der_val_ = der_y3 / κ

    return val_, der_val_
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

function characteristic_exponent(v::num_, mcProcess::proc) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess}
    return characteristic_exponent_i(im * v, mcProcess)
end
function characteristic_exponent_v(v, mcProcess::proc) where {proc <: FinancialMonteCarlo.AbstractMonteCarloProcess}
    return characteristic_exponent_vi(im .* v, mcProcess)
end
function characteristic_exponent(v::num_, mcProcess::proc, T::num2) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
    ce = characteristic_exponent(v, mcProcess)
    cf = ce * T
    return cf
end

function Base.broadcasted(::S, ::typeof(characteristic_exponent), v::Array, mcProcess) where {S <: Base.Broadcast.BroadcastStyle}
    return characteristic_exponent_v(v, mcProcess[])
end
function Base.broadcasted(::S, ::typeof(characteristic_exponent), v::V, mcProcess) where {S <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.Broadcasted}
    return characteristic_exponent_v(v, mcProcess[])
end
function Base.broadcasted(::S, ::typeof(characteristic_exponent_i), v::Array, mcProcess) where {S <: Base.Broadcast.BroadcastStyle}
    return characteristic_exponent_vi(v, mcProcess[])
end
function Base.broadcasted(::S, ::typeof(characteristic_exponent_i), v::V, mcProcess) where {S <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.Broadcasted}
    return characteristic_exponent_vi(v, mcProcess[])
end
using ChainRulesCore: rrule_via_ad
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.broadcasted), ::typeof(characteristic_exponent_i), v, model)
    res = characteristic_exponent_vi(v, model)
    function update_pullback(slice)
        _, pb_fwd = ChainRulesCore.rrule_via_ad(config, characteristic_exponent_vi, v, model)
        _, der_v, der_mc = pb_fwd(slice)
        return NoTangent(), NoTangent(), der_v, der_mc
    end
    return res, update_pullback
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.broadcasted), ::typeof(characteristic_exponent), v, model)
    res = characteristic_exponent_v(v, model)
    function update_pullback(slice)
        _, pb_fwd = ChainRulesCore.rrule_via_ad(config, characteristic_exponent_v, v, model)
        _, der_v, der_mc = pb_fwd(slice)
        return NoTangent(), NoTangent(), der_v, der_mc
    end
    return res, update_pullback
end
