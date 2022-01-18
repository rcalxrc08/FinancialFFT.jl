
function CharactheristicExponent(v::num_, mcProcess::FinancialMonteCarlo.BlackScholesProcess) where {num_ <: Number}
    σ = mcProcess.σ

    val_ = 0.5 * v * σ^2 * (1im - v)

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::FinancialMonteCarlo.MertonProcess) where {num_ <: Number}
    σ = mcProcess.σ
    sigma1 = mcProcess.σⱼᵤₘₚ
    mu1 = mcProcess.μⱼᵤₘₚ
    lam = mcProcess.λ

    val_ = -σ^2 * v^2 * 0.5 + lam * (exp(-sigma1^2 * v^2 / 2 + 1im * mu1 * v) - 1)

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::FinancialMonteCarlo.KouProcess) where {num_ <: Number}
    σ = mcProcess.σ
    lamp = mcProcess.λ₊
    lamm = mcProcess.λ₋
    p = mcProcess.p
    λ = mcProcess.λ

    val_ = -σ^2 * v^2 / 2 + 1im * v * λ * (p / (lamp - 1im * v) - (1 - p) / (lamm + 1im * v))

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::FinancialMonteCarlo.VarianceGammaProcess) where {num_ <: Number}
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ

    val_ = -1 / κ * log(1 + v^2 * σ^2 * κ / 2 - 1im * θ * κ * v)

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess) where {num_ <: Number}
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ

    val_ = (1 - sqrt(1 + (v^2 * σ^2 - 2 * 1im * θ * v) * κ)) / κ

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::FinancialMonteCarlo.HestonProcess) where {num_ <: Number}
    #Not implemented
end