
function CharactheristicExponent(mcProcess::FinancialMonteCarlo.BlackScholesProcess)
    σ = mcProcess.σ

    val_(v) = 0.5 * v * σ^2 * (1im - v)

    return val_
end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.MertonProcess)
    σ = mcProcess.σ
    sigma1 = mcProcess.σⱼᵤₘₚ
    mu1 = mcProcess.μⱼᵤₘₚ
    lam = mcProcess.λ

    val_(v) = -σ^2 * v^2 * 0.5 + lam * (exp(-sigma1^2 * v^2 / 2 + 1im * mu1 * v) - 1)

    return val_
end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.KouProcess)
    σ = mcProcess.σ
    lamp = mcProcess.λ₊
    lamm = mcProcess.λ₋
    p = mcProcess.p
    λ = mcProcess.λ

    val_(v) = -σ^2 * v^2 / 2 + 1im * v * λ * (p / (lamp - 1im * v) - (1 - p) / (lamm + 1im * v))

    return val_
end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.VarianceGammaProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ

    val_(v) = -1 / κ * log(1 + v^2 * σ^2 * κ / 2 - 1im * θ * κ * v)

    return val_
end

function CharactheristicExponent(mcProcess::FinancialMonteCarlo.NormalInverseGaussianProcess)
    σ = mcProcess.σ
    θ = mcProcess.θ
    κ = mcProcess.κ

    val_(v) = (1 - sqrt(1 + (v^2 * σ^2 - 2 * 1im * θ * v) * κ)) / κ

    return val_
end

function CharactheristicExponent(v::num_, mcProcess::proc, T::num2) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
    ce = CharactheristicExponent(mcProcess)
    cf = ce(v) * T
    return cf
end

function CharactheristicFunction(v::num_, mcProcess::proc, T::num2) where {num_ <: Number, proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
    return exp(CharactheristicExponent(v, mcProcess, T))
end

function CharactheristicFunction(mcProcess::proc, T::num2) where {proc <: FinancialMonteCarlo.AbstractMonteCarloProcess, num2 <: Number}
    # ! a function is returned
    ce = CharactheristicExponent(mcProcess)
    cf(v) = exp(ce(v) * T)
    return cf
end