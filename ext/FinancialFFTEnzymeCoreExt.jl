module FinancialFFTEnzymeCoreExt

# Enzyme support (#62). The rules live in EnzymeCore, the light interface package of Enzyme, so this extension loads
# with EnzymeCore and needs no dependency on Enzyme itself.
using FinancialFFT, FFTW
using EnzymeCore
using EnzymeCore: EnzymeRules, Annotation, Const, Duplicated, BatchDuplicated
using Base: IEEEFloat

# Enzyme, like ReverseDiff and Mooncake, ignores ChainRulesCore.@ignore_derivatives. The quantities that the package keeps
# constant for differentiation come from these functions, which are inactive: the hooks that strip derivatives, the search
# of the Chernoff truncation range (whose value at the optimum is differentiated, the optimum itself is a constant), the
# exponential moment bounds, the parameters of the Carr-Madan methods chosen from a tolerance, their damping and weights.
EnzymeRules.inactive(::typeof(FinancialFFT.v_value_mod), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.ignore_derivatives_mod), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.chernoff_optimum), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.exponential_moment_bound), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.fft_grid), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.carr_madan_damping), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.carr_madan_weights), args...) = nothing
EnzymeRules.inactive(::typeof(FinancialFFT.density_integration_grid), args...) = nothing

# Enzyme cannot differentiate through the FFTW plans (foreign calls). Both transforms of the pricers are linear maps y = L x:
# the tangent is L ẋ, with one plan for the primal and every tangent, and the adjoint adds Lᵀ ȳ to the cotangent of x.
const ComplexVector = Vector{<:Complex{<:IEEEFloat}}
lanes(config, shadow) = EnzymeRules.width(config) == 1 ? (shadow,) : shadow
unlanes(config, shadows) = EnzymeRules.width(config) == 1 ? only(shadows) : shadows

function linear_forward(config, x, plan)
    y = EnzymeRules.needs_primal(config) || x isa Const ? plan * x.val : nothing
    EnzymeRules.needs_shadow(config) || return EnzymeRules.needs_primal(config) ? y : nothing
    dys = x isa Const ? map(_ -> zero(y), ntuple(identity, Val(EnzymeRules.width(config)))) : map(dx -> plan * dx, lanes(config, x.dval))
    EnzymeRules.needs_primal(config) || return unlanes(config, dys)
    return EnzymeRules.width(config) == 1 ? Duplicated(y, only(dys)) : BatchDuplicated(y, dys)
end

function linear_augmented_primal(config, x, map_x)
    y = map_x(x.val)
    dys = EnzymeRules.needs_shadow(config) ? ntuple(_ -> zero(y), Val(EnzymeRules.width(config))) : nothing
    shadow = isnothing(dys) ? nothing : unlanes(config, dys)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? y : nothing, shadow, dys)
end

function linear_reverse!(config, dys, x, adjoint)
    (isnothing(dys) || x isa Const) && return nothing
    for (dx, dy) in zip(lanes(config, x.dval), dys)
        dx .+= adjoint(dy)
        fill!(dy, zero(eltype(dy)))
    end
    return nothing
end

# fft: the adjoint is the unnormalized inverse transform.
function EnzymeRules.forward(config::EnzymeRules.FwdConfig, ::Const{typeof(fft)}, ::Type{<:Annotation}, x::Annotation{<:ComplexVector})
    return linear_forward(config, x, plan_fft(x.val))
end
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(fft)}, ::Type{<:Annotation}, x::Annotation{<:ComplexVector})
    return linear_augmented_primal(config, x, fft)
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(fft)}, ::Type{<:Annotation}, dys, x::Annotation{<:ComplexVector})
    linear_reverse!(config, dys, x, bfft)
    return (nothing,)
end

# brfft_mod (the FFT density): the output is real, and the adjoint is brfft_adjoint, an rfft with weights 1, 2, …, 2, 1.
function EnzymeRules.forward(config::EnzymeRules.FwdConfig, ::Const{typeof(FinancialFFT.brfft_mod)}, ::Type{<:Annotation}, x::Annotation{<:ComplexVector}, n::Const{Int})
    return linear_forward(config, x, plan_brfft(x.val, n.val))
end
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(FinancialFFT.brfft_mod)}, ::Type{<:Annotation}, x::Annotation{<:ComplexVector}, n::Const{Int})
    return linear_augmented_primal(config, x, v -> brfft(v, n.val))
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(FinancialFFT.brfft_mod)}, ::Type{<:Annotation}, dys, x::Annotation{<:ComplexVector}, n::Const{Int})
    linear_reverse!(config, dys, x, dy -> FinancialFFT.brfft_adjoint(dy, n.val))
    return (nothing, nothing)
end

end
