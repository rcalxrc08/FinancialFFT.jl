# Validation of the arguments (#89): explicit exceptions, never @assert, which throws a generic AssertionError (not to be
# told apart from a bug) and may be disabled. DomainError for a number out of its domain; an invalid combination throws an
# ArgumentError where it is found (fft_parameters.jl). `check_domain` is a local copy of FinancialMonteCarlo's helper
# (FinancialMonteCarlo#71), internal there; the throw is out of line, as in PatternVectors' constructors (Enzyme aborted
# on an inlined constructor error, EnzymeAD/Enzyme.jl#4053, #86).
@noinline throw_domain_error(value, msg) = throw(DomainError(value, msg))
check_domain(cond::Bool, value, msg::AbstractString) = cond ? nothing : throw_domain_error(value, msg)

# One validation function per constructor, marked @non_differentiable and called without a closure: a
# ChainRulesCore.@ignore_derivatives block builds a closure whose captures Zygote traces: ~11 μs each with Zygote 0.7.18
# (FinancialMonteCarlo#80, FluxML/Zygote.jl#1753), ~2.5 μs with 0.7.19, against nothing measurable for the function
# (FinancialMonteCarlo#85). Enzyme and Mooncake go through them as through any function with no
# active output.
function check_fft_grid(A, Npow)
    check_domain(A > 0, A, "A must be positive")
    check_domain(Npow > 2, Npow, "Npow must be greater than 2")
    check_domain(Npow < 24, Npow, "Npow must be smaller than 24 (overflow)")
end
function check_fft_tolerance(ε, Npow_max)
    check_domain(ε > 0, ε, "the tolerance must be positive")
    check_domain(2 < Npow_max < 24, Npow_max, "Npow_max must be between 3 and 23")
end
check_damping(α) = check_domain(α > 0, α, "α must be positive")
check_nodes(N) = check_domain(N > 2, N, "N must be greater than 2")
function check_lewis(A, N)
    check_domain(A > 0, A, "A must be positive")
    check_nodes(N)
end
function check_quadrature_tolerance(ε, N_max)
    check_domain(ε > 0, ε, "the tolerance must be positive")
    check_nodes(N_max)
end
function check_density_method(N, M)
    check_nodes(N)
    check_domain(M > 2 && iseven(M), M, "M must be even and greater than 2")
end
ChainRulesCore.@non_differentiable check_fft_grid(::Any, ::Any)
ChainRulesCore.@non_differentiable check_fft_tolerance(::Any, ::Any)
ChainRulesCore.@non_differentiable check_damping(::Any)
ChainRulesCore.@non_differentiable check_nodes(::Any)
ChainRulesCore.@non_differentiable check_lewis(::Any, ::Any)
ChainRulesCore.@non_differentiable check_quadrature_tolerance(::Any, ::Any)
ChainRulesCore.@non_differentiable check_density_method(::Any, ::Any)
