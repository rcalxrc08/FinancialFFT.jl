# Shared helpers for the test suites.

# Black Scholes market data used by the analytic comparisons (blsprice / blsbin from FinancialToolbox).
const BS_ARGS = (S0 = 100.0, K = 105.0, r = 0.02, T = 1.1, sigma = 0.2, d = 0.01)
const BS_ARG_NAMES = (:S0, :K, :r, :T, :sigma, :d)
bs_args() = collect(values(BS_ARGS))

# Payoffs priced by each method, built from (T, K, iscall).
const PAYOFF_KINDS = (:european, :binary, :european_smile, :binary_smile)
function build_payoff(kind, T, K, iscall)
    kind == :european && return EuropeanOption(T, K, iscall)
    kind == :binary && return BinaryEuropeanOption(T, K, iscall)
    kind == :european_smile && return FinancialMonteCarlo.EuropeanOptionSmile(T, [K], [iscall])
    kind == :binary_smile && return FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, [K], [iscall])
end
is_binary(kind) = kind in (:binary, :binary_smile)

# Price as a scalar function of the Black Scholes inputs, so that every AD back-end can differentiate it.
function fft_bs_pricer(method, kind, iscall)
    return function (S0, K, r, T, sigma, d)
        price = pricer(BlackScholesProcess(sigma, Underlying(S0, d)), ZeroRate(r), method, build_payoff(kind, T, K, iscall))
        return price isa AbstractArray ? price[1] : price
    end
end
analytic_bs_pricer(kind, iscall) = is_binary(kind) ? ((S0, K, r, T, sigma, d) -> blsbin(S0, K, r, T, sigma, d, iscall)) : ((S0, K, r, T, sigma, d) -> blsprice(S0, K, r, T, sigma, d, iscall))

# Value and all derivative components of a number, used to compare results of any number type.
components(x::Real) = [x]
components(x::Dual) = [DualNumbers.value(x), DualNumbers.epsilon(x)]
components(x::Hyper) = [x.value, x.epsilon1, x.epsilon2, x.epsilon12]
components(x::ForwardDiff.Dual) = [ForwardDiff.value(x), ForwardDiff.partials(x)...]
components(x::Taylor1) = x.coeffs
max_abs_diff(x, y) = maximum(abs.(components(x) .- components(y)))
max_abs_diff(x::AbstractArray, y::AbstractArray) = maximum(abs.(x .- y))

# Reverse mode gradient of a scalar function of a vector with Mooncake.
mooncake_gradient(f, x) = Mooncake.value_and_gradient!!(Mooncake.prepare_gradient_cache(f, x), f, x)[2][2]

# A combination that is known to fail (error or inaccurate result): it is recorded as broken against `toll`,
# so that the suite reports it as soon as it starts working.
struct Broken
    toll::Float64
end
Broken() = Broken(1e-4)

test_within(check, toll::Real) = @test check() < toll
test_within(check, b::Broken) = @test_broken check() < b.toll
