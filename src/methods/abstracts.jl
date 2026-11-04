using FinancialMonteCarlo
import FinancialMonteCarlo.pricer, FinancialMonteCarlo.AbstractMethod
abstract type AbstractIntegrationMethod <: FinancialMonteCarlo.AbstractMethod end
abstract type AbstractFFTMethod <: AbstractIntegrationMethod end
abstract type AbstractIntegralMethod <: AbstractIntegrationMethod end
Base.broadcastable(x::AbstractIntegrationMethod) = Ref(x)

# Methods that compute a density (density), and price by integrating the payoff against it.
abstract type AbstractDensityMethod <: AbstractIntegralMethod end
