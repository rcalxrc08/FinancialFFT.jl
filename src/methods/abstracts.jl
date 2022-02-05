using FinancialMonteCarlo
import FinancialMonteCarlo.pricer, FinancialMonteCarlo.AbstractMethod
abstract type AbstractIntegrationMethod <: FinancialMonteCarlo.AbstractMethod end
abstract type AbstractFFTMethod <: AbstractIntegrationMethod end
abstract type AbstractIntegralMethod <: AbstractIntegrationMethod end