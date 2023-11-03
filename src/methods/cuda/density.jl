function adapt_array(x, ::FinancialMonteCarlo.AbstractCudaMode)
    # @show "cuda"
    return cu(x)
end