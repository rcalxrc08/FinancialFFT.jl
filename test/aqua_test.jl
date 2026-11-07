# Quality checks of Aqua.jl: ambiguities, piracy, unbound type parameters, stale dependencies, compat entries.
using Aqua
@testset "Aqua" begin
    # The broadcast of characteristic_exponent_i over a lazy argument (charexp.jl) takes any BroadcastStyle, so that every
    # array type and AD back-end gets the vectorised exponent (TrackedStyle is a plain BroadcastStyle in older ReverseDiff
    # versions); its only ambiguity is with the DictionaryStyle of Dictionaries.jl (a dependency of FinancialMonteCarlo),
    # which never broadcasts a characteristic exponent.
    Aqua.test_all(FinancialFFT; ambiguities = (exclude = [Base.broadcasted],))
end
