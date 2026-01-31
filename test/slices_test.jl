using FinancialFFT, FinancialMonteCarlo

@testset "Smiles Struct Consistency" begin
    K_vec = [90.0, 100.0, 110.0]
    K_vec_w = [90.0, 100.0, 110.0, 111.0]
    K_vec_w2 = [90.0, 100.0, -111.0]
    isCall_vec = [true, false, true]
    T = 1.0
    smile_eu = FinancialMonteCarlo.EuropeanOptionSmile(T, K_vec, isCall_vec)
    smile_bin = FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, K_vec, isCall_vec)

    @test_throws "K and isCall must have the same length" FinancialMonteCarlo.EuropeanOptionSmile(T, K_vec_w, isCall_vec)
    @test_throws "Strikes must be positive" FinancialMonteCarlo.EuropeanOptionSmile(T, K_vec_w2, isCall_vec)
    @test_throws "Maturity must be positive" FinancialMonteCarlo.EuropeanOptionSmile(-1.0, K_vec, isCall_vec)

    @test_throws "K and isCall must have the same length" FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, K_vec_w, isCall_vec)
    @test_throws "Strikes must be positive" FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, K_vec_w2, isCall_vec)
    @test_throws "Maturity must be positive" FinancialMonteCarlo.BinaryEuropeanOptionSmile(-1.0, K_vec, isCall_vec)
end