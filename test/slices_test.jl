using FinancialFFT, FinancialMonteCarlo

@testset "Smiles Struct Consistency" begin
    K_vec = [90.0, 100.0, 110.0]
    K_vec_w = [90.0, 100.0, 110.0, 111.0]
    K_vec_w2 = [90.0, 100.0, -111.0]
    isCall_vec = [true, false, true]
    T = 1.0
    smile_eu = FinancialFFT.EuropeanOptionSmile(T, K_vec, isCall_vec)
    smile_bin = FinancialFFT.BinaryEuropeanOptionSmile(T, K_vec, isCall_vec)

    @test_throws "K and isCall must have the same length" FinancialFFT.EuropeanOptionSmile(T, K_vec_w, isCall_vec)
    @test_throws "Strikes must be positive" FinancialFFT.EuropeanOptionSmile(T, K_vec_w2, isCall_vec)
    @test_throws "Maturity must be positive" FinancialFFT.EuropeanOptionSmile(-1.0, K_vec, isCall_vec)

    @test_throws "K and isCall must have the same length" FinancialFFT.BinaryEuropeanOptionSmile(T, K_vec_w, isCall_vec)
    @test_throws "Strikes must be positive" FinancialFFT.BinaryEuropeanOptionSmile(T, K_vec_w2, isCall_vec)
    @test_throws "Maturity must be positive" FinancialFFT.BinaryEuropeanOptionSmile(-1.0, K_vec, isCall_vec)
end