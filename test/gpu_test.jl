# GPU execution in CudaMode. Not in SUITES, since GitHub CI has no GPU: on a machine with a CUDA GPU run
#     Pkg.test("FinancialFFT"; test_args = ["gpu"])
# which adds CUDA to the test environment (see runtests.jl). Smiles keep their strikes on the device. CarrMadanMethod
# ignores the mode and runs on the CPU; ReverseDiff and Mooncake are not supported on the GPU.
using CUDA

function gpu_payoff(kind, T, K, iscall)
    kind == :european_smile && return FinancialMonteCarlo.EuropeanOptionSmile(T, CuArray([K]), CuArray([iscall]))
    kind == :binary_smile && return FinancialMonteCarlo.BinaryEuropeanOptionSmile(T, CuArray([K]), CuArray([iscall]))
    return build_payoff(kind, T, K, iscall)
end

# A smile has a single strike here: its prices are summed, since scalar indexing is disallowed on the GPU.
function gpu_bs_pricer(method, kind, iscall)
    return function (S0, K, r, T, sigma, d)
        price = pricer(BlackScholesProcess(sigma, Underlying(S0, d)), ZeroRate(r), method, gpu_payoff(kind, T, K, iscall), FinancialMonteCarlo.CudaMode())
        return price isa AbstractArray ? sum(price) : price
    end
end

@testset "GPU" begin
    @test !isnothing(Base.get_extension(FinancialFFT, :FinancialFFTCUDAExt))
    if !CUDA.functional()
        @warn "No functional CUDA GPU: only the loading of the extension is tested."
    else
        gpu_cases = filter(case -> !(case.method isa CarrMadanMethod), METHOD_CASES)

        # Prices and gradients with respect to (S0, K, r, T, σ, d), with the tolerances of the CPU suites.
        @testset "Black Scholes vs FinancialToolbox" begin
            args = bs_args()
            for case in gpu_cases, (kind, toll) in case.payoffs, iscall in (true, false)
                f = gpu_bs_pricer(case.method, kind, iscall)
                g = analytic_bs_pricer(kind, iscall)
                reference = ForwardDiff.gradient(x -> g(x...), args)
                @testset "$(case.name) $kind $(iscall ? "call" : "put")" begin
                    @testset "Float64" begin
                        test_within(() -> abs(f(args...) - g(args...)), toll.price)
                    end
                    @testset "ForwardDiff" begin
                        test_within(() -> max_abs_diff(ForwardDiff.gradient(x -> f(x...), args), reference), toll.forwarddiff)
                    end
                    @testset "Zygote" begin
                        test_within(() -> max_abs_diff([something(x, 0.0) for x in Zygote.gradient(f, args...)], reference), toll.zygote)
                    end
                end
            end
        end

        # The characteristic exponents of the other models, evaluated on the device.
        @testset "Other models, GPU vs CPU" begin
            S0, r, d, T, K = 100.0, 0.02, 0.01, 1.0, 105.0
            for (model_name, model) in model_cases(S0, d), case in gpu_cases, kind in keys(case.payoffs)
                # outside Black Scholes, DensityInverter needs the truncation of the density, as in other_models_test.jl
                method = case.method isa FinancialFFT.DensityInverter ? FinancialFFT.DensityInverter(18, 3.0) : case.method
                @testset "$model_name $(case.name) $kind" begin
                    gpu_price = sum(pricer(model, ZeroRate(r), method, gpu_payoff(kind, T, K, true), FinancialMonteCarlo.CudaMode()))
                    cpu_price = sum(pricer(model, ZeroRate(r), method, build_payoff(kind, T, K, true)))
                    @test abs(gpu_price - cpu_price) < 1e-10
                end
            end
        end

        @testset "Several options, GPU vs CPU" begin
            model = MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(100.0, 0.01))
            zero_rate = ZeroRate(0.02)
            Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
            calls = [true, false, true, false, true]
            @testset "Array of options with different maturities, $(case.name)" for case in filter(case -> !(case.method isa FinancialFFT.DensityInverter), gpu_cases)
                opts = [EuropeanOption(1.0, 100.0), EuropeanOption(0.5, 70.0), EuropeanOption(1.0, 120.0, false), EuropeanOption(2.0, 100.0)]
                @test max_abs_diff(pricer(model, zero_rate, case.method, opts, FinancialMonteCarlo.CudaMode()), pricer(model, zero_rate, case.method, opts)) < 1e-10
            end
            @testset "Smiles, $(case.name)" for case in filter(case -> haskey(case.payoffs, :european_smile), gpu_cases)
                smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, CuArray(Ks), CuArray(calls))
                @test max_abs_diff(Array(pricer(model, zero_rate, case.method, smile, FinancialMonteCarlo.CudaMode())), pricer(model, zero_rate, case.method, FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls))) < 1e-10
            end
            @testset "Vector of strikes, CarrMadanLewisMethod" begin
                method = CarrMadanLewisMethod(600.0, 16)
                @test max_abs_diff(pricer(model, Ks, zero_rate, 1.0, method, FinancialMonteCarlo.CudaMode()), pricer(model, Ks, zero_rate, 1.0, method)) < 1e-10
            end
        end
    end
end
