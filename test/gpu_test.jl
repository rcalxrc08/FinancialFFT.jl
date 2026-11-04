# GPU execution in CudaMode. Not in SUITES, since GitHub CI has no GPU: on a machine with a CUDA GPU run
#     Pkg.test("FinancialFFT"; test_args = ["gpu"])
# which adds CUDA to the test environment (see runtests.jl). In CudaMode the strikes of a smile must be on the device. CarrMadanMethod
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

        # Densities computed and evaluated on the device, against the CPU; scores with ForwardDiff and Zygote.
        @testset "Densities, GPU vs CPU" begin
            r, d, T, σ = 0.02, 0.01, 1.1, 0.2
            y = collect(range(-1.0, 1.0, length = 101))
            y_cu = CuArray(y)
            for (name, method) in ("FFTDensityInverter" => FFTDensityInverter(), "CosDensityMethod" => CosDensityMethod())
                @testset "$name $model_name" for (model_name, model) in vcat(["BlackScholes" => BlackScholesProcess(σ, Underlying(100.0, d))], model_cases(100.0, d))
                    f_gpu = density(model, ZeroRate(r), method, T, FinancialMonteCarlo.CudaMode())
                    f_cpu = density(model, ZeroRate(r), method, T)
                    @test max_abs_diff(Array(FinancialFFT.density_values(f_gpu, y_cu)), f_cpu.(y)) < 1e-12
                end
                scores(s, mode, points) = sum(log.(FinancialFFT.density_values(density(BlackScholesProcess(s[1], Underlying(100.0, d)), ZeroRate(r), method, T, mode), points)))
                reference = ForwardDiff.gradient(s -> scores(s, FinancialMonteCarlo.SerialMode(), y), [σ])
                @testset "$name ForwardDiff" begin
                    @test ForwardDiff.gradient(s -> scores(s, FinancialMonteCarlo.CudaMode(), y_cu), [σ]) ≈ reference rtol = 1e-10
                end
                @testset "$name Zygote" begin
                    @test Zygote.gradient(s -> scores(s, FinancialMonteCarlo.CudaMode(), y_cu), [σ])[1] ≈ reference rtol = 1e-10
                end
            end
            # a density computed on the device is evaluated on the host after adapt
            f_gpu = density(BlackScholesProcess(σ, Underlying(100.0, d)), ZeroRate(r), FFTDensityInverter(), T, FinancialMonteCarlo.CudaMode())
            @test FinancialFFT.adapt(Array, f_gpu)(0.1) ≈ density(BlackScholesProcess(σ, Underlying(100.0, d)), ZeroRate(r), FFTDensityInverter(), T)(0.1) rtol = 1e-12
        end

        @testset "Smiles, GPU vs CPU" begin
            model = MertonProcess(0.2, 5.0, 0.03, 0.02, Underlying(100.0, 0.01))
            zero_rate = ZeroRate(0.02)
            Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
            calls = [true, false, true, false, true]
            @testset "Smiles, $(case.name)" for case in filter(case -> haskey(case.payoffs, :european_smile), gpu_cases)
                method = case.method isa FinancialFFT.DensityInverter ? FinancialFFT.DensityInverter(18, 3.0) : case.method
                smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, CuArray(Ks), CuArray(calls))
                @test max_abs_diff(Array(pricer(model, zero_rate, method, smile, FinancialMonteCarlo.CudaMode())), pricer(model, zero_rate, method, FinancialMonteCarlo.EuropeanOptionSmile(1.0, Ks, calls))) < 1e-10
            end
        end
    end
end
