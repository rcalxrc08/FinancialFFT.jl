using BenchmarkTools
using Pkg, FinancialFFT;
#get(ENV, "CI", nothing) == "true" ? Pkg.instantiate() : nothing;
path1 = joinpath(dirname(pathof(FinancialFFT)), "..", "benchmark")
test_listTmp = readdir(path1);
BlackList = ["Project.toml", "benchmarks.jl", "runner.jl", "cuda", "af", "Manifest.toml"]
test_list = [test_element for test_element in test_listTmp if !Bool(sum(test_element .== BlackList))]
println("Running tests:\n")
for (current_test, i) in zip(test_list, 1:length(test_list))
    println("------------------------------------------------------------")
    println("  * $(current_test) *")
    include(joinpath(path1, current_test))
    println("------------------------------------------------------------")
    if (i < length(test_list))
        println("")
    end
end
