# Runs the suite of benchmarks.jl and saves the median of each benchmark:
#     julia benchmark/run.jl OUTPUT.json [PACKAGE_DIR]
# PACKAGE_DIR (default: this repository) is the copy of FinancialFFT to measure: the Benchmark workflow passes a
# checkout of the base of a pull request, measured with the suite of the pull request.
using Pkg
const output = abspath(ARGS[1])
const package_dir = abspath(length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, ".."))

# A temporary environment with the dependencies of benchmark/Project.toml and FinancialFFT from package_dir (the
# [sources] entry of the project points to this repository, so it is dropped).
environment = mktempdir()
project = read(joinpath(@__DIR__, "Project.toml"), String)
write(joinpath(environment, "Project.toml"), replace(project, r"\[sources\][^\[]*" => ""))
Pkg.activate(environment)
Pkg.develop(path = package_dir)
Pkg.instantiate()

using BenchmarkTools
# One second per benchmark: the suite has about a hundred of them. Set before the suite is built, since each
# @benchmarkable copies the default parameters when it is created.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0
include(joinpath(@__DIR__, "benchmarks.jl"))
tune!(SUITE)
results = run(SUITE; verbose = true)
BenchmarkTools.save(output, median(results))
println("Saved ", output)
