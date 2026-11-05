# Runs the suite of benchmarks.jl and saves the median of each benchmark:
#     julia benchmark/run.jl OUTPUT.json [PACKAGE_DIR]
# PACKAGE_DIR (default: this repository) is the copy of FinancialFFT to measure: the Benchmark workflow passes a
# checkout of the base of a pull request, measured with the suite of the pull request.
# Each part of the suite (PARTS in benchmarks.jl) runs in its own Julia process, which keeps the memory to the code
# compiled for one part (see benchmarks.jl), with the heap size hint BENCHMARK_HEAP_SIZE_HINT if it is set (a hint below
# the live heap of a part makes the GC collect without end: 2G stalls the build of the LewisMethod part). Such a
# process runs
#     julia --project=ENV benchmark/run.jl --part PART OUTPUT.json
# in the environment ENV made here.
using Pkg
const is_part = length(ARGS) >= 3 && ARGS[1] == "--part"

if !is_part
    # A temporary environment with the dependencies of benchmark/Project.toml and FinancialFFT from PACKAGE_DIR (the
    # [sources] entry of the project points to this repository, so it is dropped).
    const output = abspath(ARGS[1])
    const package_dir = abspath(length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, ".."))
    const environment = mktempdir()
    project = read(joinpath(@__DIR__, "Project.toml"), String)
    write(joinpath(environment, "Project.toml"), replace(project, r"\[sources\][^\[]*" => ""))
    Pkg.activate(environment)
    Pkg.develop(path = package_dir)
    Pkg.instantiate()
end

using BenchmarkTools
# One second per benchmark. Set before the suite is built, since each @benchmarkable copies the default parameters
# when it is created.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# a ∪ b, recursively: the parts are disjoint, but share the groups above them ("Black Scholes").
function merge_groups!(a, b)
    for (key, value) in b
        if haskey(a, key) && a[key] isa BenchmarkGroup
            merge_groups!(a[key], value)
        else
            a[key] = value
        end
    end
    return a
end

is_part && include(joinpath(@__DIR__, "benchmarks.jl"))

if is_part
    part, part_output = ARGS[2], abspath(ARGS[3])
    group = suite(part)
    println("Tuning ", part)
    flush(stdout)
    tune!(group; verbose = true)
    results = run(group; verbose = true)
    BenchmarkTools.save(part_output, median(results))
    println(part, ": peak memory ", round(Sys.maxrss() / 2^30, digits = 2), " GB")
else
    julia = Base.julia_cmd()
    benchmarks = joinpath(@__DIR__, "benchmarks.jl")
    parts = readlines(`$julia --project=$environment -e 'include(ARGS[1]); foreach(println, PARTS)' $benchmarks`)
    heap_size_hint = haskey(ENV, "BENCHMARK_HEAP_SIZE_HINT") ? ["--heap-size-hint=" * ENV["BENCHMARK_HEAP_SIZE_HINT"]] : String[]
    results = BenchmarkGroup()
    for part in parts
        part_file = joinpath(environment, "part.json")
        run(`$julia --project=$environment $heap_size_hint $(@__FILE__) --part $part $part_file`)
        merge_groups!(results, only(BenchmarkTools.load(part_file)))
    end
    BenchmarkTools.save(output, results)
    println("Saved ", output)
end
