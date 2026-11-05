# Markdown report of the results saved by run.jl, compared with a baseline when one is given:
#     julia benchmark/compare.jl HEAD.json [BASE.json] > report.md
# A change is flagged when the time moves by more than TIME_TOLERANCE: shared CI runners are noisy, so only large
# changes mean anything there.
using Pkg
Pkg.activate(; temp = true)
Pkg.add("BenchmarkTools"; io = devnull)
using BenchmarkTools

const TIME_TOLERANCE = 0.15

load_estimates(file) = only(BenchmarkTools.load(file))
head = load_estimates(ARGS[1])
base = length(ARGS) > 1 && isfile(ARGS[2]) ? load_estimates(ARGS[2]) : nothing

format_time(t) = BenchmarkTools.prettytime(t)
format_memory(m) = BenchmarkTools.prettymemory(m)
benchmark_name(keys) = join(keys, " / ")
function mark(judgement)
    t = time(judgement)
    return t == :regression ? "🔴 slower" : t == :improvement ? "🟢 faster" : ""
end

println("## Benchmarks\n")
println("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", ", Sys.CPU_THREADS, " threads. Medians.\n")
if isnothing(base)
    println("| Benchmark | Time | Memory | Allocations |")
    println("|:--|--:|--:|--:|")
    for (keys, estimate) in sort(collect(BenchmarkTools.leaves(head)); by = first)
        println("| ", benchmark_name(keys), " | ", format_time(time(estimate)), " | ", format_memory(memory(estimate)), " | ", allocs(estimate), " |")
    end
else
    println("Ratios are this commit over the base; changes beyond ", round(Int, 100 * TIME_TOLERANCE), "% are flagged.\n")
    println("| Benchmark | Base | This commit | Ratio | Memory ratio | |")
    println("|:--|--:|--:|--:|--:|:--|")
    base_leaves = Dict(BenchmarkTools.leaves(base))
    for (keys, estimate) in sort(collect(BenchmarkTools.leaves(head)); by = first)
        if haskey(base_leaves, keys)
            reference = base_leaves[keys]
            judgement = judge(estimate, reference; time_tolerance = TIME_TOLERANCE)
            r = ratio(estimate, reference)
            println("| ", benchmark_name(keys), " | ", format_time(time(reference)), " | ", format_time(time(estimate)), " | ", round(time(r), digits = 2), " | ", round(memory(r), digits = 2), " | ", mark(judgement), " |")
        else
            println("| ", benchmark_name(keys), " | (new) | ", format_time(time(estimate)), " | | | |")
        end
    end
end
