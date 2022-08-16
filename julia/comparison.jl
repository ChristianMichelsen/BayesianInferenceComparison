using Distributions: BetaBinomial, Beta, Exponential
using Turing: @model, sample, NUTS, MLE, MAP
using Plots
using StatsPlots
using BenchmarkTools: @benchmark, @btime
import Logging

##


k = [2036195, 745632, 279947, 200865, 106383, 150621]
N = [7642688, 7609177, 8992872, 8679915, 8877887, 8669401]
x = 1:length(N)

##

function μϕ_to_αβ(μ, ϕ)
    α = μ * ϕ
    β = (1 - μ) * ϕ
    return α, β
end

function Beta_μϕ(μ, ϕ)
    α, β = μϕ_to_αβ(μ, ϕ)
    return Beta(α, β)
end

function BetaBinomial_μϕ(N, μ, ϕ)
    α, β = μϕ_to_αβ(μ, ϕ)
    return BetaBinomial(N, α, β)
end



@model function model(x, k, N)

    A ~ Beta_μϕ(0.2, 5)
    q ~ Beta_μϕ(0.2, 5)
    c ~ Beta_μϕ(0.1, 10)
    ϕ ~ Exponential(1000) + 2

    for i in eachindex(x)
        μ = A * (1 - q)^(x[i] - 1) + c
        μ_clamped = clamp(μ, 0, 1)
        k[i] ~ BetaBinomial_μϕ(N[i], μ_clamped, ϕ)
    end
end


function get_samples(x, k, N)
    samples = Logging.with_logger(Logging.SimpleLogger(Logging.Error)) do
        sample(model(x, k, N), NUTS(1000, 0.65), 1000; progress = false, discard_initial = true)
    end
    return samples
end

samples = get_samples(x, k, N)

# @benchmark get_samples(x, k, N)
# BenchmarkTools.Trial: 59 samples with 1 evaluation.
#  Range (min … max):  79.766 ms … 112.350 ms  ┊ GC (min … max): 7.25% … 10.22%
#  Time  (median):     83.975 ms               ┊ GC (median):    6.84%
#  Time  (mean ± σ):   85.113 ms ±   4.803 ms  ┊ GC (mean ± σ):  7.75% ±  2.02%
#  Memory estimate: 73.09 MiB, allocs estimate: 753034.


##

plot(samples)


#%%

using Optim

p0 = [0.1, 0.5, 0.01, 1000]

function compute_MLE(x, k, N, p0)
    return optimize(model(x, k, N), MLE(), p0)
end
compute_MLE(x, k, N, p0)
# @benchmark compute_MLE(x, k, N, p0)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  269.333 μs …   9.318 ms  ┊ GC (min … max): 0.00% … 95.23%
#  Time  (median):     275.166 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   297.682 μs ± 413.614 μs  ┊ GC (mean ± σ):  6.53% ±  4.54%
#  Memory estimate: 175.07 KiB, allocs estimate: 2384.

function compute_MAP(x, k, N, p0)
    return optimize(model(x, k, N), MAP(), p0)
end
compute_MAP(x, k, N, p0)
# @benchmark compute_MAP(x, k, N, p0)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  294.584 μs …  13.320 ms  ┊ GC (min … max): 0.00% … 95.54%
#  Time  (median):     300.375 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   322.029 μs ± 415.090 μs  ┊ GC (mean ± σ):  5.89% ±  4.43%
#  Memory estimate: 175.07 KiB, allocs estimate: 2384.
