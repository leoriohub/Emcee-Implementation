#!/usr/bin/env julia
# benchmark.jl — Julia MCMC benchmark measuring Effective Sample Size (ESS) per second

using Pkg
Pkg.activate(dirname(@__FILE__))

# Include the source modules
include("src/physics.jl")
include("src/models.jl")
include("src/data.jl")
include("src/likelihoods.jl")
include("src/inference.jl")

using .PhysicsUtils
using .Models
using .DataUtils
using .Likelihoods
using .InferenceUtils

using Turing
using MCMCChains
using Printf

function main()
    cc_path = joinpath(dirname(@__FILE__), "..", "Cosmic_chronometers_data.tex")
    sn_path = joinpath(dirname(@__FILE__), "..", "Pantheon+SH0ES.dat.txt")
    
    data_CC = load_CC(cc_path)
    data_SN = load_PPSH0ES(sn_path, nothing) # DIAGONAL mode for fair comparison
    
    model_name = get(ARGS, 1, "LCDM")
    available_models = Dict(
        "LCDM" => Models.LCDM,
        "WCDM" => Models.wCDM,
        "CPL" => Models.CPL
    )
    model_key = uppercase(model_name)
    if !haskey(available_models, model_key)
        error("Unknown model name: $model_name. Use LCDM, wCDM, or CPL.")
    end
    model = available_models[model_key]()
    
    turing_model = cosmology_mcmc_turing(data_CC, data_SN, model)
    
    println("Pre-compiling model functions (runs 50 warmup steps)...")
    # Warmup to compile the code so JIT latency does not affect the benchmark time
    sample(turing_model, NUTS(), 50; progress=false)
    
    nsamples = 1000
    println("Running Julia Turing NUTS benchmark (1000 samples)...")
    
    start_time = time()
    chain = sample(turing_model, NUTS(), nsamples; progress=false)
    end_time = time()
    
    total_time = end_time - start_time
    
    # Extract ESS using MCMCChains.ess
    ess_df = ess(chain)
    
    println("\n==================================================")
    println("❄️ Julia Turing NUTS Benchmark Results ($model_name Model)")
    println("==================================================")
    @printf("Total Wall Time:      %.2f seconds\n", total_time)
    println("--------------------------------------------------")
    for name in model.param_names
        sym = Symbol(name)
        vals = vec(chain[sym])
        med = quantile(vals, 0.50)
        p_ess = ess_df[sym]
        
        @printf("%-10s Median:      %.4f\n", name, med)
        @printf("%-10s ESS:         %.1f samples\n", name, p_ess)
        @printf("%-10s ESS/sec:     %.3f samples/sec\n", name, p_ess / total_time)
        println("--------------------------------------------------")
    end
    println("==================================================")
end

main()
