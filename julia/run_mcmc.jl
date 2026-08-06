#!/usr/bin/env julia
# run_mcmc.jl — Unified MCMC cosmological solver pipeline in Julia

# 1. Setup local environment
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

using Serialization
using MCMCChains

# 2. Parse arguments
model_name = get(ARGS, 1, "LCDM")
nsamples = parse(Int, get(ARGS, 2, "1000"))
use_covariance = parse(Bool, get(ARGS, 3, "true"))

println("==================================================")
println("🪐 Unified Cosmological Parameter Estimation (Julia)")
println("==================================================")
println("Model:         $model_name")
println("Samples:       $nsamples")
println("Covariance:    $(use_covariance ? "FULL Pantheon+" : "DIAGONAL only")")
println("==================================================")

# 3. Load Data
cc_path = joinpath(dirname(@__FILE__), "..", "Cosmic_chronometers_data.tex")
sn_path = joinpath(dirname(@__FILE__), "..", "Pantheon+SH0ES.dat.txt")
cov_path = use_covariance ? joinpath(dirname(@__FILE__), "..", "Pantheon+SH0ES_STAT+SYS.cov.txt") : nothing

data_CC = load_CC(cc_path)
data_SN = load_PPSH0ES(sn_path, cov_path)

# 4. Resolve Model
available_models = Dict(
    "LCDM" => Models.LCDM,
    "WCDM" => Models.wCDM,
    "CPL" => Models.CPL,
    "DGP" => Models.DGP,
    "JBP" => Models.JBP,
    "GCG" => Models.GCG,
    "IDE" => Models.IDE,
    "LOGW" => Models.LogW,
    "STAROBINSKY" => Models.Starobinsky,
    "BRANSDICKE" => Models.BransDicke
)

model_key = uppercase(model_name)
if !haskey(available_models, model_key)
    error("Unknown model name: $model_name. Available: $(keys(available_models))")
end

model_instance = available_models[model_key]()

# 5. Run MCMC Inference
# Use NUTS by default. If you need MH, change sampler_type to :mh
chain = run_inference(model_instance, data_CC, data_SN, sampler_type=:nuts, n_samples=nsamples)

# 6. Show results
println("\n=== Posterior Estimates ===")
display(chain)

# Calculate parameters quantiles manually for display
using Printf
println("\n=== Quantile Summary ===")
for name in model_instance.param_names
    # Turing parameters are saved as symbols, e.g., :H0
    sym = Symbol(name)
    vals = vec(chain[sym])
    med = quantile(vals, 0.50)
    low = quantile(vals, 0.16)
    high = quantile(vals, 0.84)
    
    err_low = med - low
    err_high = high - med
    
    @printf("%s = %.4f (-%.4f, +%.4f)\n", name, med, err_low, err_high)
end

# 7. Save results
output_dir = joinpath(dirname(@__FILE__), "results")
mkpath(output_dir)
save_path = joinpath(output_dir, "$(model_name)_chain.jls")
serialize(save_path, chain)
println("\nChain saved successfully to: $save_path")
println("You can generate corner plots using: julia plot_results.jl $model_name")
println("==================================================")
