#!/usr/bin/env julia
# plot_results.jl — Corner plot generation for MCMC chains in Julia

using Pkg
Pkg.activate(dirname(@__FILE__))

using Serialization
using Turing
using MCMCChains
using Plots
using StatsPlots
using LaTeXStrings

# Parse arguments
model_name = get(ARGS, 1, "LCDM")

# Load chain
results_dir = joinpath(dirname(@__FILE__), "results")
chain_path = joinpath(results_dir, "$(model_name)_chain.jls")

if !isfile(chain_path)
    error("Serialized chain file not found at: $chain_path\nPlease run run_mcmc.jl first.")
end

println("Loading chain from: $chain_path...")
chain = Chains(deserialize(chain_path))

# Custom parameter labels matching the cosmology parameters
param_labels = Dict(
    "H0" => L"H_0 \text{ [km/s/Mpc]}",
    "Omega_m" => L"\Omega_m",
    "w" => L"w",
    "w0" => L"w_0",
    "wa" => L"w_a",
    "As" => L"A_s",
    "alpha" => L"\alpha",
    "xi" => L"\xi",
    "w1" => L"w_1",
    "epsilon" => L"\epsilon"
)

# Extract parameters present in the chain
chain_params = names(chain, :parameters)
labels = [get(param_labels, string(p), string(p)) for p in chain_params]

println("Generating corner plot for parameters: $chain_params...")

# Setup plot defaults for publication style
gr() # Use GR backend
theme(:default)

# Convert parameters to 2D matrix of samples
samples_mat = Array(chain[chain_params])

plt = cornerplot(
    samples_mat,
    label = labels,
    size = (800, 800),
    title = "$model_name Parameter Estimation",
    margin = 5Plots.mm,
    color = :goldenrod,
    alpha = 0.6
)

output_path = joinpath(results_dir, "$(model_name)_corner.png")
savefig(plt, output_path)
println("Corner plot saved successfully to: $output_path")
