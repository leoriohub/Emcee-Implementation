#!/usr/bin/env julia
# setup.jl — Automated dependency installer for the Julia cosmology MCMC project

using Pkg

# Activate the local project environment
project_dir = dirname(@__FILE__)
Pkg.activate(project_dir)

println("==================================================")
println("⚙️ Initializing Julia Cosmology Project Environment")
println("==================================================")

dependencies = [
    "Turing",      # Core MCMC / NUTS framework
    "QuadGK",      # Adaptive integration for dL(z)
    "MCMCChains",  # Chain analysis and summary outputs
    "Plots",       # Base plotting library
    "StatsPlots",  # Statistical and corner plot macro expansions
    "LaTeXStrings" # LaTeX typesetting for parameter labels
]

for dep in dependencies
    println("\nInstalling: $dep...")
    try
        Pkg.add(dep)
        println("✓ Installed $dep")
    catch e
        println("✗ Failed to install $dep: $e")
    end
end

println("\n==================================================")
println("✓ Project setup complete! Environment is ready.")
println("Usage:")
println("  1. Run MCMC:   julia run_mcmc.jl <ModelName> <Samples> <UseCovariance>")
println("  2. Plot:       julia plot_results.jl <ModelName>")
println("==================================================")
