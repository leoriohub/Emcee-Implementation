module InferenceUtils

using Turing
using ..Models
using ..Likelihoods

export cosmology_mcmc_turing, run_inference

"""
Turing probabilistic model for cosmological MCMC fits.
Defines flat priors dynamically for all 10 expansion history models.
"""
@model function cosmology_mcmc_turing(data_CC, data_SN, model::AbstractCosmologyModel)
    # Define priors dynamically based on the model bounds
    params = Vector{Real}(undef, length(model.param_names))
    
    if model.name == "LCDM"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.0, 1.0)
        params = [H0, Omega_m]
    elseif model.name == "wCDM"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.0, 1.0)
        w ~ Uniform(-2.0, 0.0)
        params = [H0, Omega_m, w]
    elseif model.name == "CPL"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.0, 1.0)
        w0 ~ Uniform(-2.0, 0.0)
        wa ~ Uniform(-3.0, 3.0)
        params = [H0, Omega_m, w0, wa]
    elseif model.name == "DGP Brane-world"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.0, 1.0)
        params = [H0, Omega_m]
    elseif model.name == "JBP w(z)"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.1, 0.5)
        w0 ~ Uniform(-2.0, 0.0)
        wa ~ Uniform(-5.0, 5.0)
        params = [H0, Omega_m, w0, wa]
    elseif model.name == "Chaplygin Gas"
        H0 ~ Uniform(40.0, 100.0)
        As ~ Uniform(0.0, 1.0)
        alpha ~ Uniform(-1.0, 2.0)
        params = [H0, As, alpha]
    elseif model.name == "Interacting DE"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.01, 0.99)
        xi ~ Uniform(-0.5, 0.5)
        params = [H0, Omega_m, xi]
    elseif model.name == "Log-w Parameterization"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.1, 0.5)
        w0 ~ Uniform(-2.0, 0.0)
        w1 ~ Uniform(-2.0, 2.0)
        params = [H0, Omega_m, w0, w1]
    elseif model.name == "Starobinsky f(R)"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.1, 0.5)
        alpha ~ Uniform(-1.0, 1.0)
        params = [H0, Omega_m, alpha]
    elseif model.name == "Brans-Dicke MG"
        H0 ~ Uniform(40.0, 100.0)
        Omega_m ~ Uniform(0.1, 0.5)
        epsilon ~ Uniform(-0.5, 0.5)
        params = [H0, Omega_m, epsilon]
    else
        error("Model prior not implemented for: $(model.name)")
    end
    
    # Calculate log-likelihoods
    ll_CC = log_likelihood_CC(model, params, data_CC)
    ll_SN = log_likelihood_SN(model, params, data_SN)
    
    # Accumulate log probability in Turing
    Turing.@addlogprob! (ll_CC + ll_SN)
end

"""
Executes the MCMC parameter estimation.
"""
function run_inference(model::AbstractCosmologyModel, data_CC, data_SN; sampler_type=:nuts, n_samples=1000, n_chains=1)
    turing_model = cosmology_mcmc_turing(data_CC, data_SN, model)
    
    if sampler_type == :nuts
        println("Running MCMC using No-U-Turn Sampler (NUTS)...")
        if n_chains > 1
            # Parallel chain execution across CPU threads
            chain = sample(turing_model, NUTS(), MCMCThreads(), n_samples, n_chains)
        else
            chain = sample(turing_model, NUTS(), n_samples)
        end
    elseif sampler_type == :mh
        println("Running MCMC using Random Walk Metropolis-Hastings (MH)...")
        # MH requires more iterations to converge due to random walk properties
        chain = sample(turing_model, MH(), n_samples * 5)
    else
        error("Unknown sampler type: $sampler_type. Use :nuts or :mh.")
    end
    
    return chain
end

end
