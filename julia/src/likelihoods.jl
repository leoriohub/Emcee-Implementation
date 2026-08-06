module Likelihoods

using LinearAlgebra
using ..Models

export log_likelihood_CC, log_likelihood_SN, within_bounds

"""
Helper function to verify if parameters reside within their model-defined physical bounds.
"""
function within_bounds(model::AbstractCosmologyModel, params)
    for (val, bounds) in zip(params, model.param_bounds)
        if !(bounds[1] <= val <= bounds[2])
            return false
        end
    end
    return true
end

"""
Log-likelihood for Cosmic Chronometers.
"""
function log_likelihood_CC(model::AbstractCosmologyModel, params, data_CC)
    if !within_bounds(model, params)
        return -Inf
    end
    
    z = data_CC["z"]
    Hz_data = data_CC["Hz"]
    sigma_Hz = data_CC["sigma_Hz"]
    
    # Calculate H(z) for each data point
    Hz_model = [H(model, zi, params...) for zi in z]
    
    chi2 = sum(((Hz_data .- Hz_model) ./ sigma_Hz).^2)
    return -0.5 * chi2
end

"""
Log-likelihood for Supernovae. Supports both diagonal-only and full covariance modes.
"""
function log_likelihood_SN(model::AbstractCosmologyModel, params, data_SN)
    if !within_bounds(model, params)
        return -Inf
    end
    
    z = data_SN["z"]
    mu_data = data_SN["mu"]
    sigma_mu = data_SN["sigma_mu"]
    chol_cov = data_SN["cov"]
    
    mu_model = mu(model, z, params...)
    
    # Handle NaN/numerical failures gracefully
    if any(isnan, mu_model)
        return -Inf
    end
    
    diff = mu_data .- mu_model
    
    if chol_cov !== nothing
        # Full Covariance: diff^T * C^-1 * diff
        # chol_cov \ diff solves C * x = diff using the pre-computed Cholesky factor
        chi2 = dot(diff, chol_cov \ diff)
    else
        # Diagonal only
        chi2 = sum((diff ./ sigma_mu).^2)
    end
    
    return -0.5 * chi2
end

end
