module Models

using ..PhysicsUtils

export AbstractCosmologyModel, LCDM, wCDM, CPL, DGP, JBP, GCG, IDE, LogW, Starobinsky, BransDicke
export H, mu

abstract type AbstractCosmologyModel end

# ── LCDM ──────────────────────────────────────────────────────────────────
struct LCDM <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
LCDM() = LCDM("LCDM", ["H0", "Omega_m"], [(40.0, 100.0), (0.0, 1.0)])

function H(model::LCDM, z::Real, H0::Real, Omega_m::Real)
    return H0 * sqrt(Omega_m * (1.0 + z)^3 + (1.0 - Omega_m))
end

# ── wCDM ──────────────────────────────────────────────────────────────────
struct wCDM <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
wCDM() = wCDM("wCDM", ["H0", "Omega_m", "w"], [(40.0, 100.0), (0.0, 1.0), (-2.0, 0.0)])

function H(model::wCDM, z::Real, H0::Real, Omega_m::Real, w::Real)
    om_m = Omega_m * (1.0 + z)^3
    om_de = (1.0 - Omega_m) * (1.0 + z)^(3.0 * (1.0 + w))
    return H0 * sqrt(om_m + om_de)
end

# ── CPL ───────────────────────────────────────────────────────────────────
struct CPL <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
CPL() = CPL("CPL", ["H0", "Omega_m", "w0", "wa"], [(40.0, 100.0), (0.0, 1.0), (-2.0, 0.0), (-3.0, 3.0)])

function H(model::CPL, z::Real, H0::Real, Omega_m::Real, w0::Real, wa::Real)
    om_m = Omega_m * (1.0 + z)^3
    z_ratio = z / (1.0 + z)
    om_de = (1.0 - Omega_m) * (1.0 + z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z_ratio)
    return H0 * sqrt(om_m + om_de)
end

# ── DGP Brane-world ───────────────────────────────────────────────────────
struct DGP <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
DGP() = DGP("DGP Brane-world", ["H0", "Omega_m"], [(40.0, 100.0), (0.0, 1.0)])

function H(model::DGP, z::Real, H0::Real, Omega_m::Real)
    term = Omega_m * (1.0 + z)^3 + (1.0 - Omega_m)^2 / 4.0
    Ez = 0.5 * (1.0 - Omega_m) + sqrt(max(term, 0.0))
    return H0 * Ez
end

# ── JBP w(z) ──────────────────────────────────────────────────────────────
struct JBP <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
JBP() = JBP("JBP w(z)", ["H0", "Omega_m", "w0", "wa"], [(40.0, 100.0), (0.1, 0.5), (-2.0, 0.0), (-5.0, 5.0)])

function H(model::JBP, z::Real, H0::Real, Omega_m::Real, w0::Real, wa::Real)
    exp_term = 1.5 * wa * z^2 / (1.0 + z)^2
    f_z = (1.0 + z)^(3.0 * (1.0 + w0)) * exp(exp_term)
    Ez_sq = Omega_m * (1.0 + z)^3 + (1.0 - Omega_m) * f_z
    return H0 * sqrt(max(Ez_sq, 0.0))
end

# ── Generalized Chaplygin Gas (GCG) ───────────────────────────────────────
struct GCG <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
GCG() = GCG("Chaplygin Gas", ["H0", "As", "alpha"], [(40.0, 100.0), (0.0, 1.0), (-1.0, 2.0)])

function H(model::GCG, z::Real, H0::Real, As::Real, alpha::Real)
    term = As + (1.0 - As) * (1.0 + z)^(3.0 * (1.0 + alpha))
    Ez = (max(term, 1e-9))^(1.0 / (2.0 * (1.0 + alpha)))
    return H0 * Ez
end

# ── Interacting DE (IDE) ──────────────────────────────────────────────────
struct IDE <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
IDE() = IDE("Interacting DE", ["H0", "Omega_m", "xi"], [(40.0, 100.0), (0.01, 0.99), (-0.5, 0.5)])

function H(model::IDE, z::Real, H0::Real, Omega_m::Real, xi::Real)
    Ez_sq = Omega_m * (1.0 + z)^(3.0 + xi) + (1.0 - Omega_m)
    return H0 * sqrt(max(Ez_sq, 0.0))
end

# ── Logarithmic w(z) ──────────────────────────────────────────────────────
struct LogW <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
LogW() = LogW("Log-w Parameterization", ["H0", "Omega_m", "w0", "w1"], [(40.0, 100.0), (0.1, 0.5), (-2.0, 0.0), (-2.0, 2.0)])

function H(model::LogW, z::Real, H0::Real, Omega_m::Real, w0::Real, w1::Real)
    lnz = log1p(z)
    f_z = (1.0 + z)^(3.0 * (1.0 + w0)) * exp(1.5 * w1 * lnz^2)
    Ez_sq = Omega_m * (1.0 + z)^3 + (1.0 - Omega_m) * f_z
    return H0 * sqrt(max(Ez_sq, 0.0))
end

# ── Starobinsky f(R) ──────────────────────────────────────────────────────
struct Starobinsky <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
Starobinsky() = Starobinsky("Starobinsky f(R)", ["H0", "Omega_m", "alpha"], [(40.0, 100.0), (0.1, 0.5), (-1.0, 1.0)])

function H(model::Starobinsky, z::Real, H0::Real, Omega_m::Real, alpha::Real)
    Ez_sq = Omega_m * (1.0 + z)^3 + (1.0 - Omega_m) * (1.0 + alpha * z^2)
    return H0 * sqrt(max(Ez_sq, 0.0))
end

# ── Brans-Dicke MG ────────────────────────────────────────────────────────
struct BransDicke <: AbstractCosmologyModel
    name::String
    param_names::Vector{String}
    param_bounds::Vector{Tuple{Float64, Float64}}
end
BransDicke() = BransDicke("Brans-Dicke MG", ["H0", "Omega_m", "epsilon"], [(40.0, 100.0), (0.1, 0.5), (-0.5, 0.5)])

function H(model::BransDicke, z::Real, H0::Real, Omega_m::Real, epsilon::Real)
    om_m = Omega_m * (1.0 + z)^(3.0 - epsilon)
    om_de = (1.0 - Omega_m)
    Ez_sq = om_m + om_de
    return H0 * sqrt(max(Ez_sq, 0.0))
end

# ── Generic Distance Modulus Helper ───────────────────────────────────────
"""
Generic distance modulus calculation.
Uses multiple dispatch to evaluate H(z) depending on the model type.
"""
function mu(model::AbstractCosmologyModel, z_array::Vector{<:Real}, params...)
    H_closure = (z_val, p...) -> H(model, z_val, p...)
    dL = dl_array(H_closure, z_array, params...)
    return distance_modulus(dL)
end

end
