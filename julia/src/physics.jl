module PhysicsUtils

export C_LIGHT, dl_array, distance_modulus

# Speed of light in km/s
const C_LIGHT = 299792.458

"""
Fast O(1) differentiable linear interpolation on a regularly spaced grid starting at 0.0.
"""
function interpolate_regular_grid(dx::Real, y_grid::Vector{T}, x_queries::Vector{<:Real}) where {T}
    N = length(y_grid)
    y_queries = Vector{T}(undef, length(x_queries))
    for (i, x) in enumerate(x_queries)
        idx_f = x / dx
        idx = floor(Int, idx_f) + 1
        if idx < 1
            y_queries[i] = y_grid[1]
        elseif idx >= N
            y_queries[i] = y_grid[end]
        else
            t = idx_f - (idx - 1)
            y_queries[i] = (1.0 - t) * y_grid[idx] + t * y_grid[idx+1]
        end
    end
    return y_queries
end

"""
Computes the luminosity distance dL(z) = (1+z) * DM(z) for an array of redshifts
using a fast regularly-spaced grid cumulative integration.
"""
function dl_array(H_func::Function, z_array::Vector{<:Real}, params...)
    z_max = maximum(z_array)
    if z_max == 0
        return zero(z_array)
    end
    
    npoints = 5000
    dx = z_max / (npoints - 1)
    
    # Evaluate H(z) on the grid
    Hz_grid = [H_func(i * dx, params...) for i in 0:(npoints-1)]
    
    # Cumulative integration
    integral_grid = Vector{eltype(Hz_grid)}(undef, npoints)
    integral_grid[1] = 0.0
    for i in 2:npoints
        integral_grid[i] = integral_grid[i-1] + dx * 0.5 * (C_LIGHT/Hz_grid[i] + C_LIGHT/Hz_grid[i-1])
    end
    
    # Interpolate
    DM = interpolate_regular_grid(dx, integral_grid, z_array)
    dL = (1.0 .+ z_array) .* DM
    return max.(dL, 1e-9)
end

"""
Computes the distance modulus mu = 5 log10(dL) + 25.
"""
function distance_modulus(dL)
    return 5.0 .* log10.(dL) .+ 25.0
end

end
