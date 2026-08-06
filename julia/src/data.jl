module DataUtils

using LinearAlgebra

export load_CC, load_PPSH0ES

"""
Loads Cosmic Chronometers data from a .tex file.
"""
function load_CC(filename::String)
    z_list = Float64[]
    H_list = Float64[]
    sH_list = Float64[]
    
    # Regex matches LaTeX table line: $z$ & $H \pm \sigma$
    pattern = r"\$([\d\.]+)\$\s*&\s*\$([\d\.]+)\s*\\pm\s*([\d\.]+)\$"
    
    open(filename, "r") do io
        for line in eachline(io)
            m = match(pattern, line)
            if m !== nothing
                push!(z_list, parse(Float64, m.captures[1]))
                push!(H_list, parse(Float64, m.captures[2]))
                push!(sH_list, parse(Float64, m.captures[3]))
            end
        end
    end
    
    return Dict(
        "z" => z_list,
        "Hz" => H_list,
        "sigma_Hz" => sH_list
    )
end

"""
Loads Pantheon+ covariance matrix from a flat column text file and computes Cholesky.
"""
function load_cov(filename::String)
    open(filename, "r") do io
        n_str = readline(io)
        N = parse(Int, n_str)
        flat_cov = [parse(Float64, line) for line in eachline(io)]
        cov_mat = reshape(flat_cov, N, N)
        # Compute Cholesky decomposition (for quick linear solving: C \ x)
        return cholesky(Symmetric(cov_mat))
    end
end

"""
Loads Pantheon+SH0ES supernova data.
Optionally takes the path to the covariance matrix file for full-covariance mode.
"""
function load_PPSH0ES(filename::String, cov_filename::Union{String, Nothing}=nothing)
    z_list = Float64[]
    mu_list = Float64[]
    mu_err_list = Float64[]
    
    lines = readlines(filename)
    # Find header line (first line that does not start with #)
    header_idx = findfirst(l -> !startswith(strip(l), "#"), lines)
    if header_idx === nothing
        error("Header not found in Supernova data file: $filename")
    end
    
    headers = split(strip(lines[header_idx]))
    z_col = findfirst(==("zHD"), headers)
    mu_col = findfirst(==("MU_SH0ES"), headers)
    err_col = findfirst(==("MU_SH0ES_ERR_DIAG"), headers)
    
    if z_col === nothing || mu_col === nothing || err_col === nothing
        error("Required columns (zHD, MU_SH0ES, MU_SH0ES_ERR_DIAG) not found in header: $headers")
    end
    
    for i in (header_idx+1):length(lines)
        line = strip(lines[i])
        if startswith(line, "#") || isempty(line)
            continue
        end
        parts = split(line)
        push!(z_list, parse(Float64, parts[z_col]))
        push!(mu_list, parse(Float64, parts[mu_col]))
        push!(mu_err_list, parse(Float64, parts[err_col]))
    end
    
    chol = nothing
    if cov_filename !== nothing && isfile(cov_filename)
        println("Loading covariance matrix from $cov_filename and computing Cholesky factorization...")
        chol = load_cov(cov_filename)
        println("Cholesky factorization completed successfully.")
    end
    
    return Dict(
        "z" => z_list,
        "mu" => mu_list,
        "sigma_mu" => mu_err_list,
        "cov" => chol
    )
end

end
