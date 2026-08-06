import numpy as np
import matplotlib.pyplot as plt
import corner

def plot_cosmo_corner(samples, labels, main_color="#e0b72f", show=True):
    """
    Standardized corner plot for cosmological parameters with custom styling.
    """
    hist_kwargs_style = {
        "fill": True,
        "alpha": 0.6,
        "color": main_color,
        "edgecolor": "black"
    }

    contour_kwargs_style = {
        "colors": "black", 
        "linewidths": 1
    }

    quantile_kwargs_style = {
        "color": "black",    
        "linestyles": "dashed"
    }

    sigma_levels = np.array([1.0, 2.0])
    levels_2d = 1.0 - np.exp(-0.5 * sigma_levels**2)

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        
        fill_contours=True,
        plot_datapoints=False,
        levels=levels_2d,
        
        color=main_color,
        
        hist_kwargs=hist_kwargs_style,        
        contour_kwargs=contour_kwargs_style,  
        quantile_kwargs=quantile_kwargs_style,
        
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 14}
    )
   
    if show:
        plt.show()
    return fig

def compare_models_corner(chains_list, labels_list, names_list, colors=None, show=True):
    """
    Overlays corner plots for multiple models.
    chains_list: list of sample arrays
    labels_list: list of parameter label lists (must share common first params for useful overlap)
    names_list: list of model names for legend
    """
    if colors is None:
        colors = ["#e0b72f", "#2f7be0", "#e02f2f", "#2fe07b", "#7b2fe0"]

    fig = None
    for i, (samples, labels, name) in enumerate(zip(chains_list, labels_list, names_list)):
        color = colors[i % len(colors)]
        
        # Only overlay common parameters (e.g., H0, Omega_m)
        # Assuming most models start with [H0, Omega_m, ...]
        common_count = min(len(labels), 2) # Plot first 2 common params usually
        sub_samples = samples[:, :common_count]
        sub_labels = labels[:common_count]
        
        if fig is None:
            fig = corner.corner(
                sub_samples,
                labels=sub_labels,
                color=color,
                fill_contours=True,
                plot_datapoints=False,
                label_kwargs={"fontsize": 12}
            )
        else:
            corner.corner(
                sub_samples,
                labels=sub_labels,
                color=color,
                fill_contours=True,
                plot_datapoints=False,
                fig=fig
            )
    
    # Add custom legend
    plt.legend(names_list, loc="upper right", bbox_to_anchor=(1, common_count))
    if show:
        plt.show()
    return fig

def plot_theory_comparison(models_list, params_list, data_CC=None, data_SN=None, show=True):
    """
    Plots H(z) and mu(z) comparisons for different models against data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    z_grid = np.linspace(0, 2.5, 100)
    
    colors = ["#e0b72f", "#2f7be0", "#e02f2f", "#2fe07b", "#7b2fe0"]

    # 1. Hubble Plot
    if data_CC is not None:
        axes[0].errorbar(data_CC["z"], data_CC["Hz"], yerr=data_CC["sigma_Hz"], 
                        fmt='o', color='black', alpha=0.5, label='Data (CC)')

    for i, (model, params) in enumerate(zip(models_list, params_list)):
        Hz = model.H(z_grid, *params)
        axes[0].plot(z_grid, Hz, label=model.name, color=colors[i % len(colors)], lw=2)

    axes[0].set_xlabel("Redshift z", fontsize=12)
    axes[0].set_ylabel(r"$H(z)$ [km/s/Mpc]", fontsize=12)
    axes[0].legend()
    axes[0].set_title("Hubble Expansion Rate Comparison")

    # 2. Distance Modulus Plot
    if data_SN is not None:
        axes[1].errorbar(data_SN["z"], data_SN["mu"], yerr=data_SN["sigma_mu"], 
                        fmt='.', color='black', alpha=0.2, label='Data (SN)')

    for i, (model, params) in enumerate(zip(models_list, params_list)):
        mu = model.mu(z_grid, *params)
        axes[1].plot(z_grid, mu, color=colors[i % len(colors)], lw=2)

    axes[1].set_xlabel("Redshift z", fontsize=12)
    axes[1].set_ylabel(r"$\mu(z)$", fontsize=12)
    axes[1].set_title("Distance Modulus Comparison")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_extensions_1d(chains_list, labels_list, names_list, colors=None, show=True):
    """
    Plots 1D histograms for model-specific parameters (extension parameters).
    Typically assumes first 2 parameters are H0 and Omega_m.
    """
    if colors is None:
        colors = ["#e0b72f", "#2f7be0", "#e02f2f", "#2fe07b", "#7b2fe0"]

    # Identify chains that have more than 2 parameters
    ext_data = []
    for samples, labels, name, color in zip(chains_list, labels_list, names_list, colors):
        if len(labels) > 2:
            for j in range(2, len(labels)):
                ext_data.append({
                    'samples': samples[:, j],
                    'label': labels[j],
                    'model': name,
                    'color': color
                })

    if not ext_data:
        print("No extension parameters found (H0, Omega_m are considered standard).")
        return None

    n_ext = len(ext_data)
    fig, axes = plt.subplots(1, n_ext, figsize=(4 * n_ext, 4), squeeze=False)
    
    for i, data in enumerate(ext_data):
        ax = axes[0, i]
        ax.hist(data['samples'], bins=30, density=True, histtype='step', 
                color=data['color'], lw=2, label=data['model'])
        ax.set_xlabel(data['label'], fontsize=14)
        title_label = data['label']
        if not title_label.startswith("$"):
            title_label = f"${title_label}$"
        ax.set_title(f"Extension: {title_label}", fontsize=12)
        ax.legend(fontsize=10)
        
    plt.tight_layout()
    if show:
        plt.show()
    return fig
