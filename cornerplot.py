import numpy as np
import matplotlib.pyplot as plt
import corner

def plot_cosmo_corner(samples, labels, main_color="#e0b72f"):
    
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
   
    plt.show() 