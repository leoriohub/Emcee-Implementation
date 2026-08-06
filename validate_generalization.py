import numpy as np
import sys
import os

# Add current dir and src to path
sys.path.append(os.getcwd())

from cosmology_lcdm import log_likelihood_CC as ll_cc_old, log_likelihood_SN as ll_sn_old
from src.data_utils import load_CC, load_PPSH0ES
from src.models import LCDMModel
from src.likelihoods import log_likelihood_CC as ll_cc_new, log_likelihood_SN as ll_sn_new

def validate():
    # Load data once
    cc_path = "Cosmic_chronometers_data.tex"
    sn_path = "Pantheon+SH0ES.dat.txt"
    
    data_CC = load_CC(cc_path)
    data_SN = load_PPSH0ES(sn_path)
    
    # Test point
    params = [70.0, 0.3]
    
    # Old calculation
    print("--- Old Implementation ---")
    val_cc_old = ll_cc_old(params, data_CC)
    val_sn_old = ll_sn_old(params, data_SN)
    print(f"CC Likelihood: {val_cc_old}")
    print(f"SN Likelihood: {val_sn_old}")
    
    # New calculation
    print("\n--- New Implementation ---")
    model = LCDMModel()
    val_cc_new = ll_cc_new(model, params, data_CC)
    val_sn_new = ll_sn_new(model, params, data_SN)
    print(f"CC Likelihood: {val_cc_new}")
    print(f"SN Likelihood: {val_sn_new}")
    
    # Compare
    diff_cc = abs(val_cc_old - val_cc_new)
    diff_sn = abs(val_sn_old - val_sn_new)
    
    print(f"\nDifferences: CC={diff_cc}, SN={diff_sn}")
    
    if diff_cc < 1e-6 and diff_sn < 1e-6:
        print("SUCCESS: New implementation matches old results!")
    else:
        print("FAILURE: Results do not match.")

if __name__ == "__main__":
    validate()
