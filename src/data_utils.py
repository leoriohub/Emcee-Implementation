import re
import numpy as np

def load_CC(filename):
    """Loads Cosmic Chronometers data from a .tex file."""
    z_list, H_list, sH_list = [], [], []
    pattern = re.compile(
        r"\$([\d\.]+)\$.*?\$([\d\.]+)\s*\\pm\s*([\d\.]+)\$"
    )
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                z, H, sH = map(float, match.groups())
                z_list.append(z)
                H_list.append(H)
                sH_list.append(sH)
    
    return {
        "z": np.array(z_list),
        "Hz": np.array(H_list),
        "sigma_Hz": np.array(sH_list)
    }

def load_PPSH0ES(filename):
    """Loads Pantheon+SH0ES supernova data."""
    with open(filename, 'r') as f:
        header_line = ''
        for line in f:
            if not line.strip().startswith('#'):
                header_line = line.strip()
                break
        column_names = header_line.split()

    data = np.genfromtxt(filename, names=column_names, comments='#', skip_header=1)

    return {
        "z": data["zHD"],
        "mu": data["MU_SH0ES"],
        "sigma_mu": data["MU_SH0ES_ERR_DIAG"]
    }
