import numpy as np
import pandas as pd
from igimf import classes as inst

Z_solar = 0.0142

# input (edit as needed)
SFR = 2 # Msun/yr
metal_mass_fraction = 0.1 * Z_solar
mass_star = np.logspace(np.log10(0.08), np.log10(150), 100)
alpha1slope = 'logistic' # or 'linear' 

o_IGIMF = inst.IGIMF(metal_mass_fraction=metal_mass_fraction, SFR=SFR, alpha1slope=alpha1slope)

igimf_v = o_IGIMF.IGIMF_func(mass_star)