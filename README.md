# pyIGIMF 
Fast computation of the integrated galaxy-wide initial mass function (IGIMF)

Direct queries to Eda Gjergo (Nanjing University) <GalacticCEM@gmail.com>

## Default use

From a Terminal
```
pip install igimf
```

You can choose the linear or logistic alpha1 prescription.

In a script or directly in a Python console:
```
import numpy as np
from igimf import classes as inst

Z_solar = 0.0142

# input (edit as needed)
SFR = 2 # Msun/yr
metal_mass_fraction = 0.1 * Z_solar
mass_star = np.logspace(np.log10(0.08), np.log10(150), 100)
alpha1slope = 'logistic' # or 'linear' 

o_IGIMF = inst.IGIMF(metal_mass_fraction=metal_mass_fraction, SFR=SFR, alpha1slope=alpha1slope)

igimf_v = o_IGIMF.IGIMF_func(mass_star)
```

o_IGIMF creates an instance of the IGIMF class. 
This object contains IGIMF_func, the IGIMF function computed at the given SFR and metal mass fraction.
IGIMF_func can be applied to any stellar mass vector.
In the above script, igimf_v is the IGIMF function computed for the stallar mass vector mass_star


## Setup
```
conda env create -f environment.yaml
conda activate igimf
```

## Run the minimum working example automatically
```
python examples/mwe.py
```
