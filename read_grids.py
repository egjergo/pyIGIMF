import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from igimf import plots as plts
plots = plts.Plots()

txts = os.listdir('grid')
txts.remove('.DS_Store')
df = pd.DataFrame({col:[] for col in ['SFR', 'metal_mass_fraction',
                                      'mass_star', 'IGIMF']})

for txt in txts:
    df_txt = pickle.load(open('grid/'+txt, 'rb'))
    df = df.append(df_txt)

SFR_v = np.unique(df['SFR'])
metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
mstar_v = np.unique(df['mass_star'])

#Retrieve a value
df.loc[(df['SFR']==SFR_v[0]) & 
       (df['metal_mass_fraction']==metal_mass_fraction_v[0])]

# No IMF for SFR < 5e-5, i.e. Migal ~<6e5
fig1, ax1 = plt.subplots(1,1, figsize=(7,5))
for m in metal_mass_fraction_v:
    grid_sel = df.loc[(df['SFR']==SFR_v[18]) & (df['metal_mass_fraction']==m)]
    ax1.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])

fig2, ax2 = plt.subplots(1,1, figsize=(7,5))  
for s in SFR_v:
    for m in metal_mass_fraction_v:
        grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==m)]
        ax2.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])
        
plots.IGIMF_3D_plot(df, SFR_v, metal_mass_fraction_v, mstar_v, by_v='SFR', col_ax_idx=20)