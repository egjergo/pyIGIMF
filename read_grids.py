import glob
import pickle
import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from igimf import plots as plts
from FriendlyInterpolants import LinearAndNearestNeighbor_FI
plots = plts.Plots()

txts = glob.glob('grid/resolution50/*pkl')
#txts.remove('.DS_Store')
#txts.remove('resolution50')
df = pd.DataFrame({col:[] for col in ['SFR', 'metal_mass_fraction',
                                      'mass_star', 'IGIMF']})

for i,txt in enumerate(txts):
    df_txt = pickle.load(open(txt, 'rb'))
    df = df.append(df_txt)
    if i==5: break # TO BE REMOVED IN THE FUTURE!!!
df.dropna(inplace=True) # some rows have na values
df.index = np.arange(len(df)) # avoid repeated index from appending dataframes
print('df.shape: %s\n'%str(df.shape))
print(df.head())

fi_igimf_full = LinearAndNearestNeighbor_FI(
    df = df,
    tf_funs = {},
    xcols = ['SFR','metal_mass_fraction','mass_star'],
    ycol = 'IGIMF',
    name = 'fi_igimf_full')
print(fi_igimf_full)

SFR_v = np.unique(df['SFR'])
metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
mstar_v = np.unique(df['mass_star'])

# Retrieve a value
fi_igimf_marginal = lambda mass_star: fi_igimf_full(pd.DataFrame({
    'mass_star': np.array(mass_star),
    'metal_mass_fraction': 0.0134,
    'SFR': 1}))
mass_star_query = df[(df['SFR']==df['SFR'][0])&(df['metal_mass_fraction']==df['metal_mass_fraction'][0])]['mass_star']
yhat = fi_igimf_marginal(mass_star=mass_star_query)
print(yhat)
import sys; sys.exit(0)

#plots.sIMF_subplot(metal_mass_fraction_v, self.Mecl_v, self.mstar_v, self.IMF_Z_v_list)
## No IMF for SFR < 5e-5, i.e. Migal ~<6e5
fig1, ax1 = plt.subplots(1,1, figsize=(7,5))
for m in metal_mass_fraction_v:
    #num_colors=len(metal_mass_fraction_v)
    #currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
    #currentColor = itertools.cycle(currentColors)
    grid_sel = df.loc[(df['SFR']==SFR_v[20]) & (df['metal_mass_fraction']==m)]
    ax1.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])

fig2, ax2 = plt.subplots(1,1, figsize=(7,5))  
for s in SFR_v:
    for m in metal_mass_fraction_v:
        grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==m)]
        ax2.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])
      
      
fig3, ax3 = plt.subplots(1,1, figsize=(7,5))
for s in SFR_v:
    #num_colors=len(metal_mass_fraction_v)
    #currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
    #currentColor = itertools.cycle(currentColors)
    grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==metal_mass_fraction_v[40])]
    ax3.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])
      
#plots.IGIMF_3D_plot(df, SFR_v, metal_mass_fraction_v, mstar_v, by_v='SFR', col_ax_idx=40)
#plots.IGIMF_3D_plot(df, SFR_v, metal_mass_fraction_v, mstar_v, by_v='SFR', col_ax_idx=15, azim_rot=-120, elev_rot=20)
#plots.IGIMF_3Dlines_plot(df, SFR_v, metal_mass_fraction_v, mstar_v)
        