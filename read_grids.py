import glob
import pickle
import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from igimf import plots as plts
from create_interpolant import LinearAndNearestNeighbor_GCI
plots = plts.Plots()

txts = glob.glob('grid/resolution15/*pkl')
#txts.remove('.DS_Store')
#txts.remove('resolution50')
df = pd.DataFrame({col:[] for col in ['SFR', 'metal_mass_fraction',
                                      'mass_star', 'IGIMF']})

print('Importing pickle files')
for txt in txts:
    df_txt = pickle.load(open(txt, 'rb'))
    df = pd.concat([df,df_txt])
print('Pickle files imported')
#df_grids = copy(df)

SFR_v = np.unique(df['SFR'])
metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
mstar_v = np.unique(df['mass_star'])

SFR_v_fit = np.logspace(np.log10(np.min(SFR_v)), np.log10(np.max(SFR_v)))
mstar_v_fit = np.logspace(np.log10(np.min(mstar_v)), np.log10(np.max(mstar_v)))
metal_mass_fraction_v_fit = np.logspace(np.log10(np.min(metal_mass_fraction_v)), np.log10(np.max(metal_mass_fraction_v)))

#get_IGIMF = IGIMF_interpolation({'metal_mass_fraction':0.0134, 'SFR':1})
#IGIMF = get_IGIMF(mass_star)


#Retrieve a value
df.loc[(df['SFR']==SFR_v[0]) & 
       (df['metal_mass_fraction']==metal_mass_fraction_v[0])]
       
#plots.sIMF_subplot(metal_mass_fraction_v, self.Mecl_v, self.mstar_v, self.IMF_Z_v_list)
## No IMF for SFR < 5e-5, i.e. Migal ~<6e5
fig1, ax1 = plt.subplots(1,1, figsize=(7,5))
for m in metal_mass_fraction_v:
    #num_colors=len(metal_mass_fraction_v)
    #currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
    #currentColor = itertools.cycle(currentColors)
    grid_sel = df.loc[(df['SFR']==SFR_v[10]) & (df['metal_mass_fraction']==m)]
    ax1.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])
plt.show(block=False)
fig1.savefig('figs/IGIMFbyZ_fixedSFR.pdf')

fig2, ax2 = plt.subplots(1,1, figsize=(7,5))  
for s in SFR_v:
    for m in metal_mass_fraction_v:
        grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==m)]
        ax2.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])
plt.show(block=False)
fig2.savefig('figs/IGIMFbyZ_andbySFR.pdf')
       
fig3, ax3 = plt.subplots(1,1, figsize=(7,5))
for s in SFR_v:
    #num_colors=len(metal_mass_fraction_v)
    #currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
    #currentColor = itertools.cycle(currentColors)
    grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==metal_mass_fraction_v[10])]
    ax3.loglog(grid_sel['mass_star'], grid_sel['IGIMF'])
plt.show(block=False)
fig3.savefig('figs/IGIMFbySFR_fixedZ.pdf')

print('Creating interpolation')
IGIMF_interp = LinearAndNearestNeighbor_GCI(
    df=df,
    ycol='IGIMF',
    tf_funs={
        #'metallicity':lambda x:np.log10(x), 'metallicity_prime':lambda x:1/(x*np.log(10)),
        'Mecl':lambda x:np.log10(x), 'Mecl_prime':lambda x:1/(x*np.log(10)),
        #'IGIMF':lambda y:np.log10(y), 'IGIMF_inv':lambda y:10**y, 'IGIMF_prime':lambda y:1/(y*np.log(10))
    }, 
    name='IGIMF')
print(IGIMF_interp)
print('Interpolation created')

print('Plotting IGIMF fit')
fig4, ax4 = plt.subplots(1,1, figsize=(7,5))  
for j,SFR_v_i in enumerate(SFR_v_fit):
    for k, Z_v_i in enumerate(metal_mass_fraction_v_fit):
        IGIMF_hats = np.zeros_like(mstar_v_fit)
        for i,mstar_v_i in enumerate(mstar_v_fit):
            df_test = pd.DataFrame({'SFR':[SFR_v_i], 'metal_mass_fraction':[Z_v_i],'Mecl':[10**5],'mass_star':[mstar_v_i]})
            IGIMF_hats[i] = IGIMF_interp(df_test)
        ax4.loglog(mstar_v_fit,IGIMF_hats,'-o')
plt.show(block=False)
print('Plotting of IGIMF fit completed')
fig4.savefig('figs/IGIMFbyZ_fixedSFR.pdf')



print('Plotting of the 3D figure')
#plots.IGIMF_3D_plot(df, SFR_v, metal_mass_fraction_v, mstar_v, by_v='SFR', col_ax_idx=10)
plots.IGIMF_3D_plot(df, SFR_v, metal_mass_fraction_v, mstar_v, by_v='SFR', col_ax_idx=10, azim_rot=-120, elev_rot=20)
#plots.IGIMF_3Dlines_plot(df, SFR_v, metal_mass_fraction_v, mstar_v)
print('3D figure plotted')

#pickle.dump(IGIMF_interp,open('IGIMFinterpolant.pkl','wb'))