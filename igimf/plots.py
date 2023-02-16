import numpy as np
from igimf import util

class Plots:
    # Plotting functions
    def Migal_plot(self, M_igal_v, SFR, downsizing_time):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax1 = plt.subplots(1,1, figsize=(7,5))
        ax0 = ax1.twinx()
        ax0.loglog(M_igal_v, SFR, linewidth=3, color='tab:red')
        ax0.set_ylabel(f'SFR [{Msun}/yr]', fontsize=15, color='tab:red')
        ax0.set_xlabel(r'$M_{igal}$ '+f'[{Msun}]', fontsize=15)
        ax1.semilogx(M_igal_v, downsizing_time, linewidth=3, color='tab:blue')
        ax1.set_ylabel(r'$\Delta\tau$ [Gyr]', fontsize=15, color='tab:blue')
        ax1.set_xlabel(r'$M_{igal}$ '+f'[{Msun}]', fontsize=15)
        #ax.set_ylim(1e-8,1)
        ax0.tick_params(width=2, axis='both', labelsize=15)
        ax1.tick_params(width=2, axis='both', labelsize=15)
        fig.tight_layout()
        #plt.savefig(f'figs/Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)
        return None
        
    def ECMF_plot(self, Mecl_v, ECMF_v, SFR):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.loglog(Mecl_v, ECMF_v, linewidth=3, color='navy')
        ax.scatter(Mecl_v, ECMF_v, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$\xi_{ECMF}$'+f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$ [%s]'%(Msun), fontsize=15)
        plt.title(r'$\,$ SFR = %.2e [%s/yr]' %(SFR, Msun), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/ECMF_plot_SFR{SFR:.2e}.pdf', bbox_inches='tight')
        #plt.show(block=False)
            
    def beta_ECMF_bySFR_plot(self, SFR_v, beta_ECMF_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.semilogx(SFR_v, beta_ECMF_v, linewidth=3, color='navy')
        ax.scatter(SFR_v, beta_ECMF_v, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$\beta_{ECMF}$', fontsize=15)
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        #ax.set_title(r'[Z] = %.2f' %(Z), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/beta_ECMF_bySFR.pdf', bbox_inches='tight')
        #plt.show(block=False)
               
    def MeclMax_bySFR_plot(self, SFR_v, MeclMax_list):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.loglog(SFR_v, MeclMax_list, linewidth=3, color='navy')
        ax.scatter(SFR_v, MeclMax_list, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$M_{\rm ecl,max}$ [%s]'%(Msun), fontsize=15)
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        #ax.set_title(r'[Z] = %.2f' %(Z), fontsize=15)
        #ax.set_ylim(5e-2,1e8)
        #ax.set_xlim(1e-4,1e3)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/MeclMax_bySFR_plot.pdf', bbox_inches='tight')
        #plt.show(block=False)
        
    def Mecl_power_beta_plot(self, Mecl_v, beta_ECMF_list):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        import colorcet as cc
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = cc.cm.isolum
        num_colors = len(beta_ECMF_list)
        Z = [[0,0],[0,0]]
        beta_ECMF_list = np.flip(beta_ECMF_list)
        levels = np.linspace(np.min(beta_ECMF_list), np.max(beta_ECMF_list),
                             100, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        for b in beta_ECMF_list:
            y = Mecl_v**(-b)
            color = next(currentColor)
            ax.loglog(Mecl_v, y, linewidth=3, c=color)
            #ax.scatter(Mecl_v, y, linewidth=3, c=color,s=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\xi_{ECMF}$'+f' [#/{Msun}]'+' not normalized',
                      fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$'+f' [{Msun}]', fontsize=15)
        ax.set_title(r'$k_{\rm ecl}=1$', fontsize=15)
        ax.set_ylim(1e-15,1e1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(.2)).set_label(
                                label=r'$\beta_{\rm ECMF}$',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_power_beta.pdf', bbox_inches='tight')
        #plt.show(block=False)
        
    def ECMF_plots(self, M_ecl_v, ECMF_v_list, SFR_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = plt.cm.get_cmap(name='inferno')
        num_colors = len(ECMF_v_list)
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, #109 
                             endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        SFR_colormap = (SFR_v)#np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        for i,ECMF in enumerate(ECMF_v_list):
            ax.loglog(M_ecl_v,ECMF, linewidth=3, c=next(currentColor))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\xi_{ECMF}$'+ f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$ [%s]' %(Msun), fontsize=15)
        #ax.set_ylim(1e-1,1e5)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}({\rm SFR})$'+f' [({Msun}/yr)]',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/ECMF_plots.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None
  
    def gwIMF_plots(self, star_v, gwIMF_bySFR_eval, SFR_v, metal_mass_fraction):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = plt.cm.get_cmap(name='magma')
        num_colors = len(gwIMF_bySFR_eval)
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), 100, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        SFR_colormap = (SFR_v)#np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        for i,gwIMF in enumerate(gwIMF_bySFR_eval):
            ax.loglog(star_v,gwIMF, linewidth=3, c=next(currentColor))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        metallicity = np.log10(metal_mass_fraction/0.0142)
        ax.set_title(f'[Z] = {metallicity:.2f}', fontsize=15)
        ax.set_ylabel(r'$\xi_{gwIMF}$'+f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'stellar mass [%s]' %(Msun), fontsize=15)
        #ax.set_ylim(1e-1,1e5)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}({\rm SFR})$',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/gwIMF_plots_Z{metallicity:.2f}.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None
    
    def IMF_plot(self, Mstar_v, IMF_v, Mtot, metallicity, SFR):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.loglog(Mstar_v, IMF_v, linewidth=3, color='navy')
        ax.scatter(Mstar_v, IMF_v, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$\xi_{\star}={\rm d} N_{\star}/{\rm d} m$'
                      +f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$M_{\rm star}$ [%s]'%(Msun), fontsize=15)
        plt.title(r'$M_{\rm ecl}$ = %.2e [%s],$\quad$ [Z] = %.2f' 
                  %(Mtot, Msun, metallicity), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/IMF_plot_Mecl{Mtot:.2e}_Z{metallicity:.2f}'+
                      f'_SFR{SFR:.2e}.pdf', bbox_inches='tight')
        #plt.show(block=False)
        
    def IMF_plots(self, mstar_v, IMF_v_list, Mecl_v, k_idx, massfrac):
        from matplotlib import pyplot as plt
        import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = cc.cm.CET_L20
        eff_Mecl_v = Mecl_v[k_idx]
        eff_IMF_v_list = np.array(IMF_v_list)[k_idx]
        num_colors = len(eff_Mecl_v)
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(eff_Mecl_v[0]), np.log10(eff_Mecl_v[-1]), num_colors, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        for i, IMF in enumerate(eff_IMF_v_list):
            ax.loglog(mstar_v, IMF, linewidth=3, c=next(currentColor))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\xi_{\star}={\rm d} N_{\star}/{\rm d} m$'
                      +f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$M_{\rm star}$ [%s]' %(Msun), fontsize=15)
        ax.set_ylim(1e-2,1e10)
        Z = np.log10(massfrac/0.0134)
        ax.set_title(r"[Z] = %.2f"%(Z), fontsize=15)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}(M_{\rm ecl})$'+f' [({Msun})]',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/IMF_plots_Z{Z:.2f}.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None
        
    def IMF_3D_plot(self, m_v, M_ecl_v, sIMF_func):
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        
        def z_func(m,M):
            return np.reshape([[sIMF_func[i](m_j) for m_j in m] for i,val in enumerate(M)], (len(m), len(M)))
        
        def resh(x):
            return np.reshape(list(x) * len(x), (len(x),len(x)))
        
        fig = plt.figure(figsize=(10,8))
        # syntax for 3-D projection
        ax = plt.axes(projection ='3d')
        m = resh(m_v)
        M = resh(M_ecl_v).T
        xi = z_func(m_v, M_ecl_v)
        
        # plotting
        #ax.plot3D(x, y, z, 'green')
        ax.plot_surface(np.log10(m), np.log10(M), np.ma.log10(xi), cmap ='plasma', linewidth=0.25)
        ax.set_xlabel(r'stellar mass $m_{\star}$ [$\log_{10}(M_{\odot})$]', fontsize=15)
        ax.set_ylabel(r'E. cluster mass $M_{\rm ecl}$ [$\log_{10}(M_{\odot})$]', fontsize=15)
        ax.set_zlabel(r'$\xi_{\star}={\rm d}N_{\star}/ {\rm d} m$'
                      +f' [#/{Msun}]', fontsize=15)
        ax.set_title(r'stellar IMF $\xi_{\star}(m_{\star},M_{\rm ecl},Z)$'+f' [#/{Msun}]', fontsize=17)
        fig.tight_layout()
        plt.show(block=False)
        #plt.savefig(f'figs/IMF_plot_3D.pdf', bbox_inches='tight')
    
    def IGIMF_3D_plot(self, df, SFR_v, metal_mass_fraction_v, mstar_v, 
                      by_v='SFR', col_ax_idx=10, azim_rot=-120, elev_rot=20):
        '''
        by_v can be "SFR" or "metal_mass_fraction"
        '''
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        
        Msun = r'$M_{\odot}$'
        if by_v == 'SFR':
            y_ax = SFR_v
            color_ax = metal_mass_fraction_v
            title = '[Z]'
            metallicity_val = np.log10(color_ax[col_ax_idx]/0.0134)
            units = f'[{Msun}/yr]'
        elif by_v == 'metal_mass_fraction':
            y_ax = metal_mass_fraction_v
            color_ax = SFR_v
            title = 'SFR'
            units = ''
        else:
            raise ValueError("set by_v either to 'SFR' or 'metal_mass_fraction'. ")
        
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection ='3d')
        x = np.outer(mstar_v, np.ones(len(y_ax)))
        y = np.outer(y_ax, np.ones(len(mstar_v))).T
        xi = np.reshape([df.loc[((df['mass_star']==ival) & (df[by_v]==jval) 
            & (df['metal_mass_fraction']==metal_mass_fraction_v[col_ax_idx])
            )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
            for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
        
        ## Setting a mask to exclude zero values
        #xi_mask = np.ma.masked_where(np.isnan(xi), xi)
        #xi_masked = xi.copy()
        #xi_masked[np.isnan(xi)] = -0.
        
        #ax.plot_surface(np.log10(x), np.log10(y), np.log10(xi_masked), cmap ='plasma', linewidth=0.25)
        ax.plot_surface(np.log10(x[:47,:47]), np.log10(y[:47,:47]), np.ma.log10(xi[:47,:47]), cmap ='plasma', linewidth=0.25)
        #ax.plot_surface(np.log10(x), np.log10(y), np.log10(xi_masked), cmap ='plasma', linewidth=0.25)
        ax.set_xlabel(r'stellar mass $m_{\star}$ [$\log_{10}(M_{\odot})$]', fontsize=15)
        ax.set_ylabel(f'{by_v}  {units}', fontsize=15)
        ax.set_zlabel(r'$\xi_{\rm IGIMF}={\rm d}N_{\star}/ {\rm d} m$'+
                      f'['+r'$\log_{10}({\rm #}/M_{\odot})$'+f']', fontsize=15)
        ax.set_title(f'{title} {metallicity_val:.2f}', fontsize=17)
        ax.azim = azim_rot
        ax.elev = elev_rot
        fig.tight_layout()
        plt.show(block=False)
        plt.savefig(f'figs/IGIMF_plot_3D.pdf', bbox_inches='tight')
     
    def IGIMF_3Dlines_plot(self, df, SFR_v, metal_mass_fraction_v, mstar_v):
        #from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import colorcet as cc
        import itertools
        mpl.rcParams['legend.fontsize'] = 10
        M = r'$M_{\odot}$'
        cm = cc.cm.CET_L8
        num_colors=len(metal_mass_fraction_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        fig = plt.figure(figsize=(10,8))
        ax = fig.gca(projection ='3d')       
        for m in metal_mass_fraction_v: 
            for s in SFR_v:
                grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==m)]
                ax.loglog(grid_sel['mass_star'], grid_sel['SFR'], grid_sel['IGIMF'], color=next(currentColor))
        fig.tight_layout()
        plt.show(block=False)
        plt.savefig(f'figs/IGIMF_plot_3Dlines.pdf', bbox_inches='tight')
        return None
       
    def sIMF_subplot(self, metallicity_v, Mecl_v, mstar_v, sIMF, res=20):
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_R2
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], 100,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 3,3 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in enumerate(metallicity_v):
                ax.annotate(r'$M_{\rm ecl}=$%.2e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                ax.loglog(mstar_v, sIMF[i][j], color=next(currentColor),
                          alpha=1)
                ax.set_ylim(5e-3,1e11)
                ax.set_xlim(2e-2,5e2)
        #for nr in range(3):
        for nr in range(nrow):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != 4-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\xi_{\star}={\rm d} N_{\star}/{\rm d} m$'+
                                  f' [#/{Msun}]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(
                                label=r'[Z]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/stellarIMF_subplots_Zcolorbar.pdf')



    def mw_sIMF_subplot(self, metallicity_v, Mecl_v, mstar_v, mw_sIMF, res=20):
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_R3
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], 100,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 3,3 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in enumerate(metallicity_v):
                ax.annotate(r'$M_{\rm ecl}=$%.2e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                ax.loglog(mstar_v, np.divide(mw_sIMF[i][j], Mecl_v[i]), color=next(currentColor),
                          alpha=0.8)
                ax.set_ylim(1e-8,1e3)
                ax.set_xlim(2e-2,5e2)
        #for nr in range(3):
        for nr in range(nrow):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != 4-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$m \xi_{\star}(m) / M_{\rm ecl} \propto \frac{{\rm d} N / {\rm d} \log_{10}m}{M_{\rm ecl}} \quad$ [#/$M_{\odot}$]', fontsize = 14)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(
                                label=r'[Z]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/massweighted_stellarIMF_subplots_Zcolorbar.pdf')

    def sIMF_subplot_Mecl(self, metallicity_v, Mecl_v, mstar_v, sIMF, res=20):
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        #from mpl_toolkits import mplot3d
        #from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        #cm = plt.cm.get_cmap(name='viridis')
        cm = cc.cm.CET_L20
        levels = np.linspace(np.log10(Mecl_v[0]), np.log10(Mecl_v[-1]), 100,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(Mecl_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 3, 3 #util.find_closest_prod(res)
        #fig, axs = plt.subplots(3, 3, figsize=(8,6))
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, M in enumerate(Mecl_v):
                ax.annotate(r'$[Z]=$%.2f'%(metallicity_v[i]), xy=(0.5, 0.9), 
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=.1)
                ax.loglog(mstar_v, sIMF[j][i], color=next(currentColor))
                ax.set_ylim(5e-3,1e11)
                ax.set_xlim(2e-2,5e2)
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        #axs[1,0].set_ylabel(r'$\xi_{stellar}$', fontsize = 15)
        #axs[2,1].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize = 15)
        axs[nrow//2,0].set_ylabel(r'$\xi_{\star}={\rm d} N_{\star}/{\rm d} m$'
                                  +f' [#/{Msun}]', fontsize = 15)
        axs[nrow-1,ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]', 
                                       fontsize = 15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(
                                label=r'$\log_{10}(M_{\rm ecl})$'+f' ['+
                                r'$\log_{10}$'+f'({Msun})]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/stellarIMF_subplots_Meclcolorbar.pdf')

    def Cook23_plot(self):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        import pandas as pd
        import numpy as np
        Cook23 = pd.read_csv('data/Cook23.dat', comment='#', sep='&')
        Cook23bin = pd.read_csv('data/Cook23bin.dat', comment='#', sep='&')
        Cook23bin10 = Cook23bin.loc[Cook23bin['SFR-Method']=='1-10Myr']
        Cook23binHalpha = Cook23bin.loc[Cook23bin['SFR-Method']=='Halpha']
        Dinnbier22 = pd.read_csv('data/Dinnbier22.dat', comment='#', sep=',')
        D22low = Dinnbier22.iloc[:3]
        D22high = Dinnbier22.iloc[3:]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.axhline(y=100, xmin=-4, xmax=1, linewidth=4, color='purple', label='IGIMF at birth')
        ax.semilogy(D22low['logSFR'], np.power(10, D22low['Gamma'])*100, linestyle='--', color='purple', linewidth=2)
        ax.semilogy(D22high['logSFR'], np.power(10, D22high['Gamma'])*100, linestyle='--', color='purple', linewidth=2)
        ax.fill_between(D22high['logSFR'], np.power(10, D22high['Gamma'])*100, np.power(10, D22low['Gamma'])*100, where=(np.power(10, D22low['Gamma'].to_numpy())*100<np.power(10, D22high['Gamma'].to_numpy())*100),  alpha=0.1, color='purple', label=r'DKA22 $<10$ Myr')
        ax.errorbar(Cook23binHalpha['sfrsig-bin'], Cook23binHalpha['Gamma'], xerr=Cook23binHalpha['sfrsig-u'], yerr=Cook23binHalpha['Gamma-u'], fmt='o', color='red', ecolor='red', elinewidth=3, capsize=0, label=r'C23 H$_{\alpha}$ $<10$ Myr', marker='s', alpha=0.8)
        ax.errorbar(Cook23bin10['sfrsig-bin'], Cook23bin10['Gamma'], xerr=Cook23bin10['sfrsig-u'], yerr=Cook23bin10['Gamma-u'], fmt='o', color='black', ecolor='black', elinewidth=3, capsize=0,label=r'C23 res $<10$ Myr', marker='s', alpha=0.8)
        #ax.errorbar(Cook23['logSFRsig'], Cook23['Gamma'], yerr=Cook23['Gamma-u'], fmt='o', color='blue', ecolor='blue', elinewidth=1, capsize=0,label=r'C23 lit', marker='o', alpha=0.4)
        ax.set_xlim(-3.7, 0.5)
        ax.set_ylim(2e-2, 2e2)
        ax.legend(loc='lower right', fontsize=12, frameon=True)
        ax.set_ylabel(r'$\Gamma$ (%)', fontsize=15)
        ax.set_xlabel(r'$\log(\Sigma_{\rm SFR})$ ($M_{\star} yr^{-1} kpc^{-2}$)', fontsize=15)
        fig.tight_layout()
        plt.savefig('figs/Cook23.pdf', bbox_inches='tight')
        plt.show(block=False)

    def Fig11_plot(self):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        CMOl = np.loadtxt('../data/Capuzzo-dolcetta17CMOl.csv', delimiter=',')
        CMOu = np.loadtxt('../data/Capuzzo-dolcetta17CMOu.csv', delimiter=',')
        SMBH = np.loadtxt('../data/Capuzzo-dolcetta17BH.csv', delimiter=',')
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        #ax.loglog(time, DTD_SNIa, color='blue', label='SNIa')
        #ax.legend(loc='best', frameon=False, fontsize=13)
        ax.scatter(CMOl[:,0], CMOl[:,1], color='red', marker='s', alpha=.7)
        ax.scatter(CMOu[:,0], CMOu[:,1], color='magenta', marker='^', alpha=.7)
        ax.scatter(SMBH[:,0], SMBH[:,1], color='black', marker='o', alpha=.7)
        ax.set_ylabel(r'$\log_{10}(M_{\rm CMO}/M_{\odot})$', fontsize=15)
        ax.set_xlabel(r'$\log_{10}(M_{\rm pgal}/M_{\odot})$', fontsize=15)
        
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
        ax.tick_params(width=1, length=10, axis='x', which='minor', 
                       bottom=True, top=True, direction='in')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
        ax.tick_params(width=2, length=15, axis='x', which='major', 
                       bottom=True, top=True, direction='in')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
        ax.tick_params(width=1, length=10, axis='x', which='minor', 
                       bottom=True, top=True, direction='in')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
        ax.tick_params(width=2, length=15, axis='x', which='major', 
                       bottom=True, top=True, direction='in')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        ax.set_ylim(0,11.5)
        ax.set_xlim(6, 13.7)
        fig.tight_layout()
        plt.savefig('figs/Fig11.pdf', bbox_inches='tight')
        #plt.show(block=False)
              
    def k_Z_plot(self, Z_massfrac_v, k_IMF_Z_list, m_max_Z_list, Mecl_v,
                 m_star_max=150):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cm = plt.cm.get_cmap(name='plasma')
        cm2 = plt.cm.get_cmap(name='plasma')
        num_colors = len(Z_massfrac_v)
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(Z_massfrac_v[0]/0.0142), 
                             np.log10(Z_massfrac_v[-1]/0.0142), 
                             100, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        SFR_colormap = (Z_massfrac_v) #np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        currentColors2 = [cm2(1.*i/num_colors) for i in range(num_colors)]
        currentColor2 = iter(currentColors2)
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(2,1, figsize=(5,7))
        #ax2 = ax.twinx()
        #ax.plot(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        #ax.scatter(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        for i,Z in enumerate(Z_massfrac_v):
            color = next(currentColor)
            ax[1].semilogx(Mecl_v, np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.4)
            #ax.plot((Mecl_v), (k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.4)
            color2 = next(currentColor2)
            ax[0].semilogx(Mecl_v, m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
            #ax2.plot((Mecl_v), m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
        ax[1].set_ylabel(r'$\log_{10}(k_{\rm IMF})$', fontsize=15)
        ax[1].set_xlabel(r'$\log_{10}(M_{\rm ecl})$[$M_{\odot}$]', fontsize=15)
        #ax.set_ylabel(r'$k_{\rm IMF}$', fontsize=15)
        #ax.set_xlabel(r'$M_{\rm ecl}$[$M_{\odot}$]', fontsize=15)
        ax[0].set_ylabel(r'$m_{\rm max}$', fontsize=15)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("top", size="5%", pad="10%")#, pack_start=True)
        #cax = divider.new_vertical(size="5%", pad=2.6, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1),orientation="horizontal").set_label(label=r'$[Z]$',size=15)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_vs_k_mmax{m_star_max}.pdf', bbox_inches='tight')
        #plt.show(block=False)


    def alpha1_Z_plot(self, Z_massfrac_v, alpha1_Z_list):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        #ax.plot(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        #ax.scatter(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        ax.plot(Z_massfrac_v- 0.0142, [alpha1_Z_list[i][0] for i in range(len(Z_massfrac_v))], linewidth=3, color='magenta', alpha=0.4)
        ax.scatter(Z_massfrac_v- 0.0142, [alpha1_Z_list[i][0] for i in range(len(Z_massfrac_v))], linewidth=3, color='magenta', alpha=0.4)
        ax.set_ylabel(r'$\alpha_1$', fontsize=15)
        ax.set_xlabel(r'[Z - $Z_{\odot}$]', fontsize=15)
        ax.axhline(1.3 - 63*0.0142, linestyle=':', color='orange')
        ax.axhline(1.3 + 63*0.0142, linestyle=':', color='orange')
        ax.axhline(1.3, linestyle='--', color='orange')
        ax.axvline(0, linestyle='--', color='orange')
        ax.plot(Z_massfrac_v - 0.0142, 1.3 + np.arctan(1.3e2*(Z_massfrac_v - 0.0142))/1.3, color='red', linewidth=3)
        #plt.title(r'SFR = %.2e [%s/yr]' %(self.SFR, Msun), fontsize=15)
        #ax.set_ylim(5e-2,1e8)
        #ax.set_xlim(1e-11,1e0)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/alpha1_Z_plot_{name}.pdf', bbox_inches='tight')
        #plt.show(block=False)
