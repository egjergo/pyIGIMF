import numpy as np
from igimf import utilities as util
from igimf import instance as inst

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
               
    def MeclMax_bySFR_plot(self, SFR_v, MeclMax_list, k_ECMF_list, beta_ECMF_list):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        
        fig, ax = plt.subplots(1,1, figsize=(7,4))
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        
        ax.loglog(SFR_v, MeclMax_list, linewidth=3, color='#0000ff')
        #ax.scatter(SFR_v, MeclMax_list, linewidth=3, color='#0000ff',s=1)
        ax.set_ylabel(r'$M_{\rm ecl,max}$ [%s]'%(Msun), fontsize=15, color='#0000ff')
        
        ax2.loglog(SFR_v, k_ECMF_list, linewidth=3, color='#ff0000', linestyle='--')
        #ax2.scatter(SFR_v, k_ECMF_list, linewidth=3, color='#ff0000',s=1, marker='s')
        ax2.set_ylabel(r'$k_{\rm ecl}$', fontsize=15, color='#ff0000')
        
        ax3.semilogx(SFR_v, beta_ECMF_list, linewidth=3, color='black', linestyle=':')
        #ax3.scatter(SFR_v, k_ECMF_list, linewidth=3, color='black',s=1, marker='s')
        ax3.set_ylabel(r'$\beta_{\rm ecl}$', fontsize=15, color='black')
        
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        #ax.set_title(r'[Z] = %.2f' %(Z), fontsize=15)
        #ax.set_ylim(5e-2,1e8)
        #ax.set_xlim(1e-4,1e3)
        ax3.tick_params(labelsize=12)
        ax2.tick_params(labelsize=15)
        ax.tick_params(labelsize=15)
        ax.tick_params(labelsize=15)
        ax.tick_params(width=2)
        ax2.tick_params(width=2)
        ax3.tick_params(width=2)
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
        num_colors = len(beta_ECMF_list) + 1
        Z = [[0,0],[0,0]]
        beta_ECMF_list = np.flip(beta_ECMF_list)
        levels = np.linspace(np.min(beta_ECMF_list), np.max(beta_ECMF_list),
                             num_colors, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        
        fig, ax = plt.subplots(1,1, figsize=(6,4.2))
        for b in beta_ECMF_list:
            y = Mecl_v**(-b)
            color = next(currentColor)
            ax.plot(np.log10(Mecl_v), np.log10(y), linewidth=3, c=color)
            #ax.scatter(Mecl_v, y, linewidth=3, c=color,s=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\propto \log_{10}(\xi_{ECMF})$'+f' [#/{Msun}]',
                      fontsize=15)
        ax.set_xlabel(r'$\log_{10}(M_{\rm ecl})$'+f' [{Msun}]', fontsize=15)
        ax.set_title(r'normalized to $k_{\rm ecl}=1$', fontsize=15)
        #ax.set_ylim(1e-15,1e1)
        ax.set_ylim(-16,0)
        ax.set_xlim(np.log10(3),10.2)
        plt.yticks(fontsize=11)
        plt.xticks(fontsize=11)
        ax.tick_params(labelsize=12)
        #ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(.1))
        cbar.set_label(label=r'$\beta_{\rm ecl}$',size=15)
        cbar.ax.invert_yaxis()
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_power_beta.pdf', bbox_inches='tight')
        #plt.show(block=False)
        
    def ECMF_plots(self, M_ecl_v_list, ECMF_v_list, SFR_v):
        from matplotlib import pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm_default = plt.cm.get_cmap(name='inferno')
        cm = mcolors.LinearSegmentedColormap.from_list(
                "truncated_inferno", cm_default(np.linspace(0, 0.9, 256))  # 0.9 removes the lightest colors
            )
        num_colors = len(ECMF_v_list) + 1 # center the colorbar
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, #109 
                             endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        SFR_colormap = (SFR_v)#np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        for i,ECMF in enumerate(ECMF_v_list):
            logECMF = np.log10(ECMF)
            logMecl = np.log10(M_ecl_v_list[i])
            current_color = next(currentColor)
            ax.plot(logMecl, logECMF, linewidth=2.5, c=current_color)
            ax.vlines(logMecl[-1], ymin=-11, ymax=logECMF[-1], colors=current_color, linestyle="-", linewidth=3.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\log_{10}(\xi_{ECMF})$'+ f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$\log_{10}(M_{\rm ecl})$ [%s]' %(Msun), fontsize=15)
        ax.set_ylim(-10,6)
        ax.set_xlim(np.log10(3),10.2)
        #ax.tick_params(width=2)
        ax.tick_params(labelsize=12)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}({\rm SFR})$'+f' [{Msun}/yr]',size=15)
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
        metallicity = np.log10(metal_mass_fraction/self.solar_metallicity)
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
        
    # def IMF_plots(self, mstar_v, IMF_v_list, Mecl_v, k_idx, massfrac):
    #     from matplotlib import pyplot as plt
    #     import colorcet as cc
    #     import matplotlib.ticker as ticker
    #     from mpl_toolkits.axes_grid1 import make_axes_locatable
    #     Msun = r'$M_{\odot}$'
    #     cm = cc.cm.CET_L20
    #     eff_Mecl_v = Mecl_v[k_idx]
    #     eff_IMF_v_list = np.array(IMF_v_list)[k_idx]
    #     num_colors = len(eff_Mecl_v)
    #     Z = [[0,0],[0,0]]
    #     #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
    #     levels = np.linspace(np.log10(eff_Mecl_v[0]), np.log10(eff_Mecl_v[-1]), num_colors, endpoint=True)
    #     CS3 = plt.contourf(Z, levels, cmap=cm)
    #     plt.clf()
    #     fig, ax = plt.subplots(1,1, figsize=(7,5))
    #     currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
    #     currentColor = iter(currentColors)
    #     for i, IMF in enumerate(eff_IMF_v_list):
    #         ax.loglog(mstar_v, IMF, linewidth=3, c=next(currentColor))
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad="2%")
    #     ax.set_ylabel(r'$\xi_{\star}={\rm d} N_{\star}/{\rm d} m$'
    #                   +f' [#/{Msun}]', fontsize=15)
    #     ax.set_xlabel(r'$M_{\rm star}$ [%s]' %(Msun), fontsize=15)
    #     ax.set_ylim(1e-2,1e10)
    #     Z = np.log10(massfrac/0.0134)
    #     ax.set_title(r"[Z] = %.2f"%(Z), fontsize=15)
    #     ax.tick_params(width=2)
    #     cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}(M_{\rm ecl})$'+f' [({Msun})]',size=15)
    #     fig.tight_layout()
    #     plt.savefig(f'figs/IMF_plots_Z{Z:.2f}.pdf', bbox_inches='tight')
    #     #plt.show(block=False)
    #     return None
    
    def IMF_plots(self, mstar_v, IMF_v_list, Mecl_v, k_idx, massfrac):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = plt.cm.get_cmap(name='viridis')
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
        ax.set_ylabel(r'$\xi_{IMF}$', fontsize=15)
        ax.set_xlabel(r'$M_{\rm star}$ [%s]' %(Msun), fontsize=15)
        ax.set_ylim(1e-2,1e10)
        ax.set_title(r"metal mass fraction = %.2e"%(massfrac), fontsize=15)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}(M_{\rm ecl})$',size=15)
        fig.tight_layout()
        plt.savefig(f'IMF_plots_massfrac{massfrac:.2e}.pdf', bbox_inches='tight')
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
        Msun = r'$M_{\odot}$'
        
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
    
    # def IGIMF_3D_plot(self, df, SFR_v, metal_mass_fraction_v, mstar_v, 
    #                   by_v='SFR', col_ax_idx=10, azim_rot=-120, elev_rot=20):
    #     '''
    #     by_v can be "SFR" or "metal_mass_fraction"
    #     '''
    #     from mpl_toolkits import mplot3d
    #     import matplotlib.pyplot as plt
        
    #     Msun = r'$M_{\odot}$'
    #     if by_v == 'SFR':
    #         y_ax = SFR_v
    #         color_ax = metal_mass_fraction_v
    #         title = '[Z]'
    #         metallicity_val = np.log10(color_ax[col_ax_idx]/0.0134)
    #         units = f'[{Msun}/yr]'
    #     elif by_v == 'metal_mass_fraction':
    #         y_ax = metal_mass_fraction_v
    #         color_ax = SFR_v
    #         title = 'SFR'
    #         units = ''
    #     else:
    #         raise ValueError("set by_v either to 'SFR' or 'metal_mass_fraction'. ")
        
    #     fig = plt.figure(figsize=(10,8))
    #     ax = plt.axes(projection ='3d')
    #     x = np.outer(mstar_v, np.ones(len(y_ax)))
    #     y = np.outer(y_ax, np.ones(len(mstar_v))).T
    #     xi = np.reshape([df.loc[((df['mass_star']==ival) & (df[by_v]==jval) 
    #         & (df['metal_mass_fraction']==metal_mass_fraction_v[col_ax_idx])
    #         )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
    #         for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
        
    #     ## Setting a mask to exclude zero values
    #     #xi_mask = np.ma.masked_where(np.isnan(xi), xi)
    #     #xi_masked = xi.copy()
    #     #xi_masked[np.isnan(xi)] = -0.
        
    #     #ax.plot_surface(np.log10(x), np.log10(y), np.log10(xi_masked), cmap ='plasma', linewidth=0.25)
    #     ax.plot_surface(np.log10(x[:47,:47]), np.log10(y[:47,:47]), np.ma.log10(xi[:47,:47]), cmap ='plasma', linewidth=0.25)
    #     #ax.plot_surface(np.log10(x), np.log10(y), np.log10(xi_masked), cmap ='plasma', linewidth=0.25)
    #     ax.set_xlabel(r'stellar mass $m_{\star}$ [$\log_{10}(M_{\odot})$]', fontsize=15)
    #     ax.set_ylabel(f'{by_v}  {units}', fontsize=15)
    #     ax.set_zlabel(r'$\xi_{\rm IGIMF}={\rm d}N_{\star}/ {\rm d} m$'+
    #                   f'['+r'$\log_{10}({\rm #}/M_{\odot})$'+f']', fontsize=15)
    #     ax.set_title(f'{title} {metallicity_val:.2f}', fontsize=17)
    #     ax.azim = azim_rot
    #     ax.elev = elev_rot
    #     fig.tight_layout()
    #     plt.show(block=False)
    #     plt.savefig(f'figs/IGIMF_plot_3D.pdf', bbox_inches='tight')
     
     
    def IGIMF_3D_plot(self, df, SFR_v, metal_mass_fraction_v, mstar_v, 
                      by_v='SFR', col_ax_idx=10, azim_rot=-120, elev_rot=20):
        '''
        by_v can be "SFR" or "metal_mass_fraction"
        '''
        from mpl_toolkits import mplot3d
        import colorcet as cc
        import matplotlib.pyplot as plt
        print('Plotting IGIMF_3D_plot')
        Msun = r'$M_{\odot}$'
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection ='3d')
        
        if by_v == 'SFR':
            y_ax = SFR_v
            color_ax = metal_mass_fraction_v
            title = '[Z]'
            tsave = 'Z'
            metallicity_val = np.log10(color_ax[col_ax_idx]/0.0134)
            title_val = metallicity_val
            ax.set_title(f'{title} = {metallicity_val:.2f}', fontsize=20, y=0.95)
            units = f'[{Msun}/yr]'
            xi = np.reshape([df.loc[((df['mass_star']==ival) & (df[by_v]==jval) 
            & (df['metal_mass_fraction']==metal_mass_fraction_v[col_ax_idx])
            )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
            for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
            by_v_axis=by_v
        elif by_v == 'metal_mass_fraction':
            by_v_axis = '[Z]'
            y_ax = metal_mass_fraction_v/0.0134
            color_ax = SFR_v
            title = r'$\log_{10}(\rm SFR)$'
            SFR_val = np.log10(color_ax[col_ax_idx])
            title_val = SFR_val
            tsave = 'SFR'
            ax.set_title(f'{title} = {SFR_val:.2f}'+f' [{Msun}/yr]', fontsize=20, y=0.95)
            units = ''
            xi = np.reshape([df.loc[((df['mass_star']==ival) & (df[by_v]==jval*0.0134) 
            & (df['SFR']==color_ax[col_ax_idx])
            )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
            for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
        else:
            raise ValueError("set by_v either to 'SFR' or 'metal_mass_fraction'. ")
        
        x = np.outer(mstar_v, np.ones(len(y_ax)))
        y = np.outer(y_ax, np.ones(len(mstar_v))).T
        
        ## Setting a mask to exclude zero values
        #xi_mask = np.ma.masked_where(np.isnan(xi), xi)
        #xi_masked = xi.copy()
        #xi_masked[np.isnan(xi)] = -0.
        
        #ax.plot_surface(np.log10(x), np.log10(y), np.log10(xi_masked), cmap ='plasma', linewidth=0.25)
        #ax.plot_surface(np.log10(x[:47,:47]), np.log10(y[:47,:47]), np.ma.log10(xi[:47,:47]), cmap ='plasma', linewidth=0.25)
        surf = ax.plot_surface(np.log10(x[:,:]), np.log10(y[:,:]), np.ma.log10(xi[:,:]), cmap =cc.cm.CET_R4, linewidth=0.25)
        cbar = fig.colorbar(surf, ax=ax, orientation='horizontal', pad=0., shrink=0.6)
        cbar.ax.tick_params(labelsize=15)  # Adjust the label size here
        cbar.set_label(r'$\log_{10}(\xi_{\rm IGIMF})$'+f' [#/{Msun}]', fontsize=20)
        pos = cbar.ax.get_position()
        cbar.ax.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])  # [left, bottom, width, height]
        ax.set_xlabel(r'stellar mass $m_{\star}$ [$\log_{10}(M_{\odot})$]', fontsize=15)
        ax.set_ylabel(f'{by_v_axis}  {units}', fontsize=15)
        ax.set_zlabel(r'$\xi_{\rm IGIMF}={\rm d}N_{\star}/ {\rm d} m$ '+
                      f'['+r'$\log_{10}$'+f'(#/{Msun})]', fontsize=15)
        ax.azim = azim_rot
        ax.elev = elev_rot
        ax.set_zlim(np.log10(5e-3),12)
        fig.tight_layout(rect=[0.,0., 1., 1.])
        plt.show(block=False)
        #plt.write_html(f"figs/IGIMF_plot_3D_{by_v}_{tsave}{title_val:.2f}.html")
        plt.savefig(f'figs/IGIMF_plot_3D_{by_v}_{tsave}{title_val:.2f}.pdf', bbox_inches='tight')
     
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
        ax = fig.add_subplot(projection ='3d')       
        for m in metal_mass_fraction_v: 
            for s in SFR_v:
                grid_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==m)]
                ax.loglog(grid_sel['mass_star'], grid_sel['SFR'], grid_sel['IGIMF'], color=next(currentColor))
        fig.tight_layout()
        plt.show(block=False)
        plt.savefig(f'figs/IGIMF_plot_3Dlines.pdf', bbox_inches='tight')
        return None
       
    def sIMF_subplot_old(self, metallicity_v, Mecl_v, mstar_v, sIMF, res=20):
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



    def mw_sIMF_subplot_old(self, metallicity_v, Mecl_v, mstar_v, mw_sIMF, res=20):
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

    def sIMF_subplot_Mecl_old(self, metallicity_v, Mecl_v, mstar_v, sIMF, res=20):
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
        currentColors = reversed(list([cm(1.*i/num_colors) for i in range(num_colors)]))
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 2,2 #3,3 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in reversed(list(enumerate(metallicity_v))):
                ax.annotate(r'$M_{\rm ecl}=$%.2e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                ax.loglog(mstar_v, sIMF[i][j], color=next(currentColor),
                          alpha=1)
                for shift in np.arange(-5,20):
                    ax.loglog(mstar_v, util.Kroupa01()(mstar_v)*np.power(10.,shift), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                ax.set_ylim(5e-3,1e11)
                ax.set_xlim(6e-2,1.6e2)
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\xi_{\star}={\rm d} N_{\star}/{\rm d} m$'+
                                  f' [#/{Msun}]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
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
        currentColors = reversed(list([cm(1.*i/num_colors) for i in range(num_colors)]))
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in reversed(list(enumerate(metallicity_v))):
                ax.annotate(r'$M_{\rm ecl}=$%.2e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                ax.loglog(mstar_v, np.divide(mw_sIMF[i][j], Mecl_v[i]), color=next(currentColor),
                          alpha=0.8)
                for shift in np.arange(-5,20):
                    ax.loglog(mstar_v, np.divide(mstar_v*util.Kroupa01()(mstar_v)*np.power(10.,shift),Mecl_v[i]), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                #ax.set_xlim(2e-2,5e2)
                #ax.set_ylim(5e-3,1e11)
                ax.set_xlim(6e-2,1.6e2)
                ax.set_ylim(1e-8,1e3)
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$m \xi_{\star}(m) / M_{\rm ecl} \propto \frac{{\rm d} N / {\rm d} \log_{10}m}{M_{\rm ecl}} \quad$ [#/$M_{\odot}$]', fontsize = 14)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.2, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
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
        cm = cc.cm.CET_I1
        levels = np.linspace(np.log10(Mecl_v[0]), np.log10(Mecl_v[-1]), 100,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(Mecl_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        #fig, axs = plt.subplots(3, 3, figsize=(8,6))
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, M in enumerate(Mecl_v):
                ax.annotate(r'$[Z]=$%.2f'%(metallicity_v[i]), xy=(0.5, 0.9), 
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=.1)
                ax.loglog(mstar_v, sIMF[j][i], color=next(currentColor))
                for shift in np.arange(-5,20):
                    ax.loglog(mstar_v, util.Kroupa01()(mstar_v)*np.power(10.,shift), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                ax.set_ylim(5e-3,1e11)
                ax.set_xlim(6e-2,1.6e2)
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
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
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

    def alpha3_plot(self, alpha3_v, x_v, parameter_space, resolution=15):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        import colorcet as cc
        fig,ax = plt.subplots(figsize=(6,4))
        x,y = np.meshgrid(np.log10(parameter_space['rho_cl']), parameter_space['[Z]'])
        z = np.array(alpha3_v).reshape(resolution,resolution).T
        cax = ax.contourf(x, y, z, resolution, cmap=cc.cm.CET_L6)
        plt.xlabel(r'$\log_{10}(\rho_{cl})$ [$\log_{10}(M_{\odot}/{\rm pc}^3)$]', fontsize=15)
        plt.ylabel(r'[$Z$]', fontsize=15)
        cbar = fig.colorbar(cax)
        cbar.set_label(r'$\alpha_3$', fontsize=15)
        plt.tight_layout()
        plt.savefig('figs/alpha3plot.pdf')

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
        
        ax.errorbar(Cook23binHalpha['sfrsig-bin'], Cook23binHalpha['Gamma'], xerr=Cook23binHalpha['sfrsig-u'], yerr=Cook23binHalpha['Gamma-u'], color='red', ecolor='red', elinewidth=3, capsize=0, label=r'C23 H$_{\alpha}$ $<10$ Myr', marker='o', markersize=9, alpha=0.8)
        ax.errorbar(Cook23bin10['sfrsig-bin'], Cook23bin10['Gamma'], xerr=Cook23bin10['sfrsig-u'], yerr=Cook23bin10['Gamma-u'], color='black', ecolor='black', elinewidth=3, capsize=0,label=r'C23 res $<10$ Myr', marker='s', markersize=9, alpha=0.8)
        
        #ax.errorbar(Cook23['logSFRsig'], Cook23['Gamma'], yerr=Cook23['Gamma-u'], fmt='o', color='blue', ecolor='blue', elinewidth=1, capsize=0,label=r'C23 lit', marker='o', alpha=0.4)
        ax.set_xlim(-3.7, 0.5)
        ax.set_ylim(2e-2, 2e2)
        ax.legend(loc='lower right', fontsize=12, frameon=True)
        ax.set_ylabel(r'$\Gamma_e$ (%)', fontsize=15)
        ax.set_xlabel(r'$\log(\Sigma_{\rm SFR})$ ($M_{\star} yr^{-1} kpc^{-2}$)', fontsize=15)
        fig.tight_layout()
        plt.savefig('figs/Cook23.pdf', bbox_inches='tight')
        #plt.show(block=False)

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

    def create_centered_gradient_image(self, ax, extent, cmap, alpha=0.4):
        """
        Create a centered gradient image and set it as background in the given axis.
        """
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        gradient = np.vstack((gradient, gradient))  # Repeat to ensure it covers the background

        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent, alpha=alpha, zorder=-1000)
        return ax
   
    def Encyclopedia_main_plot(self, Mecl_v, res=13):
        print('Plotting Encyclopedia_main_plot')
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import scipy.integrate as integr
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as path_effects
        import matplotlib.patches as patches

        mstar_v = np.logspace(np.log10(0.1), np.log10(150),res*10)
        mBD_v = np.logspace(np.log10(0.01), np.log10(0.1),res*10, endpoint=True)
        if not np.any(np.isclose(mstar_v, 1)):
            mstar_v = np.sort(np.append(mstar_v, 1))
        idx_unitary = np.where(mstar_v ==1)[0][0]
        metallicity_v = np.array([-6,-2,0.,0.4])
        #SFR_v = np.array([1e-4,1e-2,1e0,1e2])
        
        Msun = r'$M_{\odot}$' 
        #cm = cc.cm.CET_CBD2
        cm = cc.cm.CET_CBL3
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], res,
                             endpoint=True)
        CS2 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = list([cm(1.*i/num_colors) for i in range(num_colors)])
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 1,1 #util.find_closest_prod(res)
        fig, ax2 = plt.subplots(nrow, ncol, figsize=(8,4))
        ax = ax2.twiny()
        
        n_bins = 100  # Discretizes the interpolation into bins

        # Define the custom colormap
        colors_b = [(.302, .627, .596), (0., .451, 1.00), (.502, .502, .502)]  # White -> Blue -> White
        cmap_name_b = 'white_blue_white'
        cmap_b = LinearSegmentedColormap.from_list(cmap_name_b, colors_b, N=n_bins)
        #log_cmap_b = self.create_log_colormap(cmap_b, np.log10(0.1), np.log10(5))
        extent_b = [np.log10(0.2), np.log10(1), 1e-5, 1e12]
        cax_b = self.create_centered_gradient_image(ax2, extent=extent_b, cmap=cmap_b)
        
        colors_g = [(1, 1, 1), (.6, .804, .196), (.302, .627, .596)]  # White -> Blue -> White
        cmap_name_g = 'white_green_white'
        cmap_g = LinearSegmentedColormap.from_list(cmap_name_g, colors_g, N=n_bins)  
        extent_g = [np.log10(0.01), np.log10(0.2001), 1e-5, 1e12]
        cax_g = self.create_centered_gradient_image(ax2, extent=extent_g, cmap=cmap_g)
        
        colors_o =  [(.502, .502, .502), (1.00, .549, 0.), (1, 1, 1)]  # White -> GRB percent -> White
        cmap_name_o = 'white_orange_white'
        cmap_o = LinearSegmentedColormap.from_list(cmap_name_o, colors_o, N=n_bins)
        extent_o = [np.log10(1), np.log10(150), 1e-5, 1e12]
        cax_o = self.create_centered_gradient_image(ax2, extent=extent_o, cmap=cmap_o)
        
        rect = patches.Rectangle((np.log10(20), 1e2), 6, 1e12, linewidth=1, edgecolor='dimgray', facecolor='none', hatch='xxx', alpha=0.3)
        ax2.add_patch(rect)
        
        #for i, ax in enumerate(axs.flat):
        linewidth_choice=0
        a_shift = 0
        a_angle = 0
        linestyle = ['-', '--', '-.', ':']
        for M_ecl in Mecl_v:
            linewidth_choice += 2
            for j, Z in list(enumerate(metallicity_v)):
                metal_mass_fraction = np.power(10, Z) * 0.0134
                o_IMF = inst.StellarIMF(M_ecl, metal_mass_fraction, 1)
                #IMF_integr = integr.quad(o_IMF.IMF_func, min_mass, max_mass)[0]
                IMF_v = o_IMF.IMF_func(mstar_v)
                BD_v = o_IMF.BD_func(mBD_v)
                ax.annotate(r'$\log_{10}(M_{\rm ecl})=$%.2f'%(np.log10(M_ecl)), xy=(5, 2e3+a_shift),
                         verticalalignment='top', color='dimgrey',
                        horizontalalignment='center', rotation=-11+a_angle, fontsize=11, alpha=1)
                thiscol = next(currentColor)
                ax.loglog(mstar_v, np.log(10)*np.multiply(IMF_v, mstar_v), color=thiscol,
                            alpha=0.99, linewidth=linewidth_choice, label=f'[Z] = {Z}')
                ax.loglog(mBD_v, np.log(10)*np.multiply(BD_v, mBD_v), color=thiscol, 
                          linewidth=linewidth_choice, alpha=0.99, linestyle = '--', zorder=0)
            a_shift += 7e5
            a_angle += 4
            if M_ecl == Mecl_v[0]:
                ax.legend(loc='lower left', fontsize=9)
                handles, labels = ax.get_legend_handles_labels()
                handles = handles[::-1]
                labels = labels[::-1]
                legend = ax.legend(handles, labels, loc='lower left', fontsize=9)
                for text in legend.get_texts():
                    text.set_color("dimgrey")  
        
        ax.axvline(1, color='dimgrey' , linewidth=1, linestyle='-.', alpha=1 , zorder=10) 
        ax.axvline(0.5, color='dimgrey', linewidth=1, linestyle='-.', alpha=1 , zorder=10) 
        #ax.axvline(0.1, color='dimgrey', linewidth=1, linestyle='-.', alpha=1 , zorder=-1) 
        
        canonical_IMF = util.Kroupa01()(mstar_v)
        canon_whole = integr.quad(util.Kroupa01(), mstar_v[0], mstar_v[-1])[0] 
        BD_func = lambda M: M**(-0.3)
        BD_whole = integr.quad(BD_func, mBD_v[0], mBD_v[-1])[0]
        BD_norm = canon_whole / (4.5 * BD_whole)
        for shift in np.arange(-5,20,.5):
            ax.loglog(mstar_v, np.log(10)*np.multiply(canonical_IMF,mstar_v)*np.power(10.,shift), color='grey', linewidth=1, linestyle='-.', alpha=0.3 , zorder=-1)
            ax.loglog(mBD_v, np.log(10)*np.multiply(BD_func(mBD_v),mBD_v)*BD_norm*np.power(10.,shift), color='grey', linewidth=1, linestyle='--', alpha=0.3 , zorder=-1)
        ax.set_xlim(7e-3,1.45e2)
        ax.set_ylim(1e-2,1e10)   
        #ax2.semilogy(np.log10(mstar_v), np.multiply(canonical_IMF,mstar_v)*np.power(10.,shift), color='white', linewidth=1, linestyle='-.', alpha=0. , zorder=-1)     
        ax2.set_xlim(np.log10(7e-3),np.log10(1.45e2))
        ax2.set_ylim(1e-2,1e10)
        
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False,
               labelbottom=False, labeltop=False, labelleft=True, labelright=False)
        #ax2.xaxis.set_label_position('top')
        #ax2.xaxis.tick_top()
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        
        text_g = ax.annotate('    peripheral\n fragmentation', xy=(1.5e-2, 1e8),
                   fontsize=9, color='white', zorder=100)      
        text_g.set_path_effects([path_effects.Stroke(linewidth=3, foreground='#8dbd2e'),
                       path_effects.Normal()]) 
        text_b = ax.annotate('      primary\n fragmentation', xy=(2e-1, 1e8),
                   fontsize=9, color='white', zorder=100)      
        text_b.set_path_effects([path_effects.Stroke(linewidth=3, foreground=colors_b[1]),
                       path_effects.Normal()])    
        text_o = ax.annotate('  dynamic\n accretion', xy=(4, 1e8),
                   fontsize=9, color='white', zorder=100)      
        text_o.set_path_effects([path_effects.Stroke(linewidth=3, foreground=colors_o[1]), #'#7aa428'
                       path_effects.Normal()])  

        text_f = ax.annotate('   feedback\n and mergers\n in embedded\n    clusters', xy=(30, 1e5),
                   fontsize=8, color='white', zorder=100)      
        text_f.set_path_effects([path_effects.Stroke(linewidth=3, foreground='dimgray'),
                       path_effects.Normal()])   
             
        ax.annotate('', xy=(0.1, 1), xytext=(0.5, 1), arrowprops=dict(arrowstyle='<->', color='dimgray'))
        ax.text(0.25, 0.4, r'$\alpha_1$', horizontalalignment='center', color='dimgray')
        ax.annotate('', xy=(0.5, 1), xytext=(1, 1), arrowprops=dict(arrowstyle='<->', color='dimgray'))
        ax.text(0.75, 0.4, r'$\alpha_2$', horizontalalignment='center', color='dimgray')
        ax.annotate('', xy=(1, 1), xytext=(150, 1), arrowprops=dict(arrowstyle='<->', color='dimgray'))
        ax.text(10, 0.4, r'$\alpha_3$', horizontalalignment='center', color='dimgray')

        ax2.set_ylabel(r'$\xi_{\rm L}(m) = \ln(10) m \xi_{\star}(m)$'+
                      r' $= {\rm d} N / {\rm d} \log_{10}m \,$ '+'[#]', 
                      fontsize = 12)
        ax.set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize = 12)
        ax.yaxis.set_label_coords(-0.1, 0.45)
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        fig.tight_layout(rect=[0, 0., 1, 0.9])
        pos = ax.get_position()
        x0, y0, width, height = pos.bounds
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cax = plt.axes([x0, y0+height, width, 0.1])
        cax.set_xlim(np.log10(7e-3),np.log10(1.45e2))
        cax.set_ylim(1e-2,1e10)
        
        star_colors = {
            'i': (1., 1., 1.),
            'O': (0.157, 0.176, 1.000),      # Blue
            'B': (0.500, 0.647, 1.000),      # Blue-white
            'A': (0.700, 0.824, 1.000),      # White
            'F': (0.914, 0.933, 1.000),      # Yellow-white
            'G': (1.000, 1.000, 0.839),      # Yellow
            'K': (1.000, 0.764, 0.490),      # Orange
            'M': (1.000, 0.500, 0.500),      # Red
            'BD': (0.647, 0.165, 0.165),     # Brown
            'e': (1., 1., 1.)
        }

        star_color_range = {
            'O': [(0.328125  , 0.41015625, 0.99609375), (0.157, 0.176, 1.000), (1., 1. , 1.)],      # Blue
            'B': [(0.6015625 , 0.734375  , 0.99609375), (0.500, 0.647, 1.000), (0.328125  , 0.41015625, 0.99609375)],      # Blue-white
            'A': [(0.80784314, 0.88235294, 1.        ), (0.700, 0.824, 1.000), (0.6015625 , 0.734375  , 0.99609375)],      # White
            'F': [(0.95686275, 0.96862745, 0.91764706), (0.914, 0.933, 1.000), (0.80784314, 0.88235294, 1.        )],      # Yellow-white
            'G': [(1.        , 0.88235294, 0.66666667), (1.000, 1.000, 0.839), (0.95686275, 0.96862745, 0.91764706)],      # Yellow
            'K': [(1.        , 0.63137255, 0.49411765), (1.000, 0.764, 0.490), (1.        , 0.88235294, 0.66666667)],      # Orange
            'M': [(0.82352941, 0.32941176, 0.32941176), (1.000, 0.500, 0.500), (1.        , 0.63137255, 0.49411765)],      # Red
            'BD': [(1., 1. , 1.), (0.647, 0.165, 0.165), (0.82352941, 0.32941176, 0.32941176)]     # Brown
        }

        # Example temperatures for spectral types (in Kelvin)
        temperatures = {
            'O': 40000,
            'B': 20000,
            'A': 9000,
            'F': 7000,
            'G': 5500,
            'K': 4000,
            'M': 3000,
            'BD': 1300
        }

        # Example mass ranges for spectral types (in solar masses)
        masses = {
            'O': 50,    # O-type stars typically range from 16 to 100 M☉
            'B': 10,    # B-type stars typically range from 2.1 to 16 M☉
            'A': 2.5,   # A-type stars typically range from 1.4 to 2.1 M☉
            'F': 1.5,   # F-type stars typically range from 1.04 to 1.4 M☉
            'G': 1.0,   # G-type stars typically range from 0.8 to 1.04 M☉
            'K': 0.6,   # K-type stars typically range from 0.45 to 0.8 M☉
            'M': 0.2,    # M-type stars typically range from 0.08 to 0.45 M☉
            'BD': 0.045
        }
        
        logmasses = {
            'O': ((np.log10(16)+np.log10(150))/2),    # O-type stars typically range from 16 to 100 M☉
            'B': ((np.log10(2.1)+np.log10(16))/2),    # B-type stars typically range from 2.1 to 16 M☉
            'A': ((np.log10(1.4)+np.log10(2.1))/2),   # A-type stars typically range from 1.4 to 2.1 M☉
            'F': ((np.log10(1.04)+np.log10(1.4))/2),   # F-type stars typically range from 1.04 to 1.4 M☉
            'G': ((np.log10(0.8)+np.log10(1.04))/2),   # G-type stars typically range from 0.8 to 1.04 M☉
            'K': ((np.log10(0.45)+np.log10(0.8))/2),   # K-type stars typically range from 0.45 to 0.8 M☉
            'M': ((np.log10(0.08)+np.log10(0.45))/2),    # M-type stars typically range from 0.08 to 0.45 M☉
            'BD': ((np.log10(0.012)+np.log10(0.08))/2)
        }
        
        logmassrange = {
            'O': np.log10([15,150]),    # O-type stars typically range from 16 to 100 M☉
            'B': np.log10([2.1,16]),    # B-type stars typically range from 2.1 to 16 M☉
            'A': np.log10([1.4,2.1]),   # A-type stars typically range from 1.4 to 2.1 M☉
            'F': np.log10([1.04,1.4]),   # F-type stars typically range from 1.04 to 1.4 M☉
            'G': np.log10([0.8,1.04]),   # G-type stars typically range from 0.8 to 1.04 M☉
            'K': np.log10([0.45,0.8]),   # K-type stars typically range from 0.45 to 0.8 M☉
            'M': np.log10([0.08,0.45]),   # M-type stars typically range from 0.08 to 0.45 M☉
            'BD': np.log10([0.012,0.08])
        }
        

        start = 'i'
        end = ['O','B','A','F','G','K','M','BD','e']
        for spectral_type, mass in masses.items():
            _text = cax.annotate(spectral_type, xy=(logmasses[spectral_type]-0.03, 3e9), fontsize=10, color='black', zorder=100)
            _text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=star_colors[spectral_type]), path_effects.Normal()]) 
            end.pop(0)
            #colors_ = [star_colors[spectral_type], star_colors[spectral_type], star_colors[spectral_type]] #[star_colors[end[0]], star_colors[spectral_type], star_colors[start]] 
            colors_ = star_color_range[spectral_type] #[star_colors[end[0]], star_colors[spectral_type], star_colors[start]] 
            start = spectral_type
            cmap_name_ = spectral_type
            cmap_ = LinearSegmentedColormap.from_list(cmap_name_, colors_, N=n_bins)
            #log_cmap_b = self.create_log_colormap(cmap_b, np.log10(0.1), np.log10(5))
            extent_ = [logmassrange[spectral_type][0], logmassrange[spectral_type][1], 1e-5, 1e12]
            cax_ = self.create_centered_gradient_image(cax, extent=extent_, cmap=cmap_, alpha=1.)
        
        cax.set_xticks([])
        cax.set_yticks([])
        cax.set_xticklabels([])
        cax.set_yticklabels([])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/chIMF_fig_Encyclopedia_main_plot.pdf')
        

    def IGIMF_panels(self,df, res=15):
        print('Plotting IGIMF_panels')
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        import igimf as ii
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        #cm = cc.cm.CET_R2
        cm = cc.cm.CET_D4
        
        nrow, ncol = 2,2 #3,3 #util.find_closest_prod(res)
        SFR_v = np.unique(df['SFR'])
        SFR_select = SFR_v[[1,4,-6,-2]]
        #metallicity_v = np.unique(df['[Z]'])
        #metal_mass_fraction = np.power(10, metallicity_v) * 0.0134
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
        metallicity_v = np.log10(metal_mass_fraction_v/0.0134)
        Z_select = metallicity_v[[7,11,-2,-1]]
        
        plt.clf()
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], res,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = reversed(list([cm(1.*i/num_colors) for i in range(num_colors)]))
        currentColor = itertools.cycle(currentColors)
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(8,6))
        for i, ax in enumerate(axs.flat):
            for j, Z in enumerate(reversed(metal_mass_fraction_v)):
                ax.annotate(r'$\log_{10}(SFR/[M_{\odot}/yr])=$%.2f'%(np.log10(SFR_select[i]))#+"["+Msun+'/yr]'
                            , xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=12, alpha=1)
                #IGIMF_dict = ii.use.compute_IGIMF(SFR=SFR_v[i],metal_mass_fraction=metal_mass_fraction[j])
                df_sel = df.loc[(df['SFR']==SFR_select[i]) & (df['metal_mass_fraction']==Z)]
                ax.loglog(df_sel['mass_star'], df_sel['IGIMF'], color=next(currentColor),
                          alpha=1)
                for shift in np.arange(-5,20):
                    ax.loglog(df_sel['mass_star'], util.Kroupa01()(df_sel['mass_star'].values)*np.power(10.,shift), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                ax.set_ylim(5e-3,1e12)
                ax.set_xlim(8e-2,1.49e2)
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\xi_{\rm IGIMF}={\rm d} N_{\star}/{\rm d} m$'+
                                  f' [#/{Msun}]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(
                                label=r'[Z]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/IGIMF_plots.pdf')
        plt.show(block=False)


    def IGIMF_massweighted_panels(self,df, res=15):
        print('Plotting IGIMF_massweighted_panels')
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        import igimf as ii
        import pandas as pd
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_D4
        
        nrow, ncol = 2,2 #3,3 #util.find_closest_prod(res)
        SFR_v = np.unique(df['SFR'])
        SFR_select = SFR_v[[1,4,-6,-2]]
        #metallicity_v = np.unique(df['[Z]'])
        #metal_mass_fraction = np.power(10, metallicity_v) * 0.0134
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
        metallicity_v = np.log10(metal_mass_fraction_v/0.0134)
        Z_select = metallicity_v[[7,11,-2,-1]]
        
        plt.clf()
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], res,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = reversed(list([cm(1.*i/num_colors) for i in range(num_colors)]))
        currentColor = itertools.cycle(currentColors)
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(8,6))
        for i, ax in enumerate(axs.flat):
            for j, Z in enumerate(reversed(metal_mass_fraction_v)):
                ax.annotate(r'$\log_{10}(SFR/[M_{\odot}/yr])=$%.2f'%(np.log10(SFR_select[i]))#+"["+Msun+'/yr]'
                            , xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=12, alpha=1)
                df_sel = df.loc[(df['SFR']==SFR_select[i]) & (df['metal_mass_fraction']==Z)]
                ax.loglog(df_sel['mass_star'], np.multiply(df_sel['mass_star'],df_sel['IGIMF'])/df_sel['IGIMF'].iloc[35], color=next(currentColor),
                          alpha=1)
            canonical_IMF = util.Kroupa01()(df_sel['mass_star'].values)
            ax.loglog(df_sel['mass_star'], np.multiply(df_sel['mass_star'],canonical_IMF)/canonical_IMF[35], color='grey', linewidth=2, linestyle='--', alpha=1, zorder=1, label='Kroupa et al. (2001)')
            if (i == 2) or (i == 3):
                vDC24 = pd.read_csv('data/vanDokkumConroy24.dat',delimiter=',')
                vDC24.columns = vDC24.columns.str.strip()
                ax.loglog(vDC24['logMass'], vDC24['logIMF'], color='darkgreen', linewidth=2, linestyle='-.', alpha=1, zorder=10)
                x_vDC24 = np.array([0.1,0.5,1,15.6])
                vDC24r = pd.read_csv('data/vanDokkumConroy24range.dat',delimiter=',')
                vDC24r.columns = vDC24r.columns.str.strip()
                mid_index = len(vDC24r) // 2
                vDC24r1 = vDC24r.iloc[:mid_index]
                vDC24r2 = vDC24r.iloc[mid_index:]
                y1_interp = np.interp(x_vDC24, vDC24r1['logMass'], vDC24r1['logIMF'])
                y2_interp = np.interp(x_vDC24, vDC24r2['logMass'], vDC24r2['logIMF'])
                ax.fill_between(x_vDC24, y1_interp, y2_interp, color='darkgreen', alpha=0.8, label='vanDokkum & Conroy (2024)', zorder=11)
            ax.axhline(y=1, color='k', linestyle=':', alpha=0.1, linewidth=0.1, zorder=-10)
            ax.axvline(x=1, color='k', linestyle=':', alpha=0.1, linewidth=0.1, zorder=-11)
            ax.set_ylim(2e-4,5e3)
            ax.set_xlim(7.5e-2,1.45e2)
            ax.tick_params(axis='x', direction='in', which='both')
            ax.tick_params(axis='y', direction='in', which='both')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
        for nr in range(nrow):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$m \xi_{\rm IGIMF}={\rm d} N_{\star}/{\rm d}\log_{10}(m) $'
                                  #+f'[#/'+r'$\log_{10}$'+f'{Msun}]'
                                  , fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        handles, labels = axs[1,0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        axs[1,0].legend(unique_labels.values(), unique_labels.keys(), loc='lower left', ncol=1, fontsize=9, frameon=False)
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
                            ticks=ticker.MultipleLocator(1)).set_label(
                                label=r'[Z]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/IGIMF_massweighted_plots.pdf')
        plt.show(block=False)
        
    def zetaI_II_plot(self, df):
        print('Plotting zetaI_II_plot')
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        import colorcet as cc
        import pandas as pd
        import scipy.integrate as integr
        import scipy.interpolate as interp
        df['zetaI'] = np.nan
        df['zetaII'] = np.nan
        SFR_v = np.unique(df['SFR'])[:-1]
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
        mstar_v = np.unique(df['mass_star'])
        df_plot = pd.DataFrame({'SFR':[], 'metal_mass_fraction':[], 'mass_star':[], 'zetaI':[], 'zetaII':[]})
        #canon_whole = integr.quad(util.Kroupa01, mstar_v[0], mstar_v[-1])[0]
        canon_norm = np.float64(util.Kroupa01()(1.))
        canon_low = integr.quad(util.Kroupa01(), mstar_v[0], 1.)[0]
        canon_high = integr.quad(util.Kroupa01(), 1., mstar_v[-1])[0]
        canon_lowratio = canon_low/canon_norm
        canon_highratio = canon_high/canon_norm
        for s in SFR_v:
            for z in metal_mass_fraction_v:
                df_sel = df.loc[(df['SFR']==s) & (df['metal_mass_fraction']==z)]
                #igimf_whole = integr.simpson(df_sel['IGIMF'].iloc[:-2],x=df_sel['mass_star'].iloc[:-2])
                igimf_interp = interp.interp1d(df_sel['mass_star'], df_sel['IGIMF'])
                igimf_norm = np.float64(igimf_interp(1.))
                print(f'{igimf_norm=}')
                igimf_low = integr.quad(igimf_interp, mstar_v[0], 1.)[0]
                igimf_high = integr.quad(igimf_interp, 1., mstar_v[-1])[0]
                #xlow = df_sel['mass_star'].iloc[np.where(df_sel['mass_star']<1)[0]].to_numpy()
                #igimf_low = integr.simpson(df_sel['IGIMF'].iloc[:len(xlow)],x=xlow)
                #xhigh = df_sel['mass_star'].iloc[np.where(df_sel['mass_star']>1)[0]].to_numpy()
                #igimf_high = integr.simpson(df_sel['IGIMF'].iloc[len(xlow):len(xlow)+len(xhigh)],x=xhigh)
                #print(f'{igimf_high+igimf_low=}')
                igimf_lowratio = igimf_low/igimf_norm
                #print(f'{igimf_lowratio=}')
                igimf_highratio = igimf_high/igimf_norm
                #print(f'{igimf_highratio=}')
                df['zetaI'].loc[(df['SFR']==s) & (df['metal_mass_fraction']==z)] = float(igimf_lowratio / canon_lowratio)
                df['zetaII'].loc[(df['SFR']==s) & (df['metal_mass_fraction']==z)] = float(igimf_highratio / canon_highratio)

        Msun = r'$M_{\odot}$'
        fig,ax = plt.subplots(figsize=(6,4))
        df_plot = df[['SFR', 'metal_mass_fraction', 'zetaI','zetaII']].drop_duplicates()
        SFR_grid, metallicity_grid = np.meshgrid(np.log10(SFR_v), np.log10(metal_mass_fraction_v/0.0142))
        points = df_plot[['SFR', 'metal_mass_fraction']].values
        points[:,0] = np.log10(points[:,0])
        points[:,1] = np.log10(points[:,1]/0.0142)
        df_plot['zetaI'].fillna(df_plot['zetaI'].mean(), inplace=True)
        values = df_plot['zetaI'].values
        zetaI_grid = interp.griddata(points, values, (SFR_grid, metallicity_grid), method='cubic')
        cax = ax.contourf(SFR_grid, metallicity_grid, zetaI_grid, 15, cmap=cc.cm.CET_L6)
        plt.xlabel(r'$\log_{10}({\rm SFR})$'+f'[{Msun}/yr]', fontsize=15)
        plt.ylabel(r'[$Z$]', fontsize=15)
        #ax.set_ylim(-.45,0.45)
        cbar = fig.colorbar(cax)
        cbar.set_label(r'$\zeta_{\rm I}$', fontsize=15)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig('figs/zetaIplot.pdf')
        
        fig,ax = plt.subplots(figsize=(6,4))
        df_plot = df[['SFR', 'metal_mass_fraction', 'zetaI','zetaII']].drop_duplicates()
        SFR_grid, metallicity_grid = np.meshgrid(np.log10(SFR_v), np.log10(metal_mass_fraction_v/0.0142))
        points = df_plot[['SFR', 'metal_mass_fraction']].values
        points[:,0] = np.log10(points[:,0])
        points[:,1] = np.log10(points[:,1]/0.0142)
        df_plot['zetaII'].fillna(df_plot['zetaII'].mean(), inplace=True)
        values = df_plot['zetaII'].values
        zetaII_grid = interp.griddata(points, values, (SFR_grid, metallicity_grid), method='cubic')
        cax = ax.contourf(SFR_grid, metallicity_grid, zetaII_grid, 15, cmap=cc.cm.CET_L6)
        plt.xlabel(r'$\log_{10}({\rm SFR})$'+f'[{Msun}/yr]', fontsize=15)
        plt.ylabel(r'[$Z$]', fontsize=15)
        cbar = fig.colorbar(cax)
        cbar.set_label(r'$\zeta_{\rm II}$', fontsize=15)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig('figs/zetaIIplot.pdf')

    def alpha1_Z_plot(self, Z_massfrac_v, alpha1_Z_list, name=''):
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
