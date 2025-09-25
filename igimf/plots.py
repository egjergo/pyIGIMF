import numpy as np
from igimf import utilities as util
from igimf import classes as inst
from matplotlib import pyplot as plt
import plotly.tools as tls
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

class Plots:
    # Plotting functions
    def Migal_plot(self, M_igal_v, SFR, downsizing_time):
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
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.loglog(Mecl_v, ECMF_v, linewidth=3, color='navy')
        ax.scatter(Mecl_v, ECMF_v, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$\xi_{ECMF}$'+f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$ [%s]'%(Msun), fontsize=15)
        plt.title(r'$\,$ SFR = %.2e [%s/yr]' %(SFR, Msun), fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/ECMF_plot_SFR{SFR:.2e}.pdf', bbox_inches='tight')
            
    def beta_ECMF_bySFR_plot(self, SFR_v, beta_ECMF_v):
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.semilogx(SFR_v, beta_ECMF_v, linewidth=3, color='navy')
        ax.scatter(SFR_v, beta_ECMF_v, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$\beta_{ECMF}$', fontsize=15)
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/beta_ECMF_bySFR.pdf', bbox_inches='tight')

               
    def MeclMax_bySFR_plot(self, SFR_v, MeclMax_list, k_ECMF_list, beta_ECMF_list):
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        
        ax.loglog(SFR_v, MeclMax_list, linewidth=3, color='#0000ff')
        ax.text(3e1, 1e7, r'$M_{\rm ecl,max}$', fontsize=13, color='#0000ff', verticalalignment='bottom', rotation=47)
        ax.set_ylabel(r'Most massive ecl, $M_{\rm ecl,max}$ [%s]'%(Msun), fontsize=15, color='#0000ff')
        
        ax2.loglog(SFR_v, k_ECMF_list, linewidth=3, color='#ff0000', linestyle='--')
        ax2.text(4e-1, 2e5, r'$k_{\rm ecl}$', fontsize=13, color='#ff0000', verticalalignment='bottom', rotation=45)
        ax2.set_ylabel(r'ECMF normalization, $k_{\rm ecl}$', fontsize=15, color='#ff0000')
        
        ax3.semilogx(SFR_v, beta_ECMF_list, linewidth=3, color='black', linestyle=':')
        ax3.text(4e1, 1.67, r'$\beta_{\rm ecl}$', fontsize=13, color='black', verticalalignment='bottom', rotation=-35)
        ax3.set_ylabel(r'ECMF slope, $\beta_{\rm ecl} \propto \log_{10}({\rm SFR})$', fontsize=15, color='black')
        
        ax.set_title(f'Constraints on the  Embedded cluster (ecl) \n mass function (ECMF, '+r'$\xi_{\rm ECMF}=k_{\rm ecl}\, M_{\rm ecl}^{-\beta_{\rm ecl}}$'+")", fontsize=15, y=1.02)
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        ax3.tick_params(labelsize=10)
        ax2.tick_params(labelsize=12)
        ax.tick_params(labelsize=15)
        ax.tick_params(labelsize=15)
        ax.tick_params(width=2)
        ax2.tick_params(width=2)
        ax3.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/MeclMax_bySFR_plot.pdf', bbox_inches='tight')

        
    def Mecl_power_beta_plot(self, Mecl_v, beta_ECMF_list):
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
        ax.set_ylabel(r'$ \log_{10}(\xi_{ECMF})$ + const',
                      fontsize=15)
        ax.set_xlabel(r'$\log_{10}(M_{\rm ecl})$'+f' [{Msun}]', fontsize=15)
        ax.set_title(r'Embedded Cluster Mass Function ($\xi_{\rm ECMF}=k_{\rm ecl}\,M_{\rm ecl}^{-\beta_{\rm ecl}}$), $k_{\rm ecl}=1$', fontsize=12, pad=10)
        ax.set_ylim(1e-15,1e1)
        ax.set_ylim(-16,0)
        ax.set_xlim(np.log10(3),10.2)
        plt.yticks(fontsize=11)
        plt.xticks(fontsize=11)
        ax.tick_params(labelsize=12)
        #ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.1f", 
                            ticks=ticker.MultipleLocator(.1))
        cbar.set_label(label=r'ECMF power-law slope ($\beta_{\rm ecl}$)',size=15)
        cbar.ax.invert_yaxis()
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_power_beta.pdf', bbox_inches='tight')
        print("Mecl_power_beta_plot done")
        #plt.show(block=False)
        
    def ECMF_plots(self, M_ecl_v_list, ECMF_v_list, SFR_v):
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
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", 
                            ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}({\rm SFR})$'+f' [{Msun}/yr]',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/ECMF_plots.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None
  
    def gwIMF_plots(self, star_v, gwIMF_bySFR_eval, SFR_v, metal_mass_fraction):
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
        ax.set_ylim(1e-1,1e5)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}({\rm SFR})$',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/gwIMF_plots_Z{metallicity:.2f}.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None
    
    def IMF_plot(self, Mstar_v, IMF_v, Mtot, metallicity, SFR):
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.loglog(Mstar_v, IMF_v, linewidth=3, color='navy')
        ax.scatter(Mstar_v, IMF_v, linewidth=3, color='navy',s=1)
        ax.set_ylabel(r'$\xi_{\star}={\rm d} N/{\rm d} m$'
                      +f' [#/{Msun}]', fontsize=15)
        ax.set_xlabel(r'$M_{\rm star}$ [%s]'%(Msun), fontsize=15)
        plt.title(r'$M_{\rm ecl}$ = %.2e [%s],$\quad$ [Z] = %.2f' 
                  %(Mtot, Msun, metallicity), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/IMF_plot_Mecl{Mtot:.2e}_Z{metallicity:.2f}.pdf', bbox_inches='tight')
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
    #     ax.set_ylabel(r'$\xi_{\star}={\rm d} N/{\rm d} m$'
    #                   +f' [#/{Msun}]', fontsize=15)
    #     ax.set_xlabel(r'$M_{\rm star}$ [%s]' %(Msun), fontsize=15)
    #     ax.set_ylim(1e-2,1e10)
    #     Z = np.log10(massfrac/0.0142)
    #     ax.set_title(r"[Z] = %.2f"%(Z), fontsize=15)
    #     ax.tick_params(width=2)
    #     cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}(M_{\rm ecl})$'+f' [({Msun})]',size=15)
    #     fig.tight_layout()
    #     plt.savefig(f'figs/IMF_plots_Z{Z:.2f}.pdf', bbox_inches='tight')
    #     #plt.show(block=False)
    #     return None
    
    def IMF_plots(self, mstar_v, IMF_v_list, Mecl_v, #k_idx, 
                  metallicity):
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = plt.cm.get_cmap(name='viridis')
        eff_Mecl_v = Mecl_v#[k_idx]
        eff_IMF_v_list = np.array(IMF_v_list)#[k_idx]
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
            #print(f'{len(IMF)=}, {len(mstar_v[i])=}')
            ax.loglog(mstar_v[i], IMF, linewidth=3, c=next(currentColor))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\xi_{IMF}$', fontsize=15)
        ax.set_xlabel(r'$M_{\rm star}$ [%s]' %(Msun), fontsize=15)
        ax.set_ylim(1e-2,1e10)
        ax.set_title(r"[Z] = %.2f"%(metallicity), fontsize=15)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}(M_{\rm ecl})$',size=15)
        fig.tight_layout()
        plt.savefig(f'figs/IMF_plots_Z{metallicity:.2e}.pdf', bbox_inches='tight')
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
        ax.set_zlabel(r'$\xi_{\star}={\rm d}N/ {\rm d} m$'
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
    #         metallicity_val = np.log10(color_ax[col_ax_idx]/0.0142)
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
    #     ax.set_zlabel(r'$\xi_{\rm IGIMF}={\rm d}N/ {\rm d} m$'+
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
            metallicity_val = np.log10(color_ax[col_ax_idx]/0.0142)
            title_val = metallicity_val
            ax.set_title(f'{title} = {metallicity_val:.2f}', fontsize=20, y=0.95)
            units = f'[{Msun}/yr]'
            xi = np.reshape([df.loc[((np.isclose(df['mass_star'],ival, rtol=1e-3)) & (np.isclose(df[by_v],jval, rtol=1e-3)) 
            & (np.isclose(df['metal_mass_fraction'], metal_mass_fraction_v[col_ax_idx], rtol=1e-3))
            )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
            for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
            by_v_axis=by_v
        elif by_v == 'metal_mass_fraction':
            by_v_axis = '[Z]'
            y_ax = metal_mass_fraction_v/0.0142
            color_ax = SFR_v
            title = r'$\log_{10}(\rm SFR)$'
            SFR_val = np.log10(color_ax[col_ax_idx])
            title_val = SFR_val
            tsave = 'SFR'
            ax.set_title(f'{title} = {SFR_val:.2f}'+f' [{Msun}/yr]', fontsize=20, y=0.95)
            units = ''
            xi = np.reshape([df.loc[((np.isclose(df['mass_star'], ival, rtol=1e-3)) & (np.isclose(df[by_v], jval*0.0142, rtol=1e-3)) 
            & (np.isclose(df['SFR'], color_ax[col_ax_idx], rtol=1e-3))
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
        ax.set_zlabel(r'$\xi_{\rm IGIMF}={\rm d}N/ {\rm d} m$ '+
                      f'['+r'$\log_{10}$'+f'(#/{Msun})]', fontsize=15)
        ax.azim = azim_rot
        ax.elev = elev_rot
        ax.set_zlim(np.log10(5e-3),12)
        fig.tight_layout(rect=[0.05,0.05, .95, .95])
        plt.show(block=False)
        plt.savefig(f'figs/IGIMF_plot_3D_{by_v}_{tsave}{title_val:.2f}.pdf', bbox_inches='tight')


    def IGIMF_3D_plot_plotly(self, df, SFR_v, metal_mass_fraction_v, mstar_v, 
                            by_v='SFR', col_ax_idx=10, azim_rot=-120, elev_rot=20):

        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np

        print('Plotting IGIMF_3D_plot')

        if by_v == 'SFR':
            y_ax = SFR_v
            color_ax = metal_mass_fraction_v
            title = '[Z]'
            tsave = 'Z'
            metallicity_val = np.log10(color_ax[col_ax_idx]/0.0142)
            title_val = metallicity_val
            units = '[M<sub>⊙</sub>/yr]'
            xi = np.reshape([df.loc[((np.isclose(df['mass_star'],ival, rtol=1e-3)) & (np.isclose(df[by_v],jval, rtol=1e-3)) 
                                    & (np.isclose(df['metal_mass_fraction'],metal_mass_fraction_v[col_ax_idx], rtol=1e-3))
                                    )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
                                    for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
            by_v_axis = 'SFR'
        elif by_v == 'metal_mass_fraction':
            by_v_axis = '[Z]'
            y_ax = metal_mass_fraction_v/0.0142
            color_ax = SFR_v
            title = r'log₁₀(SFR)'
            SFR_val = np.log10(color_ax[col_ax_idx])
            title_val = SFR_val
            tsave = 'SFR'
            units = ''
            xi = np.reshape([df.loc[((df['mass_star']==ival) & (df[by_v]==jval*0.0142) 
                                    & (df['SFR']==color_ax[col_ax_idx])
                                    )]['IGIMF'].to_numpy()[0] for i,ival in enumerate(mstar_v) 
                                    for j,jval in enumerate(y_ax)], (len(mstar_v), len(y_ax)))
        else:
            raise ValueError("Set by_v either to 'SFR' or 'metal_mass_fraction'.")

        x = np.outer(mstar_v, np.ones(len(y_ax)))
        y = np.outer(y_ax, np.ones(len(mstar_v))).T

        xi_masked = xi.copy()
        #xi_masked[xi_masked <= 1e-10] = np.nan

        log_x = np.log10(x)
        log_y = np.log10(y)
        log_xi = np.log10(xi_masked)

        original_x = 10**log_x
        original_y = 10**log_y
        original_z = 10**log_xi

        dx = np.unique(np.diff(log_x[:,0]))[0]
        dy = np.unique(np.diff(log_y[0,:]))[0]

        num_colors = 15
        original_colorscale = px.colors.sequential.Viridis
        discrete_colorscale = [
            [i / (num_colors - 1), original_colorscale[int(i * (len(original_colorscale)-1) / (num_colors-1))]]
            for i in range(num_colors)
        ]

        customdata=np.stack((original_x, original_y), axis=-1)
        
        fig_plotly = go.Figure(data=[go.Surface(
            x=log_x,
            y=log_y,
            z=log_xi,
            customdata=np.expand_dims(original_z, axis=2),
            colorscale='Viridis',
            colorbar=dict(title=r'log₁₀(ξ<sub>IGIMF</sub>)'),
            cmin=np.nanmin(log_xi),
            cmax=np.nanmax(log_xi),
            connectgaps=False,
            contours={
            #    "x": {"show": True, "start": np.nanmin(log_x), "end": np.nanmax(log_x), "size": dx, "color": "white"},
            #    "y": {"show": True, "start": np.nanmin(log_y), "end": np.nanmax(log_y), "size": dy, "color": "white"},
                "x": {"show": True, "start": np.nanmin(log_x), "end": np.nanmax(log_x), "size": 0.5, "color": "white"},
                "y": {"show": True, "start": np.nanmin(log_y), "end": np.nanmax(log_y), "size": 0.5, "color": "white"},
                "z": {"show": False}
            },
            hovertemplate=(
                'mass ★: %{x:.1e}<br>' +
                'SFR: %{y:.1e}<br>' +
                'log₁₀(ξ<sub>IGIMF</sub>): %{z:.1f}<br><extra></extra>'
            )
        )])

        fig_plotly.update_layout(
            title=f'IGIMF Surface plot: {title} = {title_val:.0f}',
            scene=dict(
                xaxis=dict(title='stellar mass log₁₀(m★/M<sub>⊙</sub>)', showgrid=True, gridwidth=2, gridcolor='lightgray'),
                yaxis=dict(title=f'{by_v_axis} {units}', showgrid=True, gridwidth=2, gridcolor='lightgray'),
                zaxis=dict(title='log₁₀(ξ<sub>IGIMF</sub>) [#/M<sub>⊙</sub>]', range=[np.nanmin(log_xi), np.nanmax(log_xi)], showgrid=True, gridwidth=2, gridcolor='lightgray'),
                aspectratio=dict(x=1.2, y=1.2, z=1.0),
                aspectmode='manual'
            ),
            autosize=True,
            width=900,
            height=700,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        fig_plotly.write_html(f'figs/fig3D_{by_v}variation_{tsave}{title_val:.0f}.html')    
    
    
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
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'figs/IGIMF_plot_3Dlines.pdf', bbox_inches='tight')
        return None
     
    def Meclmax_vs_SFR_observations(self, SFR_v, MeclMax_list, k_ML=0.01):
        import pandas as pd
        import matplotlib.ticker as ticker
        r13 = pd.read_csv('data/randriamanakoto13.dat', sep=',', comment='#', index_col=False)
        l02 = pd.read_csv('data/larsen02.dat', sep=';', comment='#', index_col=False)
        b08 = pd.read_csv('data/bastian08.dat', sep=';', comment='#', index_col=False)
        def mag_to_mass(mag_V):
            #mag_V = mag_K + 2
            Mecl_max = np.power(10, np.divide(4.79 - mag_V, 2.5)) * k_ML
            return Mecl_max
        mag_to_mass = np.vectorize(mag_to_mass)
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.loglog(SFR_v, MeclMax_list, linewidth=2, color='#110689', label='this work')
        #print(f'{SFR_v=}')
        #print(f'{MeclMax_list=}')
        ax.scatter((r13['SFR(Msun/yr)']), (mag_to_mass(r13['M_K_brightest(mag)'])+2), label='Randriamanakoto+13', marker='o', alpha=0.8, color='#1975f5')
        ax.scatter((b08['SFR(Msun/yr)']), (mag_to_mass(b08['M_V_brightest(mag)'])), label='Bastian08', marker='s', alpha=0.8, color='#f59919')
        ax.scatter((l02['SFRdensity(Msun/yr/kpc^2)']*l02['A(kpc^2)']), (mag_to_mass(l02['M_V_brightest(mag)'])), label='Larsen02', marker='^', alpha=0.8, color='#f51975')
        #ax.scatter(SFR_v, MeclMax_list, linewidth=2, color='navy',s=1)
        ax.legend(fontsize=11, loc='best') 
        ax.set_ylabel(r'$M_{\rm ecl,max}$ [%s]'%(Msun), fontsize=15)
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        ax.set_ylim(10**(0.5),10**(9.5))
        ax.set_xlim(1e-6,.5e3)
        ax.set_xlim(1e-6, 1e6)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/Meclmax_vs_SFR_observations.pdf', bbox_inches='tight')
   
    def smart_format(self,x, _):
            if x == int(x):
                return str(int(x))  # no decimal for whole numbers
            elif x*10 == int(x):
                return f"{x:.1f}".rstrip('0').rstrip('.')  # remove trailing zeros for 0.1  
            else:
                return f"{x:.2f}".rstrip('0').rstrip('.')  # remove trailing zeros for 0.01 
                
    def sIMF_subplot(self, metallicity_v, Mecl_v, mstar_v, sIMF, res=20, alpha1slope='_'):
        print('Plotting sIMF_subplot')
        import matplotlib.pyplot as plt 
        import matplotlib.patheffects as pe
        import matplotlib as mpl
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        # def smart_format(x, _):
        #     if x == int(x):
        #         return str(int(x))  # no decimal for whole numbers
        #     else:
        #         return f"{x:.1f}".rstrip('0').rstrip('.')  # remove trailing zeros
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_C6s
        #num_colors=len(metallicity_v)
        #levels = np.linspace(metallicity_v[0], metallicity_v[-1], num_colors+1,
        #                     endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors)
        nrow, ncol = 2,2 #3,3 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in (list(enumerate(metallicity_v))):
                ax.annotate(r'$M_{\rm ecl}=$%.1e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                line, = ax.loglog(mstar_v[i][j], sIMF[i][j], color=next(currentColor),
                          linewidth=2, alpha=1)
                line.set_path_effects([
                pe.Stroke(linewidth=2.1, foreground='black'),  # thin black border
                pe.Normal()
                ])
                for shift in np.arange(-5,20):
                    ax.loglog(mstar_v[-2][-5], util.Kroupa01()(mstar_v[-2][-5])*np.power(10.,shift), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                ax.set_ylim(5e-3,2e12)
                ax.set_xlim(6e-2,1.6e2)
            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            # formatter = ticker.ScalarFormatter()
            # formatter.set_scientific(False)
            # formatter.set_useOffset(False)
            # formatter.format_data = lambda x: f"{x:.1f}"  # <-- force 2 decimal digits
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
            #ax.ticklabel_format(axis='x', style='plain')   
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\xi_{\star}={\rm d} N/{\rm d} m$'+
                                  f' [#/{Msun}]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass, $m$ [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", 
                            ticks=metallicity_v)
        cbar.set_label(
                                label=r'[Z]',size=15)

        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/stellarIMF_subplots_Zcolorbar_alpha1_{alpha1slope}.pdf')
      
    def sIMF_subplot_norm1(self, metallicity_v, Mecl_v, mstar_v, sIMF, res=20, alpha1slope='_'):
        print('Plotting mw_sIMF_subplot')
        import matplotlib.pyplot as plt 
        import matplotlib as mpl
        import colorcet as cc
        import itertools
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # def smart_format(x, _):
        #     if x == int(x):
        #         return str(int(x))  # no decimal for whole numbers
        #     else:
        #         return f"{x:.1f}".rstrip('0').rstrip('.')  # remove trailing zeros
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_C6s
        #num_colors=len(metallicity_v)
        #levels = np.linspace(metallicity_v[0], metallicity_v[-1], num_colors+1,
        #                     endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        #currentColors = reversed(list([cm(1.*i/num_colors) for i in range(num_colors)]))
        #currentColor = itertools.cycle(currentColors)
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors)
        
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in reversed(list(enumerate(metallicity_v))):
                ax.annotate(r'$M_{\rm ecl}=$%.0e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                norm1_idx = util.find_nearest(mstar_v[i][j], 1)
                #ax.loglog(mstar_v[i][j], np.divide(mw_sIMF[i][j], Mecl_v[i]), color=next(currentColor), alpha=0.8)
                ax.loglog(mstar_v[i][j], np.divide(sIMF[i][j], sIMF[i][j][norm1_idx]), color=next(currentColor), alpha=1)
                for shift in np.arange(-20,40):
                    ax.loglog(mstar_v[-1][-5], util.Kroupa01()(mstar_v[-1][-5])*np.power(10.,shift/2), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                #ax.set_ylim(5e-7,2e2)
                ax.set_ylim(5e-6,2e2)
                ax.set_xlim(6e-2,1.6e2)
            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(axis='x', style='plain')   
            ax.set_yscale('log')  
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Optional: hide minor tick labels
            ax.tick_params(axis='y', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='y', which='minor', length=3, direction='inout', top=True, bottom=True)
            # formatter = ticker.ScalarFormatter(useMathText=False)
            # formatter.set_scientific(False)
            # formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\frac{\xi_{\star}(m)}{\xi_{\star}(m=1)}$ ', fontsize = 14) #\quad $ [#/$M_{\odot}$]
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.2, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)

        # Optional: make minor ticks visible
        #ax.tick_params(axis='y', which='minor', length=3, direction='inout')
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        #cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cax = plt.axes([0.85, 0.14, 0.025, 0.67])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.f", 
                            ticks=metallicity_v).set_label(
                                label=r'[Z] metallicity',size=15)
        fig.suptitle(f'Stellar IMF'+f'\n normalized so that'+r' $\xi_{\star}(m=1 M_{\odot})=1$'+f' \n ({alpha1slope} '+r'low-mass slope, $\alpha_1$)', fontsize=13, y=0.95) #+r'$\xi_{\star}(m)$'
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/stellarIMF_subplots_Zcolorbar_norm1_alpha1_{alpha1slope}.pdf')
        
    def mw_sIMF_subplot(self, metallicity_v, Mecl_v, mstar_v, mw_sIMF, res=20, alpha1slope='_'):
        print('Plotting mw_sIMF_subplot')
        import matplotlib.pyplot as plt 
        import matplotlib.patheffects as pe
        import matplotlib as mpl
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_C6s
        #num_colors=len(metallicity_v)
        #levels = np.linspace(metallicity_v[0], metallicity_v[-1], num_colors+1,
        #                     endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        #currentColors = reversed(list([cm(1.*i/num_colors) for i in range(num_colors)]))
        #currentColor = itertools.cycle(currentColors)
        
                # Assume metallicity_v is sorted and uniformly spaced
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)

        # Use bin centers (e.g. metallicity_v) to get exact color at each
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors) 

        nrow, ncol = 2,2 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in list(enumerate(metallicity_v)):
                ax.annotate(r'$M_{\rm ecl}=$%.0e $M_{\odot}$'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                line, = ax.loglog(mstar_v[i][j], np.divide(mw_sIMF[i][j], Mecl_v[i]), color=next(currentColor), linewidth=2, alpha=1)
                line.set_path_effects([
                pe.Stroke(linewidth=2.1, foreground='black'),  # thin black border
                pe.Normal()
                ])
                for shift in np.arange(-30,40):
                    ax.loglog(mstar_v[-1][-5], np.divide(mstar_v[-1][-5]*util.Kroupa01()(mstar_v[-1][-5])*np.power(10.,shift/3),Mecl_v[i]), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                if alpha1slope == 'linear':
                #    ax.set_ylim(5e-9, 2e2)
                    ax.set_ylim(7e-5,2e1)
                elif alpha1slope == 'logistic':
                    ax.set_ylim(7e-5,2e1)
                ax.set_xlim(7e-2,1.6e2)
            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='y', which='major', length=5, direction='inout', left=True, right=True)
            ax.tick_params(axis='y', which='minor', length=3, direction='inout', left=True, right=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #ax.ticklabel_format(axis='x', style='plain')   
            minor_ticks = np.logspace(np.log10(1e-5), np.log10(1e2), 2-(-5)+1)  # adjust as needed
            ax.set_yticks(minor_ticks, minor=True)
            ax.yaxis.set_minor_formatter(plt.NullFormatter()) 
            # formatter = ticker.ScalarFormatter(useMathText=False)
            # formatter.set_scientific(False)
            # formatter.set_useOffset(False)
            # ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        

        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\frac{m \, \xi_{\star}(m)}{M_{\rm ecl}} \propto \frac{{\rm d} N / {\rm d} \log_{10}m}{M_{\rm ecl}}$ [#/$M_{\odot}$]', fontsize = 14)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.2, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.14, 0.025, 0.67])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", 
                            ticks=metallicity_v)
        cbar.set_label(label=r'[Z] metallicity',size=15)
        cbar.ax.tick_params(labelsize=12)
        fig.suptitle(f'Mass-weighted stellar IMF, '+r'$m \, \xi_{\star}(m)$'+'\n normalized by the embedded cluster mass, '+r'$M_{\rm ecl}$'+f' \n ({alpha1slope} '+r'low-mass slope, $\alpha_1$)', fontsize=13, y=0.95)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/massweighted_stellarIMF_subplots_Zcolorbar_Meclnorm_alpha1_{alpha1slope}.pdf')
        

    def mw_sIMF_subplot_norm1(self, metallicity_v, Mecl_v, mstar_v, mw_sIMF, res=20, alpha1slope='_'):
        print('Plotting mw_sIMF_subplot')
        import matplotlib.pyplot as plt 
        import matplotlib as mpl
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_C6s
        #num_colors=len(metallicity_v)
        #levels = np.linspace(metallicity_v[0], metallicity_v[-1], num_colors+1,
        #                     endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors) 
        
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Z in list(enumerate(metallicity_v)):
                ax.annotate(r'$M_{\rm ecl}=$%.0e'%(Mecl_v[i]), xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=1)
                norm1_idx = util.find_nearest(mstar_v[i][j], 1)
                #ax.loglog(mstar_v[i][j], np.divide(mw_sIMF[i][j], Mecl_v[i]), color=next(currentColor), alpha=0.8)
                ax.loglog(mstar_v[i][j], np.divide(mw_sIMF[i][j], mw_sIMF[i][j][norm1_idx]), color=next(currentColor), alpha=1)
                for shift in np.arange(-10,40):
                    ax.loglog(mstar_v[-1][-5], np.divide(mstar_v[-1][-5]*util.Kroupa01()(mstar_v[-1][-5])*np.power(10.,shift/2),Mecl_v[i]), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                #ax.set_ylim(5e-7,2e2)
                ax.set_ylim(2e-3,2e1)
                ax.set_xlim(6e-2,1.6e2)
            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #ax.ticklabel_format(axis='x', style='plain')   
            formatter = ticker.ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\frac{m \xi_{\star}(m)}{\xi_{\star}(m=1)} \propto \frac{{\rm d} N / {\rm d} \log_{10}m}{\xi_{\star}(m=1)}$ ', fontsize = 14) #\quad $ [#/$M_{\odot}$]
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.2, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        #cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cax = plt.axes([0.85, 0.14, 0.025, 0.67])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.1f", 
                            ticks=metallicity_v).set_label(
                                label=r'[Z] metallicity',size=15)
        fig.suptitle(f'Mass-weighted stellar IMF, '+r'$m \, \xi_{\star}(m)$'+f'\n normalized so that'+r' $\xi_{\star}(m=1 M_{\odot})=1$'+f' \n ({alpha1slope} '+r'low-mass slope, $\alpha_1$)', fontsize=13, y=0.95)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/massweighted_stellarIMF_subplots_Zcolorbar_norm1_alpha1_{alpha1slope}.pdf')
  
    def sIMF_subplot_Mecl(self, metallicity_v, Mecl_v, mstar_v, sIMF, alpha1slope='_'):
        print('Plotting sIMF_subplot_Mecl')
        import matplotlib.pyplot as plt 
        import matplotlib as mpl
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        #from mpl_toolkits import mplot3d
        #from mpl_toolkits.mplot3d import Axes3D
        Msun = r'$M_{\odot}$'
        #cm = plt.cm.get_cmap(name='viridis')
        cm = cc.cm.CET_R2
        Mecl_v = np.log10(Mecl_v)
        #num_colors=len(Mecl_v)
        #levels = np.linspace(np.log10(Mecl_v[0]), np.log10(Mecl_v[-1]), num_colors+1,
        #                      endpoint=True)
        step = Mecl_v[1] - Mecl_v[0]  # assumes uniform spacing
        levels = np.concatenate((
            [Mecl_v[0] - 0.5 * step],
            0.5 * (Mecl_v[1:] + Mecl_v[:-1]),
            [Mecl_v[-1] + 0.5 * step]
        ))
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(M) for M in Mecl_v]
        currentColor = itertools.cycle(currentColors) 
        
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        #fig, axs = plt.subplots(3, 3, figsize=(8,6))
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, M in enumerate(Mecl_v):
                ax.annotate(r'$[Z]=$%.1f'%(metallicity_v[i]), xy=(0.5, 0.9), 
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=.1)
                ax.loglog(mstar_v[j][i], sIMF[j][i], color=next(currentColor))
                for shift in np.arange(-30,40):
                    #ax.loglog(mstar_v[-1][-1], mstar_v[-1][-1]*util.Kroupa01()(mstar_v[-1][-1])*np.power(10.,shift/2), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                    ax.loglog(mstar_v[-1][-1], util.Kroupa01()(mstar_v[-1][-1])*np.power(10.,shift/2), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='y', which='major', length=5, direction='inout', left=True, right=True)
            ax.tick_params(axis='y', which='minor', length=3, direction='inout', left=True, right=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #ax.ticklabel_format(axis='x', style='plain')   
            minor_ticks = np.logspace(np.log10(1e-3), np.log10(1e12), 12-(-3)+1)  # adjust as needed
            ax.set_yticks(minor_ticks, minor=True)
            ax.yaxis.set_minor_formatter(plt.NullFormatter()) 
            ax.set_ylim(5e-3,5.e11)
            ax.set_xlim(7e-2,1.6e2)
            ax.xaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=True))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\xi_{\star}={\rm d} N/{\rm d} m$'
                                  +f' [#/{Msun}]', fontsize = 15)
        axs[nrow-1,ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]', 
                                       fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.15, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", 
                            ticks=Mecl_v)
        cbar.set_label(label=r'$\log_{10}(M_{\rm ecl})$'+f' ['+f'{Msun}]',size=15)
        fig.suptitle(f'Stellar IMF'+f' \n ({alpha1slope} '+r'low-mass slope, $\alpha_1$)', fontsize=13, y=0.95)
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/stellarIMF_subplots_Meclcolorbar_alpha1_{alpha1slope}.pdf')

    def sIMF_subplot_Mecl_supersolar(self, metallicity_v, Mecl_v, solar_metallicity=0.0142, mstar_res=5000):
        print('Plotting sIMF_subplot_Mecl')
        import matplotlib as mpl
        import igimf.classes as inst
        import matplotlib.pyplot as plt 
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        # def smart_format(x, _):
        #     if x == int(x):
        #         return str(int(x))  # no decimal for whole numbers
        #     else:
        #         return f"{x:.2f}".rstrip('0').rstrip('.')  # remove trailing zeros
        #from mpl_toolkits import mplot3d
        #from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        #cm = plt.cm.get_cmap(name='viridis')
        cm = cc.cm.CET_R2
        #num_colors=len(Mecl_v)
        #levels = np.linspace(np.log10(Mecl_v[0]), np.log10(Mecl_v[-1]), num_colors+1,
        #                      endpoint=True)
        Mecl_v_cbar = np.log10(Mecl_v)
        step = Mecl_v_cbar[1] - Mecl_v_cbar[0]  # assumes uniform spacing
        levels = np.concatenate((
            [Mecl_v_cbar[0] - 0.5 * step],
            0.5 * (Mecl_v_cbar[1:] + Mecl_v_cbar[:-1]),
            [Mecl_v_cbar[-1] + 0.5 * step]
        ))
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(M) for M in Mecl_v_cbar]
        currentColor = itertools.cycle(currentColors) 
        
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, M in enumerate(Mecl_v):
                ccolor = next(currentColor)
                ax.annotate(r'$[Z]=$%.1f'%(metallicity_v[i]), xy=(0.5, 0.9), 
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=.1)
                
                o_IMF = inst.StellarIMF(M_ecl=M, metal_mass_fraction= 10**metallicity_v[i] * solar_metallicity, alpha1slope='linear')
                if o_IMF.m_max > 1.:
                    m_star_v = np.sort(np.concatenate([[1.],np.logspace(np.log10(o_IMF.IGIMF_params['Ml']), np.log10(o_IMF.m_max), endpoint=True, num=mstar_res)]))
                else:
                    m_star_v = np.logspace(np.log10(o_IMF.IGIMF_params['Ml']), np.log10(o_IMF.m_max), endpoint=True, num=mstar_res)               
                #print(f'{m_star_v}')
                IMF_v = o_IMF.IMF_func(m_star_v)
                #print(f'For the linear alpha1 slope, M_ecl={M:.2e}, [Z]={metallicity_v[i]:.2f}\nthe integral equals {integr.simpson(np.multiply(m_star_v,IMF_v), x=m_star_v)/M:.2f}')
                m_star_v = np.append(m_star_v, m_star_v[-1]+0.01)
                sIMF = np.append(IMF_v, 0)
                ax.loglog(m_star_v, sIMF, color=ccolor, linestyle = '-', alpha=0.7)
                
                o_IMF = inst.StellarIMF(M_ecl=M, metal_mass_fraction= 10**metallicity_v[i] * solar_metallicity, alpha1slope='logistic')
                if o_IMF.m_max > 1.:
                    m_star_v = np.sort(np.concatenate([[1.],np.logspace(np.log10(o_IMF.IGIMF_params['Ml']), np.log10(o_IMF.m_max), endpoint=True, num=mstar_res)]))
                else:
                    m_star_v = np.logspace(np.log10(o_IMF.IGIMF_params['Ml']), np.log10(o_IMF.m_max), endpoint=True, num=mstar_res) 
                #print(f'{m_star_v}')
                IMF_v = o_IMF.IMF_func(m_star_v)
                #print(f'For the logistic alpha1 slope, M_ecl={M:.2e}, [Z]={metallicity_v[i]:.2f}\nthe integral equals {integr.simpson(np.multiply(m_star_v,IMF_v), x=m_star_v)/M:.2f}')
                m_star_v = np.append(m_star_v, m_star_v[-1]+0.01)
                sIMF = np.append(IMF_v, 0)
                ax.loglog(m_star_v, sIMF, color=ccolor, linestyle = '-.', alpha=0.7)
                
                mstar_v = np.sort(np.concatenate([[1.],np.logspace(np.log10(0.08), np.log10(150), endpoint=True, num=100)]))
                for shift in np.arange(-30,40):
                    #ax.loglog(mstar_v[-1][-1], mstar_v[-1][-1]*util.Kroupa01()(mstar_v[-1][-1])*np.power(10.,shift/2), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                    ax.loglog(mstar_v, util.Kroupa01()(mstar_v)*np.power(10.,shift/2), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
            ax.tick_params(axis='x', which='major', length=8, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='y', which='major', length=8, direction='inout', left=True, right=True)
            ax.tick_params(axis='y', which='minor', length=5, direction='inout', left=True, right=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #ax.ticklabel_format(axis='x', style='plain')   
            minor_ticks = np.logspace(np.log10(1e-3), np.log10(1e12), 12-(-3)+1)  # adjust as needed
            ax.set_yticks(minor_ticks, minor=True)
            ax.yaxis.set_minor_formatter(plt.NullFormatter()) 
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format)) 
            ax.set_ylim(5e-3,5.e11)
            ax.set_xlim(7e-2,1.6e2)
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$\xi_{\star}={\rm d} N/{\rm d} m$'
                                  +f' [#/{Msun}]', fontsize = 15, labelpad=20)
        axs[nrow-1,ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]', 
                                       fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.17, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.15, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", ticks=Mecl_v_cbar)
        cbar.set_label(label=r'$\log_{10}(M_{\rm ecl})$'+f' ['+f'{Msun}]',size=15)
        fig.suptitle(f'Stellar IMF, high-metallicity for \n '+r'logistic ($-\cdot$) and linear ($-$) low-mass $\alpha_1$ slope', fontsize=13, y=0.95)

        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/stellarIMF_subplots_Meclcolorbar_supersolarcomp.pdf')
        
    def mw_sIMF_subplot_Mecl(self, metallicity_v, Mecl_v, mstar_v, mw_sIMF, res=20, alpha1slope='_'):
        print('Plotting mw_sIMF_subplot_Mecl')
        import matplotlib.pyplot as plt 
        import matplotlib as mpl
        import itertools
        import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$' 
        cm = cc.cm.CET_R4
        #num_colors=len(Mecl_v)
        # levels = np.linspace(np.log10(Mecl_v[0]), np.log10(Mecl_v[-1]), num_colors+1,
        #                       endpoint=True)
        Mecl_v_cbar = np.log10(Mecl_v)
        step = Mecl_v_cbar[1] - Mecl_v_cbar[0]  # assumes uniform spacing
        levels = np.concatenate((
            [Mecl_v_cbar[0] - 0.5 * step],
            0.5 * (Mecl_v_cbar[1:] + Mecl_v_cbar[:-1]),
            [Mecl_v_cbar[-1] + 0.5 * step]
        ))
        
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        # Use bin centers (e.g. metallicity_v) to get exact color at each
        currentColors = [sm.to_rgba(M) for M in Mecl_v_cbar]
        currentColor = itertools.cycle(currentColors) 
        
        nrow, ncol = 2,2 #util.find_closest_prod(res)
        fig, axs = plt.subplots(nrow, ncol, figsize=(7,5))
        for i, ax in enumerate(axs.flat):
            for j, Mecl in list(enumerate(Mecl_v)):
                ax.annotate(r'$[Z]=$%.2f'%(metallicity_v[i]), xy=(0.5, 0.9), 
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=10, alpha=.1)
                ax.loglog(mstar_v[j][i], np.divide(mw_sIMF[j][i], Mecl_v[j]), color=next(currentColor),
                          alpha=0.8)
                for shift in np.arange(-30,20):
                    ax.loglog(mstar_v[-1][-1], mstar_v[-1][-1]*util.Kroupa01()(mstar_v[-1][-1])*np.power(10.,shift/4), color='grey', linewidth=0.2, linestyle='--', alpha=0.1)
                ax.set_ylim(5e-5,2e1)
                ax.set_xlim(6e-2,1.6e2)
            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            # formatter = ticker.ScalarFormatter(useMathText=False)
            # formatter.set_scientific(False)
            # formatter.set_useOffset(False)
            # ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
        #for nr in range(3):
        for nr in range(nrow):
            #for nc in range(3):
            for nc in range(ncol):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != nrow-1:
                    axs[nr,nc].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'$m \xi_{\star}(m) / M_{\rm ecl} \propto \frac{{\rm d} N / {\rm d} \log_{10}m}{M_{\rm ecl}} \quad$ [dimensionless]', fontsize = 14)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        axs[nrow//2,0].yaxis.set_label_coords(-.2, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        # cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", 
        #                     ticks=ticker.MultipleLocator(1)).set_label(
        #                         label=r'$\log_{10}(M_{\rm ecl})$'+f' ['+
        #                         r'$\log_{10}$'+f'({Msun})]',size=15)
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.1f", ticks=Mecl_v_cbar)
        cbar.set_label(label=r'$\log_{10}(M_{\rm ecl})$'+f' ['+
                            r'$\log_{10}$'+f'({Msun})]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/massweighted_stellarIMF_subplots_Meclcolorbar_alpha1_{alpha1slope}.pdf')
  
    def mmax_plot(self, mmax_v, parameter_space, alpha1slope='_'):
        print('Plotting mmax_plot')
        mmax_min = np.floor(np.min(mmax_v) / 15) * 15
        mmax_max = np.ceil(np.max(mmax_v) / 15) * 15
        levels = np.arange(mmax_min, mmax_max + 1e-3, 15) # Contour levels every 15 M_sun
        tick_levels = np.arange(mmax_min, mmax_max + 1e-3, 30) # Colorbar ticks every 30 M_sun
        import colorcet as cc
        fig,ax = plt.subplots(figsize=(6,4))
        x,y = np.meshgrid(np.log10(parameter_space['M_ecl']), parameter_space['[Z]'])
        #z = np.array(np.log10(mmax_v)).reshape(resolution,resolution).T
        z = np.array(mmax_v).reshape(len(parameter_space['M_ecl']),len(parameter_space['[Z]'])).T
        #cax = ax.contourf(x, y, z, resolution, cmap=cc.cm.CET_I3)
        cax = ax.contourf(x, y, z, levels=levels, cmap=cc.cm.CET_C8s)
        plt.xlabel(r'Embedded cluster mass, $\log_{10}(M_{ecl})$ [$M_{\odot}$]', fontsize=15)
        plt.ylabel(r'[Z] Metallicity', fontsize=15)
        ax.set_ylim(0,1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        cbar = fig.colorbar(cax, ticks=tick_levels)
        cbar.set_label(r'most massive star, $m_{\rm max}$ [$\rm M_{\odot}$]', fontsize=12)
        ax.set_title(f'{alpha1slope} low-mass IMF slope '+r'($\alpha_1$) for the stellar IMF', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(f'figs/mmaxplot_alpha1_{alpha1slope}.pdf')

    def mmax_igimf_plot(self, mmax_v, parameter_space, alpha1slope='_'):
        print('Plotting mmax_plot')
        mmax_min = np.floor(np.min(mmax_v) / 15) * 15
        mmax_max = np.ceil(np.max(mmax_v) / 15) * 15
        levels = np.arange(mmax_min, mmax_max + 1e-3, 15) # Contour levels every 15 M_sun
        tick_levels = np.arange(mmax_min, mmax_max + 1e-3, 30) # Colorbar ticks every 30 M_sun
        import colorcet as cc
        fig,ax = plt.subplots(figsize=(6,4))
        x,y = np.meshgrid(np.log10(parameter_space['SFR']), parameter_space['[Z]'])
        #z = np.array(np.log10(mmax_v)).reshape(resolution,resolution).T
        z = np.array(mmax_v).reshape(len(parameter_space['SFR']),len(parameter_space['[Z]'])).T
        #cax = ax.contourf(x, y, z, resolution, cmap=cc.cm.CET_I3)
        cax = ax.contourf(x, y, z, levels=levels, cmap=cc.cm.CET_C8s)
        plt.xlabel(r'Star Formation Rate, $\log_{10}(\rm SFR)$ [$M_{\odot}$/yr]', fontsize=15)
        plt.ylabel(r'$[Z]$ Metallicity ', fontsize=15)
        ax.set_ylim(0,1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        cbar = fig.colorbar(cax, ticks=tick_levels)
        cbar.set_label(r'most massive star, $m_{\rm max}$ [$\rm M_{\odot}$]', fontsize=12)
        ax.set_title(f'{alpha1slope} low-mass IMF slope '+r'($\alpha_1$) for the galaxy-wide IMF', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(f'figs/igimf_mmaxplot_alpha1_{alpha1slope}.pdf')
        
    def alpha3_plot(self, alpha3_v, parameter_space, mu=2.33):
        '''
        40.5 is the conversion of Msun/pc^3 to cm^-3
        
        Composition,          Ionization,       \mu,    n~[\text{cm}^{-3}] per M_\odot/\text{pc}^3
        Pure H,               neutral,          1.0,    40.5
        H + He (Y = 0.24),    neutral,          1.33,   30.5
        H + He,               fully ionized,    0.61,   66.4
        Molecular H2 + He,    neutral,          2.33,   17.4
        '''
        #alpha3min = np.floor(np.min(alpha3_v) * 10) / 10 - 0.1
        #alpha3max = np.ceil(np.max(alpha3_v) * 10) / 10 + 0.1
        #levels = np.linspace(alpha3min, alpha3max, endpoint=True, num=len(parameter_space['[Z]'])+1)
        levels = np.linspace(0.35, 2.45, endpoint=True, num=8)
        level_ticks = (levels[1:] + levels[:-1])/2
        import colorcet as cc
        fig,ax = plt.subplots(figsize=(6,4))
        #ax2 = ax.twiny()
        #x,y = np.meshgrid(parameter_space['log10_rho_cl'], parameter_space['[Z]'])
        x,y = np.meshgrid(np.log10(parameter_space['M_ecl']), parameter_space['[Z]'])
        number_density = np.power(10, parameter_space['log10_rho_cl'] + np.log10(40.5 / mu))
        x2,y2 = np.meshgrid(np.log10(number_density), parameter_space['[Z]'])
        z = np.array(alpha3_v).reshape(len(parameter_space['M_ecl']), len(parameter_space['[Z]'])).T
        cax = ax.contourf(x, y, z, levels, cmap=cc.cm.CET_L6)
        #cax2 = ax2.contourf(x2, y2, z, levels, cmap=cc.cm.CET_L6, alpha=0.)
        #plt.xlabel(r'$\log_{10}(\rho_{cl})$ [$\log_{10}(M_{\odot}/{\rm pc}^3)$]', fontsize=15)
        ax.set_xlabel(r'Embedded cluster mass $\log_{10}(M_{ecl})$ [$M_{\odot}$]', fontsize=15)
        #ax2.set_xlabel(r'number density [$\log_{10}({\rm cm}^{-3})$]', fontsize=15)
        ax.set_ylabel(r'$[Z]$ Metallicity ', fontsize=15)
        
        caxes = plt.axes([0.85, 0.17, 0.025, 0.76])
        cbar = fig.colorbar(cax, cax=caxes, ticks = level_ticks)
        cbar.set_label(r'high-mass stellar IMF slope, $\alpha_3$', fontsize=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.savefig(f'figs/alpha3plot.pdf')


    def alpha3_plot_data(self, alpha3_v, parameter_space, mu=2.33):
        '''
        40.5 is the conversion of Msun/pc^3 to cm^-3
        
        Composition,          Ionization,       \mu,    n~[\text{cm}^{-3}] per M_\odot/\text{pc}^3
        Pure H,               neutral,          1.0,    40.5
        H + He (Y = 0.24),    neutral,          1.33,   30.5
        H + He,               fully ionized,    0.61,   66.4
        Molecular H2 + He,    neutral,          2.33,   17.4
        '''
        #alpha3min = np.floor(np.min(alpha3_v) * 10) / 10 - 0.1
        #alpha3max = np.ceil(np.max(alpha3_v) * 10) / 10 + 0.1
        #levels = np.linspace(alpha3min, alpha3max, endpoint=True, num=len(parameter_space['[Z]'])+1)
        levels = np.linspace(0.35, 2.45, endpoint=True, num=8)
        level_ticks = (levels[1:] + levels[:-1])/2
        import colorcet as cc
        fig,ax = plt.subplots(figsize=(6,4))
        #ax2 = ax.twiny()
        #x,y = np.meshgrid(parameter_space['log10_rho_cl'], parameter_space['[Z]'])
        x,y = np.meshgrid(np.log10(parameter_space['M_ecl']), parameter_space['[Z]'])
        number_density = np.power(10, parameter_space['log10_rho_cl'] + np.log10(40.5 / mu))
        x2,y2 = np.meshgrid(np.log10(number_density), parameter_space['[Z]'])
        z = np.array(alpha3_v).reshape(len(parameter_space['M_ecl']), len(parameter_space['[Z]'])).T
        cax = ax.contourf(x, y, z, levels, cmap=cc.cm.CET_L6)
        #cax2 = ax2.contourf(x2, y2, z, levels, cmap=cc.cm.CET_L6, alpha=0.)
        #plt.xlabel(r'$\log_{10}(\rho_{cl})$ [$\log_{10}(M_{\odot}/{\rm pc}^3)$]', fontsize=15)
        ax.set_xlabel(r'Embedded cluster mass $\log_{10}(M_{ecl})$ [$M_{\odot}$]', fontsize=15)
        #ax2.set_xlabel(r'number density [$\log_{10}({\rm cm}^{-3})$]', fontsize=15)
        ax.set_ylabel(r'$[Z]$ Metallicity ', fontsize=15)
        
        core_mass_function_slopes = [1.9,1.28,0.99,0.7]
        core_mass_function_masses = [2e4,1e5,2.4e4,1.4e6]
        core_mass_function_metallicities = [0., 0.5, 0.5, 0.5]
        colors = [cax.cmap(cax.norm(val)) for val in core_mass_function_slopes]
        for i, (slope, mass, Z, color) in enumerate(zip(core_mass_function_slopes,
                                                       core_mass_function_masses,
                                                       core_mass_function_metallicities,
                                                       colors)):
            if i == 0:
                marker = 'o'  # circle for Motte+18
            else:
                marker = 's'  # square for Kinman+25
            ax.scatter(np.log10(mass), Z, c=[color], edgecolor='k',
                       marker=marker, s=80, zorder=10)
        
        caxes = plt.axes([0.85, 0.17, 0.025, 0.76])
        cbar = fig.colorbar(cax, cax=caxes, ticks = level_ticks)
        cbar.set_label(r'high-mass stellar IMF slope, $\alpha_3$', fontsize=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.savefig(f'figs/alpha3plot.pdf')

    def alpha1_plot(self, alpha1_v, parameter_space, resolution=15, alpha1slope='_'):
        import matplotlib.ticker as ticker
        import pandas as pd
        import colorcet as cc
        fig,ax = plt.subplots(figsize=(6,4))
        x,y = np.meshgrid(parameter_space['log10_rho_cl'], parameter_space['Zmassfrac'])
        z = np.array(alpha1_v).reshape(len(parameter_space['log10_rho_cl']),len(parameter_space['Zmassfrac'])).T
        cax = ax.contourf(x, y, z, len(parameter_space['Zmassfrac'])+1, cmap=cc.cm.CET_L5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        #ax.set_ylim(0,0.05)
        ax.set_xlabel(r'$\log_{10}(\rho_{cl})$ [$\log_{10}(M_{\odot}/{\rm pc}^3)$]', fontsize=15)
        #plt.ylabel(r'[$Z$]', fontsize=15)
        ax.set_ylabel(r'metal mass fraction, $Z$', fontsize=15)
        cbar = fig.colorbar(cax, format="%.2f")
        cbar.set_label(r'$\alpha_1$', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'figs/alpha1plot_alpha1_{alpha1slope}.pdf')

    def Cook23_plot(self):
        import matplotlib.ticker as ticker
        import pandas as pd
        Cook23 = pd.read_csv('data/Cook23.dat', comment='#', sep='&')
        Cook23bin = pd.read_csv('data/Cook23bin.dat', comment='#', sep='&')
        Cook23bin10 = Cook23bin.loc[Cook23bin['SFR-Method']=='1-10Myr']
        Cook23binHalpha = Cook23bin.loc[Cook23bin['SFR-Method']=='Halpha']
        Dinnbier22 = pd.read_csv('data/Dinnbier22.dat', comment='#', sep=',')
        D22low = Dinnbier22.iloc[:3]
        D22high = Dinnbier22.iloc[3:]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.axhline(y=100, xmin=-4, xmax=1, linewidth=4, color='purple', label='100% of stars in a cluster (at birth)')
        ax.semilogy(D22low['logSFR'], np.power(10, D22low['Gamma'])*100, linestyle='--', color='purple', linewidth=2)
        ax.semilogy(D22high['logSFR'], np.power(10, D22high['Gamma'])*100, linestyle='--', color='purple', linewidth=2)
        ax.fill_between(D22high['logSFR'], np.power(10, D22high['Gamma'])*100, np.power(10, D22low['Gamma'])*100, where=(np.power(10, D22low['Gamma'].to_numpy())*100<np.power(10, D22high['Gamma'].to_numpy())*100),  alpha=0.1, color='purple', label=r'DKA22, dyn. sim. young clusters ($=10$ Myr)')
        
        ax.errorbar(Cook23binHalpha['sfrsig-bin'], Cook23binHalpha['Gamma'], xerr=Cook23binHalpha['sfrsig-u'], yerr=Cook23binHalpha['Gamma-u'], color='red', ecolor='red', elinewidth=3, capsize=0, label=r'C+23, obs. young clusters ($<10$ Myr) based on H$_{\alpha}$', marker='o', markersize=9, alpha=0.4)
        ax.errorbar(Cook23bin10['sfrsig-bin'], Cook23bin10['Gamma'], xerr=Cook23bin10['sfrsig-u'], yerr=Cook23bin10['Gamma-u'], color='black', ecolor='black', elinewidth=3, capsize=0,label=r'C+23, obs. young clusters ($<10$ Myr), fully resolved', marker='s', markersize=9, alpha=0.4)
        ax.scatter(Cook23binHalpha['sfrsig-bin'], Cook23binHalpha['Gamma'], color='red', marker='o', s=50, alpha=0.8)
        ax.scatter(Cook23bin10['sfrsig-bin'], Cook23bin10['Gamma'],  color='black',  marker='s', s=50, alpha=0.8)
        
        #ax.errorbar(Cook23['logSFRsig'], Cook23['Gamma'], yerr=Cook23['Gamma-u'], fmt='o', color='blue', ecolor='blue', elinewidth=1, capsize=0,label=r'C23 lit', marker='o', alpha=0.4)
        ax.set_xlim(-3.7, 0.5)
        ax.set_ylim(2e-2, 2e2)
        ax.tick_params(width=2, axis='both', labelsize=13)
        ax.tick_params(width=2, axis='both', labelsize=13)
        ax.legend(loc='lower left', fontsize=12, frameon=True)
        ax.set_ylabel(r'percentage of stars in a cluster, $\Gamma_e$ (%)', fontsize=15)
        ax.set_xlabel(r'SFR surface density,  $\log(\Sigma_{\rm SFR})$ [$M_{\star} {\rm yr}^{-1} {\rm kpc}^{-2}$]', fontsize=15)
        ax.set_title(f'All stars originally form in embedded clusters, but are soon dynamically lost.\n In young clusters (10 Myr) only 10% to 60% of stars remain.', fontsize=13)
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        formatter.format_data = lambda x: f"{x:.1f}"  # <-- force 2 decimal digits
        def smart_format(x, _):
            if x == int(x):
                return str(int(x))  # no decimal for whole numbers
            else:
                return f"{x:.2f}".rstrip('0').rstrip('.')  # remove trailing zeros

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(smart_format))        
        #ax.yaxis.set_major_formatter(formatter)
        #ax.ticklabel_format(style='plain', axis='y')  
        fig.tight_layout()
        plt.savefig('figs/Cook23.pdf', bbox_inches='tight')
        #plt.show(block=False)

    def Fig11_plot(self):
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
              
    def k_Z_plot_loglog(self, Z_massfrac_v, k_IMF_Z_list, m_max_Z_list, Mecl_v,
                 k_v_supsol, m_max_v_supsol,
                 m_star_max=150, alpha1slope='_', solar_metallicity=0.0142):
        print('Plotting k_Z_plot')
        import pandas as pd
        import matplotlib as mpl
        import matplotlib.ticker as ticker
        from matplotlib.colors import ListedColormap
        import colorcet as cc
        import itertools
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cm = cc.cm.CET_C6s
        cm2 = cc.cm.CET_C6s
        #cm = ListedColormap(cm(np.linspace(0, 0.8, 256)))
        #cm2 = ListedColormap(cm2(np.linspace(0, 0.8, 256)))

        mmax_Mecl = pd.read_csv('data/mmaxMecl.dat', comment='#')

        num_colors = len(Z_massfrac_v)
        metallicity_v = np.log10(Z_massfrac_v/solar_metallicity)
        #Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        print(f'{step=}')
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        print(f'{levels=}')
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        SFR_colormap = (Z_massfrac_v) #np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        #currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        #currentColor = iter(currentColors)
        #currentColors2 = [cm2(1.*i/num_colors) for i in range(num_colors)]
        #currentColor2 = iter(currentColors2)
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors) 
        norm2 = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm2 = mpl.cm.ScalarMappable(cmap=cm2, norm=norm2)
        currentColors2 = [sm2.to_rgba(z) for z in metallicity_v]
        currentColor2 = itertools.cycle(currentColors2) 
        
        fig, ax = plt.subplots(2,1, figsize=(5,7), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        #ax2 = ax.twinx()
        #ax.plot(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        #ax.scatter(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        ax[1].plot(np.log10(Mecl_v), np.log10(k_v_supsol), linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        ax[0].plot(np.log10(Mecl_v), np.log10(m_max_v_supsol), linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        for i,Z in enumerate(Z_massfrac_v):
            color = next(currentColor)
            #ax[1].semilogx(Mecl_v, np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.99)
            ax[1].plot(np.log10(Mecl_v), np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=1)
            #ax.plot((Mecl_v), (k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.4)
            color2 = next(currentColor2)
            #ax[0].loglog(Mecl_v, m_max_Z_list[i], linewidth=3, color=color2, alpha=0.99)
            ax[0].plot(np.log10(Mecl_v), np.log10(m_max_Z_list[i]), linewidth=3, color=color2, alpha=1)
            #ax2.plot((Mecl_v), m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
        ax[1].set_ylabel(r'$\log_{10}(k_{\rm \mathrm{\star}})$', fontsize=13)
        ax[1].set_xlabel(r'embedded cluster stellar mass, $\log_{10}(M_{\rm ecl})$ [$M_{\odot}$]', fontsize=13, x=0.45)
        #ax[0].scatter(np.power(10,mmax_Mecl['log10_M_ecl'].astype(float)), np.power(10,mmax_Mecl['log10_m_max'].astype(float)), edgecolors='black', marker='o', facecolors='none',zorder=100, label='YJK23')
        ax[0].scatter(mmax_Mecl['log10_M_ecl'].astype(float), mmax_Mecl['log10_m_max'].astype(float), edgecolors='black', marker='*', facecolors='none',zorder=100, label='YJK23')
        ax[0].legend(loc='lower right', fontsize='11', frameon=False)
        
        #ax.set_ylabel(r'$k_{\rm IMF}$', fontsize=15)
        #ax.set_xlabel(r'$M_{\rm ecl}$[$M_{\odot}$]', fontsize=15)
        for axs in ax:
            axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
            axs.tick_params(axis='both', which='major', labelsize=12)
            axs.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', length=6)
            axs.tick_params(axis='y', which='major', left=True, right=True, direction='inout', length=6)
            axs.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', length=4)
            axs.tick_params(axis='y', which='minor', left=True, right=True, direction='inout', length=4)
            #axs.set_xlim(9e-1, 2e10)    
        ax[0].set_ylabel(r'most massive star, $\log_{10}(m_{\rm max})$ [$M_{\odot}$]', fontsize=13)
        ax[0].tick_params(labelbottom=False)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("top", size="5%", pad="5%")#, pack_start=True)
        #cax = divider.new_vertical(size="5%", pad=2.6, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", ticks=metallicity_v,orientation="horizontal")
        cbar.set_label(label=r'$[Z]$ Metallicity ',size=14)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=12)
        # f'most massive star to embedded cluster mass\n'+
        plt.suptitle(r'$m_{\rm max}-M_{\rm ecl}$ relation'+f'\n{alpha1slope} '+r'low-mass slope ($\alpha_1$)', fontsize=13, x=0.53)
        #ax[0].set_ylim(10**-0.5, 10**2.5)
        #ax[0].set_xlim(10**0, 10**6)
        #ax[1].set_xlim(10**0, 10**6)        
        ax[0].set_ylim(-0.5, 2.5)
        ax[0].set_xlim(0, 6)
        ax[1].set_xlim(0, 6)
        ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(5)) 
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1)) 
        ax[0].tick_params(axis='y', which='major', left=True, right=True, direction='inout', length=6)
        ax[0].tick_params(axis='y', which='minor', left=True, right=True, direction='inout', length=4)
        ax[0].minorticks_on()
        # Existing lines
        ax[0].set_ylim(-0.5, 2.5)
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_vs_k_mmax_alpha1_{alpha1slope}.pdf', bbox_inches='tight')
        #plt.show(block=False)

    def k_Z_plot(self, Z_massfrac_v, k_IMF_Z_list, m_max_Z_list, Mecl_v,
                 k_v_supsol, m_max_v_supsol, solar_metallicity=0.0142,
                 alpha1slope='_'):
        print('Plotting k_Z_plot')
        import pandas as pd
        import matplotlib as mpl
        import matplotlib.ticker as ticker
        import matplotlib.patheffects as pe
        import colorcet as cc
        import itertools
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #cm = cc.cm.CET_C6s
        #cm2 = cc.cm.CET_C6s
        cm = cc.cm.CET_C6s
        cm2 = cc.cm.CET_C6s
        #cm = ListedColormap(cm(np.linspace(0, 0.8, 256)))
        #cm2 = ListedColormap(cm2(np.linspace(0, 0.8, 256)))

        mmax_Mecl = pd.read_csv('data/mmaxMecl.dat', comment='#')
        Mecl_v = np.log10(Mecl_v)
        num_colors = len(Z_massfrac_v)
        metallicity_v = np.log10(Z_massfrac_v/solar_metallicity)
        #Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        #print(f'{step=}')
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        #print(f'{levels=}')
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        currentColors2 = [cm2(1.*i/num_colors) for i in range(num_colors)]
        currentColor2 = iter(currentColors2)
        
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors) 
        norm2 = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm2 = mpl.cm.ScalarMappable(cmap=cm2, norm=norm2)
        currentColors2 = [sm2.to_rgba(z) for z in metallicity_v]
        currentColor2 = itertools.cycle(currentColors2) 
        
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(2,1, figsize=(5,7), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        #ax2 = ax.twinx()
        #ax.plot(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        #ax.scatter(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        ax[1].plot(Mecl_v, np.log10(k_v_supsol), linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        ax[0].plot(Mecl_v, m_max_v_supsol, linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        for i,Z in enumerate(Z_massfrac_v):
            color = next(currentColor)
            #ax[1].semilogx(Mecl_v, np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.99)
            line1, = ax[1].plot(Mecl_v, np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.99)
            line1.set_path_effects([
                pe.Stroke(linewidth=3.2, foreground='black'),  # thin black border
                pe.Normal()
            ])
            color2 = next(currentColor2)
            line2, = ax[0].plot(Mecl_v, m_max_Z_list[i], linewidth=3, color=color2, alpha=0.99)
            line2.set_path_effects([
                pe.Stroke(linewidth=3.2, foreground='black'),  # thin black border
                pe.Normal()
            ])
            #ax2.plot((Mecl_v), m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
        ax[1].set_ylabel(r'$\log_{10}(k_{\rm \mathrm{\star}})$', fontsize=13)
        ax[1].set_xlabel(r'embedded cluster stellar mass, $\log_{10}(M_{\rm ecl})$ [$M_{\odot}$]', fontsize=13, x=0.45)
        #ax[0].scatter(np.power(10,mmax_Mecl['log10_M_ecl'].astype(float)), np.power(10,mmax_Mecl['log10_m_max'].astype(float)), edgecolors='black', marker='*', facecolors='none',zorder=100, label='YJK23')
        ax[0].scatter(mmax_Mecl['log10_M_ecl'].astype(float), np.power(10,mmax_Mecl['log10_m_max'].astype(float)), edgecolors='black', marker='*', facecolors='none',zorder=100, label='YJK23')
        ax[0].legend(loc='upper left', fontsize='11', frameon=False)
        
        
        #ax.set_ylabel(r'$k_{\rm IMF}$', fontsize=15)
        #ax.set_xlabel(r'$M_{\rm ecl}$[$M_{\odot}$]', fontsize=15)
        for axs in ax:
            #axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            #axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
            axs.tick_params(axis='both', which='major', labelsize=12)
            axs.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', length=6)
            axs.tick_params(axis='y', which='major', left=True, right=True, direction='inout', length=6)
            axs.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', length=4)
            axs.tick_params(axis='y', which='minor', left=True, right=True, direction='inout', length=4)
            #axs.set_xlim(9e-1, 2e10)    
        ax[0].set_ylabel(r'most massive star, $\log_{10}(m_{\rm max})$ [$M_{\odot}$]', fontsize=13)
        ax[0].tick_params(labelbottom=False)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("top", size="5%", pad="5%")#, pack_start=True)
        #cax = divider.new_vertical(size="5%", pad=2.6, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", ticks=ticker.MultipleLocator(1),orientation="horizontal")
        cbar.set_label(label=r'$[Z]$ Metallicity ',size=14)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=12)
        # f'most massive star to embedded cluster mass\n'+
        plt.suptitle(r'$m_{\rm max}-M_{\rm ecl}$ relation'+f'\n{alpha1slope} '+r'low-mass slope ($\alpha_1$)', fontsize=13, x=0.53)
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_vs_k_mmax_alpha1_{alpha1slope}.pdf', bbox_inches='tight')
        

    def k_Z_alpha3alpha1_plot(self, Z_massfrac_v, m_max_Z_list, Mecl_v, m_max_v_supsol,
                 alpha3_v_list, alpha3_Z_list, alpha1_v_list, alpha1_Z_list, solar_metallicity=0.0142,
                 alpha1slope='_'):
        print('Plotting k_Z_alpha3alpha1_plot')
        import pandas as pd
        import matplotlib as mpl
        import matplotlib.ticker as ticker
        import matplotlib.patheffects as pe
        import colorcet as cc
        import itertools
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #cm = cc.cm.CET_C6s
        #cm2 = cc.cm.CET_C6s
        cm = cc.cm.CET_C6s
        cm2 = cc.cm.CET_C6s
        #cm = ListedColormap(cm(np.linspace(0, 0.8, 256)))
        #cm2 = ListedColormap(cm2(np.linspace(0, 0.8, 256)))

        mmax_Mecl = pd.read_csv('data/mmaxMecl.dat', comment='#')
        Mecl_v = np.log10(Mecl_v)
        num_colors = len(Z_massfrac_v)
        metallicity_v = np.log10(Z_massfrac_v/solar_metallicity)
        #Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        #print(f'{step=}')
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        #print(f'{levels=}')
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        currentColors2 = [cm2(1.*i/num_colors) for i in range(num_colors)]
        currentColor2 = iter(currentColors2)
        
        norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        currentColors = [sm.to_rgba(z) for z in metallicity_v]
        currentColor = itertools.cycle(currentColors) 
        norm2 = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm2 = mpl.cm.ScalarMappable(cmap=cm2, norm=norm2)
        currentColors2 = [sm2.to_rgba(z) for z in metallicity_v]
        currentColor2 = itertools.cycle(currentColors2) 
        
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(2,1, figsize=(5,7), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0})
        #ax2 = ax.twinx()
        #ax.plot(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        #ax.scatter(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        idx_supsol = np.where(np.array(m_max_v_supsol) > 1.)[0]
        
        ax[1].axvline(Mecl_v[np.min(idx_supsol)], color='black',linewidth=1,linestyle='--', zorder=-1)
        ax[1].plot(Mecl_v[idx_supsol], np.array(alpha3_v_list)[idx_supsol], linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        ax[0].plot(Mecl_v, np.array(alpha1_v_list), linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        for i,Z in enumerate(Z_massfrac_v):
            color = next(currentColor)
            #ax[1].semilogx(Mecl_v, np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.99)
            idx = np.where(np.array(m_max_Z_list[i]) > 1.)[0]
            ax[1].axvline(Mecl_v[np.min(idx)], color=color,linewidth=3,linestyle='--', zorder=-1)
            #ax[1].axvline(Mecl_v[np.min(idx)], color=color,linewidth=1,linestyle='--', zorder=-1)
            line1, = ax[1].plot(Mecl_v[idx], np.array(alpha3_Z_list[i])[idx], linewidth=3, color=color, alpha=0.99)
            line1.set_path_effects([
                pe.Stroke(linewidth=3.2, foreground='black'),  # thin black border
                pe.Normal()
            ])
            color2 = next(currentColor2)
            line2, = ax[0].semilogy(Mecl_v, alpha1_Z_list[i], linewidth=3, color=color2, alpha=0.99)
            line2.set_path_effects([
                pe.Stroke(linewidth=3.2, foreground='black'),  # thin black border
                pe.Normal()
            ])
            #ax2.plot((Mecl_v), m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
        ax[0].set_ylabel(r'low-mass sIMF slope $\alpha_1$', fontsize=13)
        ax[1].set_xlabel(r'embedded cluster stellar mass, $\log_{10}(M_{\rm ecl})$ [$M_{\odot}$]', fontsize=13, x=0.45)
        #ax[0].scatter(np.power(10,mmax_Mecl['log10_M_ecl'].astype(float)), np.power(10,mmax_Mecl['log10_m_max'].astype(float)), edgecolors='black', marker='*', facecolors='none',zorder=100, label='YJK23')
        #ax[0].scatter(mmax_Mecl['log10_M_ecl'].astype(float), np.power(10,mmax_Mecl['log10_m_max'].astype(float)), edgecolors='black', marker='*', facecolors='none',zorder=100, label='YJK23')
        #ax[0].legend(loc='upper left', fontsize='11', frameon=False)
        ax[0].set_ylim(0.2,11)
        ax[1].set_ylim(0.2,2.4)
        ax[0].set_xlim(0.3,10.3)
        ax[1].set_xlim(0.3,10.3)
        
        #ax.set_ylabel(r'$k_{\rm IMF}$', fontsize=15)
        #ax.set_xlabel(r'$M_{\rm ecl}$[$M_{\odot}$]', fontsize=15)
        for axs in ax:
            #axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            #axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
            axs.tick_params(axis='both', which='major', labelsize=12)
            axs.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', length=6)
            axs.tick_params(axis='y', which='major', left=True, right=True, direction='inout', length=6)
            axs.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', length=4)
            axs.tick_params(axis='y', which='minor', left=True, right=True, direction='inout', length=4)
            #axs.set_xlim(9e-1, 2e10)    
        ax[1].set_ylabel(r'high-mass sIMF slope $\alpha_3$', fontsize=13)
        ax[0].tick_params(labelbottom=False)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("top", size="5%", pad="5%")#, pack_start=True)
        #cax = divider.new_vertical(size="5%", pad=2.6, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", ticks=metallicity_v,orientation="horizontal")
        cbar.set_label(label=r'$[Z]$ Metallicity ',size=14)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=12)
        ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
        # f'most massive star to embedded cluster mass\n'+
        plt.suptitle(r'$\alpha_{\star}-M_{\rm ecl}$ relation'+f'\n{alpha1slope} '+r'low-mass slope ($\alpha_1$)', fontsize=13, x=0.53)
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_vs_k_mmax_alpha3alpha1_alpha1_{alpha1slope}.pdf', bbox_inches='tight')
        
     
     
    def k_Z_alpha3_plot(self, Z_massfrac_v, m_max_Z_list, Mecl_v,
                 alpha3_v_list, alpha3_Z_list, solar_metallicity=0.0142,
                 alpha1slope='_'):
        print('Plotting k_Z_alpha3_plot')
        import pandas as pd
        import matplotlib as mpl
        import matplotlib.ticker as ticker
        import matplotlib.patheffects as pe
        import colorcet as cc
        import itertools
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cm = cc.cm.CET_C6s
        cm2 = cc.cm.CET_C6s
        
        mmax_Mecl = pd.read_csv('data/mmaxMecl.dat', comment='#')
        Mecl_v = np.log10(Mecl_v)
        num_colors = len(Z_massfrac_v)
        metallicity_v = np.log10(Z_massfrac_v/solar_metallicity)
        
        # alpha3 = np.linspace(0.5,2.3,endpoint=True,num=10)
        # step = 0.2  # assumes uniform spacing
        # levels = np.concatenate((
        #     [alpha3[0] - 0.5 * step],
        #     0.5 * (alpha3[1:] + alpha3[:-1]),
        #     [alpha3[-1] + 0.5 * step]
        # ))
        
        step = metallicity_v[1] - metallicity_v[0]  # assumes uniform spacing
        #print(f'{step=}')
        levels = np.concatenate((
            [metallicity_v[0] - 0.5 * step],
            0.5 * (metallicity_v[1:] + metallicity_v[:-1]),
            [metallicity_v[-1] + 0.5 * step]
        ))
        
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        
        norm2 = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm2 = mpl.cm.ScalarMappable(cmap=cm2, norm=norm2)
        currentColors2 = [sm2.to_rgba(z) for z in metallicity_v]
        currentColor2 = itertools.cycle(currentColors2) 
        
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax.plot(Mecl_v, alpha3_v_list, linewidth=3, color='black', linestyle=':', alpha=0.99, zorder=1000)
        for i,Z in enumerate(Z_massfrac_v):
            idx = np.where(np.array(m_max_Z_list[i]) > 1.)[0]
            color2 = next(currentColor2)
            line2, = ax.plot(Mecl_v[idx], np.array(alpha3_Z_list[i])[idx], linewidth=3, color=color2, alpha=0.99)
            line2.set_path_effects([
                pe.Stroke(linewidth=3.2, foreground='black'),  # thin black border
                pe.Normal()
            ])
        ax.set_xlabel(r'embedded cluster stellar mass, $\log_{10}(M_{\rm ecl})$ [$M_{\odot}$]', fontsize=13, x=0.45)
        #ax.scatter(mmax_Mecl['log10_M_ecl'].astype(float), np.power(10,mmax_Mecl['log10_m_max'].astype(float)), edgecolors='black', marker='*', facecolors='none',zorder=100, label='YJK23')
        #ax.legend(loc='upper left', fontsize='11', frameon=False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='x', which='major', top=True, bottom=True, direction='inout', length=6)
        ax.tick_params(axis='y', which='major', left=True, right=True, direction='inout', length=6)
        ax.tick_params(axis='x', which='minor', top=True, bottom=True, direction='inout', length=4)
        ax.tick_params(axis='y', which='minor', left=True, right=True, direction='inout', length=4)
        ax.set_ylabel(r'high-mass IMF slope, $\alpha_3$', fontsize=13)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad="5%")#, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.0f", ticks=metallicity_v,orientation="horizontal")
        cbar.set_label(label=r'$[Z]$ Metallicity ',size=14)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=12)
        plt.suptitle(r'$\alpha_{3}-M_{\rm ecl}$ relation'+f'\n{alpha1slope} '+r'low-mass slope ($\alpha_1$)', fontsize=13, x=0.53)
        fig.tight_layout()
        plt.savefig(f'figs/Mecl_vs_k_mmax_alpha3_alpha1_{alpha1slope}.pdf', bbox_inches='tight')
           
        
    def create_centered_gradient_image(self, ax, extent, cmap, alpha=0.4):
        """
        Create a centered gradient image and set it as background in the given axis.
        """
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        gradient = np.vstack((gradient, gradient))  # Repeat to ensure it covers the background

        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent, alpha=alpha, zorder=-1000)
        return ax
    
    def Kroupa_canonical_plot(self):
        K01_params = {
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': 0.08,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': 150
                        }
        canonical_IMF = util.Kroupa01(sIMF_params=K01_params)
        Mstar_v = np.logspace(np.log10(0.07), np.log10(150), endpoint=True, num=100)
        from matplotlib import ticker
        fig, ax = plt.subplots(1,1,figsize=(6, 4))
        ax.loglog(Mstar_v, canonical_IMF(Mstar_v), color='black', lw=2)
        ax.set_title('Kroupa IMF (2001, canonical IMF)', fontsize=14)
        ax.set_ylabel(r'sIMF, $\xi_{\rm can}(m) \quad$ [#/$M_{\odot}$]', fontsize=15)
        ax.set_xlabel(r'stellar mass, $m$ [$M_{\odot}$]', fontsize=15)
        ax.axvline(1, linestyle='--', color='dimgray')
        ax.axvline(0.5, linestyle='--', color='dimgray')
        ax.axvline(150, linestyle='--', color='dimgray')
        ax.set_xlim(7e-2,1.8e2)
        ax.annotate('', xy=(0.08, 1), xytext=(0.5, 1), arrowprops=dict(arrowstyle='<->', color='dimgray'))
        ax.text(0.22, 0.4, r'$\alpha_1$', horizontalalignment='center', color='black', fontsize=14)
        ax.text(0.22, 1.4, f'\n'+r'$\alpha_1 = \alpha_2 - 1$', horizontalalignment='center', color='dimgray', fontsize=11)
        ax.annotate('', xy=(0.5, 1), xytext=(1, 1), arrowprops=dict(arrowstyle='<->', color='dimgray'))
        ax.text(0.75, 0.4, r'$\alpha_2$', horizontalalignment='center', color='black', fontsize=14)
        ax.annotate('', xy=(1, 1), xytext=(150, 1), arrowprops=dict(arrowstyle='<->', color='dimgray'))
        ax.text(10, 0.4, r'$\alpha_3$', horizontalalignment='center', color='black', fontsize=14)
        ax.text(0.37, 3e-3, r'$m_{12}=0.5 \, M_{\odot}$', rotation=90, va='center', ha='left', color='dimgray')
        ax.text(0.77, 2.5e-3, r'$m_{23}=1 \, M_{\odot}$', rotation=90, va='center', ha='left', color='dimgray')
        ax.text(110, 3e-3, r'$m_{\rm max}$ (e.g., $150 \, M_{\odot}$)', rotation=90, va='center', ha='left', color='dimgray')
        ax.text(0.09, 3e-3, r'$m_{\rm min}$ (e.g., $0.08 \, M_{\odot}$)', rotation=90, va='center', ha='left', color='dimgray')
        ax.tick_params(width=2, axis='both', labelsize=13)
        ax.tick_params(width=2, axis='both', labelsize=13)     
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        
        #ax.ticklabel_format(axis='x', style='plain')   
        plt.tight_layout()
        plt.savefig('figs/Kroupa_IMF.pdf')

    def IGIMF_massweighted_panels(self,df, i=7, res=15, solar_metallicity=0.0142):
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
        #SFR_select = SFR_v[[1,4,-6,-2]]
        #metallicity_v = np.unique(df['[Z]'])
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
        metallicity_v = np.log10(metal_mass_fraction_v/solar_metallicity)[np.where(metal_mass_fraction_v > 0.1 * solar_metallicity)[0]]
        #Z_select = metallicity_v[[7,11,-2,-1]]
        
        plt.clf()
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], 2,
                             endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = reversed(list([cm(1.*k/num_colors) for k in range(num_colors)]))
        currentColors = ['#b10000', '#0000b1']
        currentColor = itertools.cycle(currentColors)
        
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        # #for i, ax in enumerate(axs.flat):
        for j, Z in enumerate(reversed(metal_mass_fraction_v)):
            if Z > 0.1 * solar_metallicity:
                ax.annotate(r'$\log_{10}(SFR/[M_{\odot}/yr])=$%.1f'%(np.log10(SFR_v[i]))#+"["+Msun+'/yr]'
                            , xy=(0.5, 0.9),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=12, alpha=1)
                df_sel = df.loc[(df['SFR']==SFR_v[i]) & (df['metal_mass_fraction']==Z)]
                print(f"{df_sel['mass_star']=}")
                idx = np.where(np.isclose(df_sel['mass_star'],1, rtol=6e-2))[0]
                print(f'{idx=}')
                canonical_IMF = util.Kroupa01()(df_sel['mass_star'].values)
                print(f'{canonical_IMF[idx]=}')
                ax.loglog(df_sel['mass_star'], np.multiply(df_sel['mass_star'],df_sel['IGIMF'])/df_sel['IGIMF'].iloc[idx].values, color=next(currentColor), alpha=1, label = f'[Z] = {np.log10(Z/solar_metallicity):.1f}', linewidth=2)
        
        ax.loglog(df_sel['mass_star'], np.multiply(df_sel['mass_star'],canonical_IMF)/canonical_IMF[idx], color='grey', linewidth=3, linestyle='--', alpha=1, zorder=1, label='Kroupa et al. (2001)')
                
        vDC24 = pd.read_csv('data/vanDokkumConroy24.dat',delimiter=',')
        vDC24.columns = vDC24.columns.str.strip()
        ax.loglog(vDC24['logMass'], vDC24['logIMF'], color='#006400', linewidth=2, linestyle='-.', alpha=1, zorder=10)
        x_vDC24 = np.array([0.1,0.5,1,15.6])
        vDC24r = pd.read_csv('data/vanDokkumConroy24range.dat',delimiter=',')
        vDC24r.columns = vDC24r.columns.str.strip()
        mid_index = len(vDC24r) // 2
        vDC24r1 = vDC24r.iloc[:mid_index]
        vDC24r2 = vDC24r.iloc[mid_index:]
        y1_interp = np.interp(x_vDC24, vDC24r1['logMass'], vDC24r1['logIMF'])
        y2_interp = np.interp(x_vDC24, vDC24r2['logMass'], vDC24r2['logIMF'])
        ax.fill_between(x_vDC24, y1_interp, y2_interp, color='#006400', alpha=0.8, label='vanDokkum & Conroy (2024)', zorder=11)
        ax.axhline(y=1, color='k', linestyle=':', alpha=0.1, linewidth=0.1, zorder=-10)
        ax.axvline(x=1, color='k', linestyle=':', alpha=0.1, linewidth=0.1, zorder=-11)
        ax.set_ylim(2e-4,5e3)
        ax.set_xlim(7.5e-2,1.45e2)
        ax.tick_params(axis='x', direction='in', which='both')
        ax.tick_params(axis='y', direction='in', which='both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.set_ylabel(r'$m \xi_{\rm IGIMF}={\rm d} N/{\rm d}\log_{10}(m) $'
                                  #+f'[#/'+r'$\log_{10}$'+f'{Msun}]'
                                  , fontsize = 15)
        ax.set_xlabel(r'stellar mass [$M_{\odot}$]',
                                        fontsize = 15)
        #axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        #axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='lower left', ncol=1, fontsize=12, frameon=False)
        # plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        # cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        # cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.1f", 
        #                     ticks=ticker.MultipleLocator(1)).set_label(
        #                         label=r'[Z] metallicity',size=15)
                            
        ax.tick_params(axis='both', which='major', length=5, direction='inout', top=True, bottom=True)
        ax.tick_params(axis='both', which='minor', length=3, direction='inout', top=True, bottom=True)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))  
        ax.tick_params(labelsize=10)
        #ax.tick_params(width=2)
        ax.set_title(f'A special case of the IGIMF:\n'+r'high-$z$ elliptical galaxies', fontsize=15)
        
        #fig.tight_layout(rect=[0,0,0.85,1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('figs/IGIMF_massweighted_plot_concordance.pdf')
        plt.show(block=False)
        
        
    def IGIMF_panels_norm(self, df, alpha1slope='logistic'):
        print('Plotting IGIMF_panels_norm')
        import matplotlib.pyplot as plt 
        import numpy as np
        import colorcet as cc
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.ticker as ticker

        Msun = r'$M_{\odot}$'
        cm = cc.cm.CET_C6s
        nrow, ncol = 2, 2

        SFR_v = np.unique(df['SFR'])
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])

        # colors = [
        #     '#000000', '#7F0AE6', '#0A27E6', '#0ABAE6', 
        #     '#0AE611', '#E67F0A', '#E60A27', '#E60ADF'
        # ]
        cmap = plt.get_cmap('tab20')  # or 'cividis', 'viridis', etc.
        num_colors = len(metal_mass_fraction_v)
        colors = [cmap(x) for x in np.linspace(0, 1, num_colors, endpoint=False)]

        fig, axs = plt.subplots(nrow, ncol, figsize=(7, 5), sharex=True, sharey=True)

        for i, ax in enumerate(axs.flat):
            idx_i = i % len(SFR_v)
            ax.annotate(r'$\log_{10}(SFR/[M_{\odot}/yr])=$%.1f'%(np.log10(SFR_v[idx_i])),
                        xy=(0.55, 0.95),
                        xycoords='axes fraction', verticalalignment='top',
                        horizontalalignment='center', fontsize=12, alpha=1)

            for j, Z in enumerate(metal_mass_fraction_v):
                df_sel = df.loc[(df['SFR'] == SFR_v[idx_i]) & (df['metal_mass_fraction'] == Z)]

                color = colors[j % len(colors)]

                if ax == axs.flat[-1]:
                    ax.loglog(df_sel['mass_star'], df_sel['IGIMF']/(SFR_v[idx_i]*1e7), color=color, alpha=1, label=f'[Z] = {np.log10(Z/0.0142):+.1f}')
                    ax.legend(ncol=4, frameon=False, fontsize=10, loc='lower right', bbox_to_anchor=(.9, -.7))
                else:
                    ax.loglog(df_sel['mass_star'], df_sel['IGIMF']/(SFR_v[idx_i]*1e7), color=color, alpha=1)

                for shift in np.arange(-20, 20):
                    ax.loglog(df_sel['mass_star'],
                            util.Kroupa01()(df_sel['mass_star'].values) * np.power(10., shift),
                            color='grey', linewidth=0.2, linestyle='--', alpha=0.05, zorder=-100)

                #ax.set_ylim(2e-3, 2e12)
                ax.set_ylim(2e-8, 2e2)
                ax.set_xlim(7e-2, 1.6e2)

            ax.tick_params(axis='x', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='x', which='minor', length=3, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='y', which='major', length=5, direction='inout', left=True, right=True)
            ax.tick_params(axis='y', which='minor', length=3, direction='inout', left=True, right=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(axis='x', style='plain')
            minor_ticks = np.logspace(np.log10(1e-8), np.log10(1e3), 3-(-8)+1)  # adjust as needed
            ax.set_yticks(minor_ticks, minor=True)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())

        axs[nrow//2, 0].set_ylabel(r'$\frac{\xi_{\rm IGIMF}}{M_{*,\rm tot}}=\frac{{\rm d} N/{\rm d} m}{M_{*,\rm tot}}$'+
                                f' [#/{Msun}'+r'$^2$'+']', fontsize=15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize=15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)

        fig.suptitle(f'Galaxy-wide IMF (IGIMF)\n normalized by the total stellar mass, '+r'$M_{*,\rm tot}$,'+' produced in '+r'$\Delta t = 10^7$ yr'+f'\n({alpha1slope} '+r'low-mass slope, $\alpha_1$)',
                    fontsize=13, y=.98)
        #fig.tight_layout(rect=[0,0,0.85,1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/IGIMF_plotsnorm_alpha1_{alpha1slope}.pdf')
        
    def IGIMF_panels(self, df, alpha1slope='logistic'):
        print('Plotting IGIMF_panels')
        import matplotlib.pyplot as plt 
        import numpy as np
        import colorcet as cc
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.ticker as ticker
        
        Msun = r'$M_{\odot}$'
        #cm = cc.cm.CET_C6s
        nrow, ncol = 2, 2
        
        SFR_v = np.unique(df['SFR'])
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
        
        colors = [
            '#000000', '#7F0AE6', '#0A27E6', '#0ABAE6', 
            '#0AE611', '#E67F0A', '#E60A27', '#E60ADF', 'olive'
        ]
        iterColors = iter(colors)
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(7, 5), sharex=True, sharey=True)
        
        for i, ax in enumerate(axs.flat):
            ax.annotate(r'$\log_{10}(SFR/[M_{\odot}/yr])=$%.1f'%(np.log10(SFR_v[i])), 
                        xy=(0.55, 0.95),
                        xycoords='axes fraction', verticalalignment='top', 
                        horizontalalignment='center', fontsize=12, alpha=1)
            
            for j, Z in enumerate(metal_mass_fraction_v):
                df_sel = df.loc[(df['SFR'] == SFR_v[i]) & (df['metal_mass_fraction'] == Z)]
                m_max = df_sel['m_max']
                m_star_v = np.logspace(np.log10(0.08), np.log10(m_max), 100)
                #IGIMF_v = df_sel['IGIMF_func'].iloc[0](m_star_v)
                
                #color = colors[j]
                color = next(iterColors)
                
                if ax == axs.flat[-1]:
                    ax.loglog(df_sel['mass_star'], df_sel['IGIMF'], color=color, alpha=1, label=f'[Z] = {np.log10(Z/0.0142):+.1f}')
                    #ax.legend(loc='lower center', ncol=2, frameon=False, fontsize=10)
                    ax.legend(ncol=4, frameon=False, fontsize=10, loc='lower right', bbox_to_anchor=(.9, -.7))
                else:
                    ax.loglog(df_sel['mass_star'], df_sel['IGIMF'], color=color, alpha=1)
                
                for shift in np.arange(-5, 20):
                    ax.loglog(df_sel['mass_star'], 
                            util.Kroupa01()(df_sel['mass_star'].values) * np.power(10., shift), 
                            color='grey', linewidth=0.2, linestyle='--', alpha=0.05, zorder=-100)
                

            ax.tick_params(axis='both', which='major', length=5, direction='inout', top=True, bottom=True)
            ax.tick_params(axis='both', which='minor', length=3, direction='inout', top=True, bottom=True)
            #ax.tick_params(axis='y', which='major', length=5, direction='inout', left=True, right=True)
            #ax.tick_params(axis='y', which='minor', length=3, direction='inout', left=True, right=True)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.smart_format))        

            #ax.ticklabel_format(axis='x', style='plain')   
            minor_ticks = np.logspace(np.log10(1e-3), np.log10(1e12), 12-(-3)+1)  # adjust as needed
            ax.set_yticks(minor_ticks, minor=True)
            ax.yaxis.set_minor_formatter(plt.NullFormatter()) 
            ax.set_ylim(2e-3, 2e14)
            ax.set_xlim(7e-2, 1.6e2)
        
        axs[nrow//2, 0].set_ylabel(r'$\xi_{\rm IGIMF}={\rm d} N/{\rm d} m$'+
                                f' [#/{Msun}]', fontsize=15)
        axs[nrow-1, ncol//2].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize=15)
        axs[nrow//2,0].yaxis.set_label_coords(-.15, 1)
        axs[nrow-1, ncol//2].xaxis.set_label_coords(0., -.15)
        
        fig.suptitle(f'Galaxy-wide IMF (IGIMF)\n({alpha1slope} '+r'low-mass slope, $\alpha_1$)', 
                    fontsize=13, y=.98)
        #fig.tight_layout(rect=[0.,0.05,1.,.85])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig(f'figs/IGIMF_plots_alpha1_{alpha1slope}.pdf')

    def zetaI_II_plot(self, df, alpha1slope=None):
        print('Plotting zetaI_II_plot')
        import matplotlib.ticker as ticker
        import colorcet as cc
        import pandas as pd
        import scipy.integrate as integr
        import scipy.interpolate as interp
        df['zetaI'] = np.nan
        df['zetaII'] = np.nan
        SFR_v = np.unique(df['SFR'])#[:-1]
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
                df_sel = df.loc[(np.isclose(df['SFR'], s, rtol=1e-3)) & (np.isclose(df['metal_mass_fraction'], z, rtol=1e-3))] #& (df['SFR']==s) & (df['metal_mass_fraction']==z)]
                igimf_interp = interp.interp1d(df_sel['mass_star'], df_sel['IGIMF'])
                igimf_norm = np.float64(igimf_interp(1.))
                igimf_low = integr.quad(igimf_interp, mstar_v[0], 1.)[0]
                igimf_high = integr.quad(igimf_interp, 1., mstar_v[-1])[0]
                igimf_lowratio = igimf_low/igimf_norm
                igimf_highratio = igimf_high/igimf_norm
                print(f'{igimf_norm=}')
                print(f'{igimf_lowratio=}')
                print(f'{igimf_highratio=}')
                df['zetaI'].loc[(np.isclose(df['SFR'], s, rtol=1e-3)) & (np.isclose(df['metal_mass_fraction'], z, rtol=1e-3))] = float(igimf_lowratio / canon_lowratio)
                df['zetaII'].loc[(np.isclose(df['SFR'], s, rtol=1e-3)) & (np.isclose(df['metal_mass_fraction'], z, rtol=1e-3))] = float(igimf_highratio / canon_highratio)

        Msun = r'$M_{\odot}$'
        fig,ax = plt.subplots(figsize=(6,4))
        df_plot = df[['SFR', 'metal_mass_fraction', 'zetaI','zetaII']].drop_duplicates()
        SFR_grid, metallicity_grid = np.meshgrid(np.log10(SFR_v), np.log10(metal_mass_fraction_v/0.0142))
        points = df_plot[['SFR', 'metal_mass_fraction']].values
        points[:,0] = np.log10(points[:,0])
        points[:,1] = np.log10(points[:,1]/0.0142)
        df_plot['zetaI'].fillna(df_plot['zetaI'].mean(), inplace=True)
        values = df_plot['zetaI'].values
        print('zetaI values \n {values}')
        zetaI_grid = interp.griddata(points, values, (SFR_grid, metallicity_grid), method='cubic')
        cax = ax.contourf(SFR_grid, metallicity_grid, zetaI_grid, 15, cmap=cc.cm.CET_L6)
        plt.xlabel(r'$\log_{10}({\rm SFR})$'+f'[{Msun}/yr]', fontsize=15)
        plt.ylabel(r'[$Z$]', fontsize=15)
        #ax.set_ylim(-.45,0.45)
        cbar = fig.colorbar(cax)
        cbar.set_label(r'$\zeta_{\rm I}$', fontsize=15)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'figs/zetaIplot_alpha1_{alpha1slope}.pdf')
        
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
        plt.savefig(f'figs/zetaIIplot_alpha1_{alpha1slope}.pdf')

    def alpha1_Z_plot(self, solar_metallicity=0.0142):
        import re
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.colors import to_rgba
        import pandas as pd
        Z_massfrac_v=np.linspace(0,solar_metallicity*10)
        alpha1Z = pd.read_fwf('data/Yan2024.dat', comment = '#')
        final_df = util.import_Yan24all_xi()
        def alpha_1_func(metals):
            func = lambda Z: (1.3 + 63 * (Z - solar_metallicity))
            return np.vectorize(func)(metals)
        def alternative_alpha1(Z, maximum, Z_midpoint, growth_rate):
            return np.divide(maximum, 1 + np.exp(-np.multiply(growth_rate,Z-Z_midpoint)))
        fig, ax = plt.subplots(1,1, figsize=(6,5))
        alpha1_Z_list = alpha_1_func(Z_massfrac_v)
        ax.plot(Z_massfrac_v, alpha1_Z_list, linewidth=3, color='#110689', alpha=0.8, label=r'linear $\alpha_1(Z)$')
        ax.plot(Z_massfrac_v, alternative_alpha1(Z_massfrac_v, 1.3*2, solar_metallicity, 2/solar_metallicity), color='#f5210e', linewidth=3, label=r'logistic $\alpha_1(Z)$')
        ax.set_ylabel(r'low-mass IMF slope, $\alpha_1$', fontsize=14)
        ax.set_xlabel(r'metal mass fraction, $Z = M_Z/M_{\rm gas}$', fontsize=14)
        ax.axhline(1.3 +1.3, linestyle=':', color='black', linewidth=1)
        ax.axhline(1.3 -1.3, linestyle=':', color='black', linewidth=1)
        ax.axhline(1.3, linestyle='-.', color='black', linewidth=1, label=r'$\alpha_{1,\rm canon}=1.3$')
        ax.axvline(solar_metallicity, linestyle='--', color='black', linewidth=1, label=r'$Z_{\odot}$')
        ax.axvline(3*solar_metallicity, linestyle='-', color='#068911', linewidth=3)
        ax.text(3*solar_metallicity+0.001, 5, f'M31 Bulge (Saglia+10)', fontsize=9, alpha=1, color='#068911', rotation=90)
        ax.axvline(7*solar_metallicity, linestyle='-', color='#89067e', linewidth=3)
        ax.text(7*solar_metallicity+0.001, 3, f'some GCs in M49     (Cohen+03)', fontsize=9, alpha=1, color='#89067e', rotation=90)
        
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'H']

        # Find first character where 'source' contains '1' or '2'
        def insert_plus(s):
            return re.sub(r'(1|2)', r'+\1', s, count=1)
        mask = final_df['source'].str.contains(r'1|2')
        final_df.loc[mask, 'source'] = final_df.loc[mask, 'source'].apply(insert_plus)

        unique_sources = final_df['source'].unique()
        assert len(unique_sources) <= len(markers), "Not enough marker types for all sources"
        for source, marker in zip(unique_sources, markers):
            subset = final_df[final_df['source'] == source]
            ax.scatter(
                subset['Z'],
                subset['alpha1'],
                label=source,
                marker=marker,
                color='black',
                alpha=0.1,
                zorder=10
            )
             
        Cinquegrana22color = to_rgba('#f50ee2', alpha=0.1)
        ax.axvspan(0.04,0.1,edgecolor='#f50ee2', facecolor=Cinquegrana22color, label='Cinquegrana+22', zorder=-10, hatch='///', linewidth=0.5, alpha=0.1)
        rect = Rectangle((0.04, -1), 0.06, 11,  # x=0.04 to 0.1, y=0 to 2
                        facecolor='none',
                        edgecolor='#f50ee2',
                        hatch='///',
                        linewidth=1,
                        alpha=0.1,
                        zorder=-5)
        ax.add_patch(rect)

        x_start = 0.11
        x_end = 0.15
        n_bands = 40
        x_edges = np.linspace(x_start, x_end, n_bands + 1)
        alpha_start = 0.4

        for i in range(n_bands):
            x0 = x_edges[i]
            x1 = x_edges[i + 1]
            alpha = alpha_start * (1 - i / (n_bands - 1))  # fade to 0
            plt.axvspan(x0, x1, ymin=0, ymax=1, facecolor='gray', alpha=alpha, linewidth=0)

        y_value = 6
        delta_y = 0.1
        ax.annotate('', xy=(0.15, y_value), xytext=(0.11, y_value),
             arrowprops=dict(arrowstyle='->', linestyle='-', alpha=0.5))
        ax.text(0.115, y_value + delta_y, f'low-'+r'$z$'+' AGNs \nFloris+24', fontsize=9, alpha=1, color='black')

        ax.set_ylim(-0.5,10)
        ax.set_xlim(0,0.15)
        ax.legend(frameon=False, fontsize=9, loc='upper right', bbox_to_anchor=(1.1, 1.25), ncol=4)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'figs/alpha1_alternative.pdf', bbox_inches='tight')
