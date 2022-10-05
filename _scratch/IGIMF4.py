"""
Based on Yan, Jerabkova, and Kroupa (2021, A&A)


The IGIMF class contains 14 functions:

|-- SFR_func (construct from the IGIMF input)
|-- metal_mass_fraction_func (construct from the IGIMF input)
|-- normalization (used to normalize both stellar and cluster IMF)
|-- IGIMF (output)
    |-- stellar_IMF (normalized)
    |-- |-- initial_mass_function
    |-- |-- |-- alpha_1_func
    |-- |-- |-- alpha_2_func
    |-- |-- |-- alpha_3_func
    |-- |-- |-- |-- rho_cl
    |-- |-- |-- |-- x_alpha_3
    |-- ECMF (normalized)
    |-- |-- embedded_cluster_mass_function
    |-- |-- |-- beta


Furthermore,
 
|-- main runs the class when IGIMF.py is executed
|-- IMF_plot plots the IMF (either for stars or embedded clusters)
|-- Fig11_plot reproduces Fig11 of Kroupa+20 ()
"""
import numpy as np
from scipy import optimize
import scipy.integrate as integr
import pandas as pd
import util

class Downsizing:
    """Downsizing relations as introduced by Thomas et al., (2005)"""
    
    def __init__(self, M_igal: float) -> None:
        self.M_igal = M_igal # [Msun]
        self.downsizing_time = self.delta_tau(M_igal) # [yr]
        self.SFR = self.SFR_func(self.M_igal, self.downsizing_time) # [Msun/yr] star_formation_rate
        
    # Galaxy functions 
    def delta_tau(self, M_igal):
        '''
        Returns delta tau in Gyr for the downsizing relation as it is expressed in Recchi+09
        
        M_igal is expressed in Msun and ranges from 1e6 to 1e12
        '''
        return 8.16 * np.e**(-0.556 * np.log10(M_igal) + 3.401) + 0.027       
            
    def SFR_func(self, M_igal, downsizing_time):
        '''SFR [Msun/yr] assuming the downsizing time (Thomas et al., 2005)'''
        return np.divide(M_igal, downsizing_time * 1e9)

class IGIMF:
    ''' Computes the Integrated Galaxy-wide Initial Mass Function of stars'''

    #def __init__(self, mass_metals, mass_gas, star_formation_rate, t):
    def __init__(self, metal_mass_fraction: float, SFR: float, 
                 delta_t=1e7, solar_metallicity=0.0142, delta_alpha=63, 
                 m_star_max = 150, m_star_min=0.08,  
                 M_ecl_max = 1e10, M_ecl_min=5, suppress_warnings=True) -> None: #, downsizing_time: float, t: float) -> None:
        
        self.delta_t = delta_t # [yr] duration of SF epoch
        self.solar_metallicity = solar_metallicity
        self.delta_alpha = delta_alpha # (page 3)
        self.m_star_max = m_star_max # [Msun] stellar mass upper limit, Yan et al. (2017)
        self.m_star_min = m_star_min # [Msun]
        self.M_ecl_max = M_ecl_max # [Msun] most-massive ultra-compact-dwarf galaxy.
        self.M_ecl_min = M_ecl_min # [Msun] !!!! I've taken the lower limit from Eq. (8)
        self.metal_mass_fraction = metal_mass_fraction
        self.SFR = SFR # [Msun/yr] star_formation_rate
        
        self.beta_ECMF = self.beta_func()
        self.metallicity = np.log10(self.metal_mass_fraction/ self.solar_metallicity)
        self.Mtot = self.SFR * self.delta_t # Total stellar mass produced at a given delta t timestep
        self.alpha_1 = float(self.alpha_1_func())
        self.alpha_2 = float(self.alpha_2_func())
        if suppress_warnings:
            import warnings
            warnings.filterwarnings('ignore')

    # stellar IMF functions
    def alpha_1_func(self):
        r"""Eq. (4) pt.1"""
        # 1.3 + 63 * (1e7/1e9- 0.0142)
        return 1.3 + self.delta_alpha * (self.metal_mass_fraction - self.solar_metallicity)
    
    def alpha_2_func(self):
        r"""Eq. (4) pt. 2 $\alpha_2 - \alpha_1 = 1$ always holds"""
        return 1 + self.alpha_1
        
    def rho_cl(self, M_ecl):
        r"""Eq. (7) core density of the molecular cloud 
        which forms the embedded star cluster
        In units of [Mstar/pc^3]
    
        For example, for M_ecl = 1000 Msun:
        >>> rho_cl(10**3)
        gives a core density of 4.79e4 Msun/pc^3"""
        return 10**(0.61 * np.log10(M_ecl) + 2.85)
    
    def x_alpha_3_func(self, M_ecl):
        r"""Eq. (6)"""
        return (-0.14 * self.metallicity + 0.99 * np.log10(self.rho_cl(M_ecl)/1e6))

    def alpha_3_func(self, M_ecl):
        r"""Eq. (5)"""
        x_alpha_3 = self.x_alpha_3_func(M_ecl)
        if x_alpha_3 < -0.87:
            return 2.3
        else:
            return -0.41 * x_alpha_3 + 1.94
        
    def initial_mass_function(self, m, alpha_3=None, m_max=None):
        # power_law = lambda m, power: m**(-power)
        # if np.logical_and(m_max >= self.m_star_min, m_max < 0.5):
        #     return (2 * power_law(m, self.alpha_1))
        # if np.logical_and(m_max >= 0.5, m_max < 1.):
        #     return (2 * power_law(m, self.alpha_1)
        #             + power_law(m, self.alpha_2))
        # if np.logical_and(m_max >= 1., m_max <= self.m_star_max):
        #     return (2 * power_law(m, self.alpha_1)
        #             + power_law(m, self.alpha_2)
        #             + power_law(m, alpha_3))
        # else:
        #     return 0.
        if np.logical_and(m>=self.m_star_min, m<0.5):
            return m**(-self.alpha_1) * 2
        elif np.logical_and(m>=0.5, m<1.):
            return m**(-self.alpha_2)
        elif m>=1.:
            return util.normalized(m, m**(-alpha_3), condition=m_max)
        else:
            return 0.
        
    def stellar_IMF(self, M_ecl):
        r"""Eq. (1) Returns the stellar IMF xi: $\xi_* = d N_* / d m $"""
        alpha_3 = self.alpha_3_func(M_ecl)
        k_star, m_max = util.normalization_IMF(self.alpha_1, self.alpha_2, alpha_3, M_ecl, 
                                           self.m_star_min, self.m_star_max)
        IMF_func = lambda m: k_star * self.initial_mass_function(m, alpha_3=alpha_3, m_max=m_max)
        IMF_weighted_func = lambda m: util.weighted_func(m, IMF_func)
        return k_star, m_max, np.vectorize(IMF_func), np.vectorize(IMF_weighted_func), self.alpha_1, self.alpha_2, alpha_3

    # embedded cluster mass functions
    def beta_func(self):
        r"""Eq. (11) """
        return -0.106 * np.log10(self.SFR) + 2
    
    def embedded_cluster_mass_function(self, M_ecl, m_max=None):
        r"""Eq. (8)"""
        if M_ecl>=self.M_ecl_min:
            return util.normalized(M_ecl, M_ecl**(-self.beta_ECMF), condition=m_max)
        else:
            return 0.
               
    def ECMF(self):
        k_ecl, M_max = util.normalization_ECMF(self.embedded_cluster_mass_function, self.beta_ECMF,
                                          self.SFR * self.delta_t, self.M_ecl_min, self.M_ecl_max)
        ECMF_func = lambda M_ecl: k_ecl * self.embedded_cluster_mass_function(M_ecl, m_max=M_max)
        ECMF_weighted_func = lambda M_ecl: util.weighted_func(M_ecl, ECMF_func)
        return k_ecl, M_max, np.vectorize(ECMF_func), np.vectorize(ECMF_weighted_func)
    
    def gwIMF_integrand_func(self, M_ecl, m, ECMF_func):
        k_star, m_max, IMF_func, IMF_weighted_func, alpha_3 = self.stellar_IMF(M_ecl)
        return IMF_func(m) * ECMF_func(M_ecl)
    
    # IGIMF functions
    def gwIMF(self, resolution=20):
        r"""Eq. (12)"""
        k_ecl, M_max, ECMF_func, ECM_weighted_func = self.ECMF()
        #return lambda m: integr.quad(self.gwIMF_integrand_func, self.M_ecl_min, M_max, args=(m, ECMF_func))[0]   
        return lambda m: integr.quadrature(igimf.gwIMF_integrand_func, igimf.M_ecl_min, M_max, args=(m, ECMF_func), vec_func=False, rtol=1e-5)[0]  
    
    # Plotting functions
    def ECMF_plot(self, Mecl_v, ECMF_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.loglog(Mecl_v, ECMF_v, linewidth=3, color='navy')
        ax.scatter(Mecl_v, ECMF_v, linewidth=3, color='navy')
        ax.set_ylabel(r'$\xi_{ECMF}$', fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$ [%s]'%(Msun), fontsize=15)
        plt.title(r'SFR = %.2e [%s/yr]' %(self.SFR, Msun), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)
            
    def beta_ECMF_plot(self, SFR_v, beta_ECMF_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.semilogx(SFR_v, beta_ECMF_v, linewidth=3, color='navy')
        ax.scatter(SFR_v, beta_ECMF_v, linewidth=3, color='navy')
        ax.set_ylabel(r'$\beta_{ECMF}$', fontsize=15)
        ax.set_xlabel(r'SFR', fontsize=15)
        #plt.title(r'SFR = %.2e [%s/yr]' %(self.SFR, Msun), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)
               
    def MeclMax_plot(self, SFR_v, MeclMax_list):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.loglog(SFR_v, MeclMax_list, linewidth=3, color='navy')
        ax.scatter(SFR_v, MeclMax_list, linewidth=3, color='navy')
        ax.set_ylabel(r'$M_{\rm ecl,max}$ [%s]'%(Msun), fontsize=15)
        ax.set_xlabel(r'SFR [%s/yr]'%(Msun), fontsize=15)
        #plt.title(r'SFR = %.2e [%s/yr]' %(self.SFR, Msun), fontsize=15)
        #ax.set_ylim(5e-2,1e8)
        #ax.set_xlim(1e-4,1e3)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)
        
    def Mecl_power_beta_plot(self, Mecl_v, beta_ECMF_list, SFR_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        cm = plt.cm.get_cmap(name='magma')
        num_colors = len(beta_ECMF_list)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        for b in beta_ECMF_list:
            y = Mecl_v**(-b)
            color = next(currentColor)
            ax.loglog(Mecl_v, y, linewidth=3, c=color)
            #ax.scatter(Mecl_v, y, linewidth=3, c=color)
        ax.set_ylabel(r'$\xi_{ECMF}$ not normalized', fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$', fontsize=15)
        plt.title(r'$k_{\rm ecl}=1$', fontsize=15)
        ax.set_ylim(1e-15,1e1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)
        
    def ECMF_plots(self, M_ecl_v, ECMF_v_list, SFR_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Msun = r'$M_{\odot}$'
        cm = plt.cm.get_cmap(name='magma')
        num_colors = len(ECMF_v_list)
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), 100, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        SFR_colormap = (SFR_v)#np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        #dummy_cax = ax.scatter(M_ecl_v,ECMF_v_list[19], linewidth=3, vmin=SFR_colormap[0], vmax=SFR_colormap[-1], c=np.log10(SFR), cmap=cm, alpha=1)
        for i,ECMF in enumerate(ECMF_v_list):
            ax.loglog(M_ecl_v,ECMF, linewidth=3, c=next(currentColor))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        ax.set_ylabel(r'$\xi_{ECMF}$', fontsize=15)
        ax.set_xlabel(r'$M_{\rm ecl}$ [%s]' %(Msun), fontsize=15)
        #ax.set_ylim(1e-1,1e5)
        ax.tick_params(width=2)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$\log_{10}({\rm SFR})$',size=15)
        fig.tight_layout()
        plt.savefig(f'ECMF_plots.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None

    def IMF_plot(self, Mstar_v, IMF_v, Mtot):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.loglog(Mstar_v, IMF_v, linewidth=3, color='navy')
        ax.scatter(Mstar_v, IMF_v, linewidth=3, color='navy')
        ax.set_ylabel(r'$\xi_{IMF}$', fontsize=15)
        ax.set_xlabel(r'$M_{\rm star}$ [%s]'%(Msun), fontsize=15)
        plt.title(r'$M_{ecl}$ = %.2e [%s/yr],$\quad$ [Z] = %.2f' %(Mtot, Msun, self.metallicity), fontsize=15)
        #ax.set_ylim(1e-8,1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)
        
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
        plt.savefig(f'IMF_plots_{self.metallicity:.1f}_mmax{self.m_star_max:.2e}.pdf', bbox_inches='tight')
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
        ax.set_zlabel(r'$\xi_{\star}={\rm d}N_{\star}/ {\rm d} m$', fontsize=15)
        ax.set_title(r'stellar IMF $\xi_{\star}(m_{\star},M_{\rm ecl},Z)$', fontsize=17)
        fig.tight_layout()
        plt.show(block=False)
        #plt.savefig(f'IMF_plot_3D.pdf', bbox_inches='tight')
        
    def sIMF_subplot(self, metallicity_v, Mecl_v, mstar_v, sIMF):
        import matplotlib.pyplot as plt 
        import itertools
        #import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits import mplot3d
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #cm = plt.cm.get_cmap(cc.bmy)
        cm = plt.cm.get_cmap(name='plasma')
        levels = np.linspace(metallicity_v[0], metallicity_v[-1], 100, endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(metallicity_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        #fig, axs = plt.subplots(3, 3, figsize=(8,6))
        fig, axs = plt.subplots(4, 5, figsize=(8,6))
        for i, ax in enumerate(axs.flat):
            for j, Z in enumerate(metallicity_v):
                ax.annotate(r'$M_{ecl}=$%.2e'%(Mecl_v[i]), xy=(0.5, 0.9), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=10, alpha=.1)
                ax.loglog(mstar_v, sIMF[i][j], color=next(currentColor))
                ax.set_ylim(5e-3,1e13)
                ax.set_xlim(2e-2,5e2)
        #for nr in range(3):
        for nr in range(4):
            #for nc in range(3):
            for nc in range(5):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != 4-1:
                    axs[nr,nc].set_xticklabels([])
        #axs[1,0].set_ylabel(r'$\xi_{stellar}$', fontsize = 15)
        #axs[2, 1].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize = 15)
        axs[4//2,0].set_ylabel(r'$\xi_{stellar}$', fontsize = 15)
        axs[4-1, 5//2].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize = 15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'[Z]',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('stellarIMF_subplots_Zcolorbar.pdf')


    def sIMF_subplot_Mecl(self, metallicity_v, Mecl_v, mstar_v, sIMF):
        import matplotlib.pyplot as plt 
        import itertools
        #import colorcet as cc
        import matplotlib.ticker as ticker
        from mpl_toolkits import mplot3d
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #cm = plt.cm.get_cmap(cc.bmy)
        cm = plt.cm.get_cmap(name='viridis')
        levels = np.linspace(np.log10(Mecl_v[0]), np.log10(Mecl_v[-1]), 100, endpoint=True)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=cm)
        plt.clf()
        num_colors=len(Mecl_v)
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = itertools.cycle(currentColors)
        #fig, axs = plt.subplots(3, 3, figsize=(8,6))
        fig, axs = plt.subplots(4, 5, figsize=(8,6))
        for i, ax in enumerate(axs.flat):
            for j, M in enumerate(Mecl_v):
                ax.annotate(r'$Z=$%.2f'%(metallicity_v[i]), xy=(0.5, 0.9), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=10, alpha=.1)
                ax.loglog(mstar_v, sIMF[j][i], color=next(currentColor))
                ax.set_ylim(5e-3,1e13)
                ax.set_xlim(2e-2,5e2)
        #for nr in range(3):
        for nr in range(4):
            #for nc in range(3):
            for nc in range(5):
                if nc != 0:
                    axs[nr,nc].set_yticklabels([])
                #if nr != 3-1:
                if nr != 4-1:
                    axs[nr,nc].set_xticklabels([])
        #axs[1,0].set_ylabel(r'$\xi_{stellar}$', fontsize = 15)
        #axs[2,1].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize = 15)
        axs[4//2,0].set_ylabel(r'$\xi_{stellar}$', fontsize = 15)
        axs[4-1,5//2].set_xlabel(r'stellar mass [$M_{\odot}$]', fontsize = 15)
        #divider = make_axes_locatable(axs.flat[-1])
        plt.subplots_adjust(bottom=0., right=0.95, top=1.)
        cax = plt.axes([0.85, 0.2, 0.025, 0.7])
        cbar = plt.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1)).set_label(label=r'$M_{ecl}$',size=15)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        fig.savefig('stellarIMF_subplots_Meclcolorbar.pdf')
        
    def _IMF_plot(self, m_v, IMF_v, k, m_max, M, name : str, num_colors=1):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Msun = r'$M_{\odot}$'
        cm = plt.cm.get_cmap(name='plasma')
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.loglog(m_v,IMF_v, linewidth=3, color='darkblue')
        ax.set_ylabel(name, fontsize=15)
        ax.set_xlabel(f'Mass {Msun}', fontsize=15)
        plt.title(f'{M = :.2e} {Msun}', fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        fig.tight_layout()
        plt.savefig(f'IMF_plot_{name}.pdf', bbox_inches='tight')
        #plt.show(block=False)
        return None

    def Fig11_plot(self):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        CMOl = np.loadtxt('../data/Capuzzo-dolcetta17CMOl.csv', delimiter=',')
        CMOu = np.loadtxt('../data/Capuzzo-dolcetta17CMOu.csv', delimiter=',')
        SMBH = np.loadtxt('../data/Capuzzo-dolcetta17BH.csv', delimiter=',')
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        #ax.loglog(time, DTD_SNIa, color='blue', label='SNIa')
        #ax.legend(loc='best', frameon=False, fontsize=13)
        ax.scatter(CMOl[:,0], CMOl[:,1], color='red', marker='s', alpha=0.7)
        ax.scatter(CMOu[:,0], CMOu[:,1], color='magenta', marker='^', alpha=0.7)
        ax.scatter(SMBH[:,0], SMBH[:,1], color='black', marker='o', alpha=0.7)
        ax.set_ylabel(r'$\log_{10}(M_{\rm CMO}/M_{\odot})$', fontsize=15)
        ax.set_xlabel(r'$\log_{10}(M_{\rm pgal}/M_{\odot})$', fontsize=15)
        
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
        ax.tick_params(width=1, length=10, axis='x', which='minor', bottom=True, top=True, direction='in')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
        ax.tick_params(width=2, length=15, axis='x', which='major', bottom=True, top=True, direction='in')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
        ax.tick_params(width=1, length=10, axis='x', which='minor', bottom=True, top=True, direction='in')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
        ax.tick_params(width=2, length=15, axis='x', which='major', bottom=True, top=True, direction='in')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.tick_params(width=2)
        ax.set_ylim(0,11.5)
        ax.set_xlim(6, 13.7)
        fig.tight_layout()
        #plt.savefig('Fig11.pdf', bbox_inches='tight')
        
            
    def k_Z_plot(self, Z_massfrac_v, k_IMF_Z_list, m_max_Z_list, Mecl_v):
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cm = plt.cm.get_cmap(name='plasma')
        cm2 = plt.cm.get_cmap(name='plasma')
        num_colors = len(Z_massfrac_v)
        Z = [[0,0],[0,0]]
        #levels = np.linspace(np.log10(SFR_v[0]), np.log10(SFR_v[-1]), num_colors, endpoint=True)
        levels = np.linspace(np.log10(Z_massfrac_v[0]/0.0142), np.log10(Z_massfrac_v[-1]/0.0142), 100, endpoint=True)
        CS3 = plt.contourf(Z, levels, cmap=cm)
        plt.clf()
        SFR_colormap = (Z_massfrac_v)#np.log10(np.logspace(np.log10(SFR[0]), np.log10(SFR[-1]), 10, endpoint=True))
        currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
        currentColor = iter(currentColors)
        currentColors2 = [cm2(1.*i/num_colors) for i in range(num_colors)]
        currentColor2 = iter(currentColors2)
        Msun = r'$M_{\odot}$'
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax2 = ax.twinx()
        #ax.plot(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        #ax.scatter(np.log10(Z_massfrac_v)-0.0142, [alpha1_Z_list[i][0] for i in range(len(SFR_v))], linewidth=3, color='magenta')
        for i,Z in enumerate(Z_massfrac_v):
            color = next(currentColor)
            ax.plot(np.log10(Mecl_v), np.log10(k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.4)
            #ax.plot((Mecl_v), (k_IMF_Z_list[i]), linewidth=3, color=color, alpha=0.4)
            color2 = next(currentColor2)
            ax2.plot(np.log10(Mecl_v), m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
            #ax2.plot((Mecl_v), m_max_Z_list[i], linewidth=3, color=color2, alpha=0.4)
        ax.set_ylabel(r'$\log_{10}(k_{\rm IMF})$', fontsize=15)
        ax.set_xlabel(r'$\log_{10}(M_{\rm ecl})$[$M_{\odot}$]', fontsize=15)
        #ax.set_ylabel(r'$k_{\rm IMF}$', fontsize=15)
        #ax.set_xlabel(r'$M_{\rm ecl}$[$M_{\odot}$]', fontsize=15)
        ax2.set_ylabel(r'$m_{\rm max}$', fontsize=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad="100%")#, pack_start=True)
        #cax = divider.new_vertical(size="5%", pad=2.6, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(CS3, cmap=cm, cax=cax, format="%.2f", ticks=ticker.MultipleLocator(1),orientation="horizontal").set_label(label=r'$[Z]$',size=15)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        fig.tight_layout()
        plt.savefig(f'Mecl_vs_k_mmax.pdf', bbox_inches='tight')
        plt.show(block=False)


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
        #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
        plt.show(block=False)

    def get_lists(self):
        IMF_v_Z_list = []
        alpha1_Z_list = []
        alpha1_Z_list = []
        m_max_Z_list = []
        k_IMF_Z_list = []
        for Z in Z_massfrac_v:
            IMF_v_list = []
            alpha1_list = []
            alpha3_list = []
            m_max_list = []
            k_IMF_list = []
            for M in Mecl_v:
                igimf4 = IGIMF4.IGIMF(Z, downsizing_obj.SFR)
                sIMF = igimf4.stellar_IMF(M)
                print (f"M=%.2e,\t alpha1=%.2f,\t alpha2=%.2f,\t alpha3=%.2f,\t m_max = %.2e,\t [Z] = %.2f"%(M, sIMF[4], sIMF[5], sIMF[6], sIMF[1], igimf4.metallicity))
                #IMF_v = sIMF[2](mstar_v)
                alpha1_list.append(sIMF[4])
                alpha3_list.append(sIMF[6])
                m_max_list.append(sIMF[1])
                k_IMF_list.append(sIMF[0])
                IMF_v_list.append(IMF_v)
                #igimf4.ECMF_plot(Mecl_v, ECMF_v)
            IMF_v_Z_list.append(IMF_v_list)
            alpha1_Z_list.append(alpha1_list)
            alpha3_Z_list.append(alpha3_list)
            m_max_Z_list.append(m_max_list)
            k_IMF_Z_list.append(k_IMF_list)
        return k_IMF_Z_list, m_max_Z_list, IMF_v_Z_list, alpha1_Z_list, alpha1_Z_list
        
        
if __name__ == '__main__':
    main()

def main():
    directory_name = 'data'
    Mass_i = np.loadtxt(galcem_run_dir + directory_name + '/Mass_i.dat')
    phys = np.loadtxt(galcem_run_dir + directory_name + '/phys.dat')
    gal_time = phys[:,0]
    Mgas_v = phys[:,2]
    MZ = np.sum(Mass_i[4:, 2:], axis=0)
    SFR_v = phys[:,4]
    length_v = len(gal_time)
    
    k_ecl_v = np.zeros(length_v)
    k_star_v = np.zeros(length_v)
    M_ecl_max_v = np.zeros(length_v)
    
    for i,t in enumerate(gal_time):
        igimf_t = IGIMF(MZ(i), Mgas(i), t)
        k_star_v, k_ecl_v, M_ecl_max_v = igimf_t.solve()


