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
                 m_star_max = 150., m_star_min=0.08,  
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
        return (-0.14 * np.log10(np.divide(self.metal_mass_fraction, self.solar_metallicity)) 
                + 0.99 * np.log10(self.rho_cl(M_ecl)/1e6))

    def alpha_3_func(self, M_ecl):
        r"""Eq. (5)"""
        x_alpha_3 = self.x_alpha_3_func(M_ecl)
        if x_alpha_3 < -0.87:
            return 2.3
        else:
            return -0.41 * x_alpha_3 + 1.94
        
    def initial_mass_function(self, m, alpha_3=None, m_max=None):
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
        return util.get_norm(self.initial_mass_function, M_ecl, 
                             self.m_star_min, self.m_star_max, alpha_3=alpha_3)

    # embedded cluster mass functions
    def beta_func(self):
        r"""Eq. (11) """
        return -0.106 * np.log10(self.SFR) + 2
    
    def embedded_cluster_mass_function(self, M_ecl, m_max=None):
        r"""Eq. (8)"""
        if M_ecl>=self.M_ecl_min:
            return util.normalized(M_ecl, M_ecl**(-self.beta_func()), condition=m_max)
        else:
            return 0.
        
    def ECMF(self, *args, **kwargs):
        '''duplicate of stellar_IMF !!!!!!! '''
        return util.get_norm(self.embedded_cluster_mass_function, self.SFR * self.delta_t, 
                             self.M_ecl_min, self.M_ecl_max)
    
    def gwIMF_integrand_func(self, M_ecl, m, ECMF_func):
        k_star, m_max, IMF_func, IMF_weighted_func, alpha_3 = self.stellar_IMF(M_ecl)
        return IMF_func(m) * ECMF_func(M_ecl)
    
    # IGIMF functions
    def gwIMF(self, resolution=20):
        r"""Eq. (12)"""
        k_ecl, M_max, ECMF_func, ECM_weighted_func = self.ECMF()
        #return lambda m: integr.quad(self.gwIMF_integrand_func, self.M_ecl_min, M_max, args=(m, ECMF_func))[0]   
        return lambda m: integr.quadrature(igimf.gwIMF_integrand_func, igimf.M_ecl_min, M_max, args=(m, ECMF_func), vec_func=False, rtol=1e-5)[0]  
    

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
        
        
def IMF_3D_plot(m_v, M_ecl_v, sIMF_func):
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
    plt.savefig(f'IMF_plot_3D.pdf', bbox_inches='tight')
    
    
def IMF_plot(m_v, IMF_v, k, m_max, M, name : str, num_colors=1):
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

def Fig11_plot():
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
    plt.savefig('Fig11.pdf', bbox_inches='tight')