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

class IGIMF:
    ''' Computes the Integrated Galaxy-wide Initial Mass Function of stars'''

    #def __init__(self, mass_metals, mass_gas, star_formation_rate, t):
    def __init__(self, mass_metals: float, mass_gas: float, 
                 M_igal: float, suppress_warnings=True) -> None: #, downsizing_time: float, t: float) -> None:
        self.delta_t = 1e7 # [yr] duration of SF epoch
        self.solar_metallicity = 0.0142
        self.delta_alpha = 63 # (page 3)
        self.m_star_max = 150. # [Msun] stellar mass upper limit, Yan et al. (2017)
        self.m_star_min = 0.08 # [Msun]
        self.M_ecl_max = 1e10 # [Msun] most-massive ultra-compact-dwarf galaxy.
        self.M_ecl_min = 5 # [Msun] !!!! I've taken the lower limit from Eq. (8)
        self.M_igal = M_igal # [Msun]
        self.downsizing_time = self.delta_tau(M_igal) # [yr]
        
        self.metal_mass_fraction = self.metal_mass_fraction_func(mass_metals, mass_gas)
        self.SFR = self.SFR_func() #2 # [Msun/yr] star_formation_rate
        self.Mtot = self.SFR * self.delta_t
        self.alpha_1 = float(self.alpha_1_func())
        self.alpha_2 = float(self.alpha_2_func())
        #self.alpha_3 = self.alpha_3_func(M_ecl)
        if suppress_warnings:
            import warnings
            warnings.filterwarnings('ignore')
        
    def weighted_func(self, M, func):
        return np.multiply(M, func(M))

    def normalized(self, x, func, condition=None):
        ''' IMF behavior depending on whether or not it has been normalized '''
        if condition:
            if x <= condition:
                return func
            else:
                return 0.
        else:
            return func
        
    def normalization(self, IMF, M, lower_lim, upper_lim, *args, **kwargs) -> (float, float):
        r'''
        Function that extracts k and m_max (blue boxes in the notes)
        IMF:    mass distribution function, i.e. either Eq. (1) or (8)
        M:      value of the integrals in Eq. (2) and (9)
        k:      normalization of the IMF function, i.e. Eq. (3) and (10)
        upper_lim and lower_lim:    
                global upper and lower limits on the integrals
        guess:  guess on the local upper (lower) limit on the integrals 
                of Eq.2 and Eq.9 (of Eq.3 and Eq.10)
        x:      evaluated local upper (lower) limit on the integrals
                of Eq.2 and Eq.9 (of Eq.3 and Eq.10)
        **kwargs    optional keyword arguments
        
        -----
        
        .. math::
        `M = \int_{\mathrm{lower_lim}}^{\mathrm{m_max}}
        m \, \mathrm{IMF}(m,...)\,\mathrm{d}m`
        
        .. math::
        `1 = \int_{\mathrm{m_max}}^{{\rm upper_lim}}{\mathrm{IMF}(m,...)} \,\mathrm{d}m`
        '''
        k = lambda x: np.reciprocal(integr.quad(IMF, x, upper_lim, args=(args),
                                     epsrel=1e-8, limit=int(1e3), maxp1=int(1e3), limlst=int(1e3))[0])
        def weighted_IMF(m, x, *args):
            return m * IMF(m, *args) * k(x)
        func = lambda x: (integr.quad(weighted_IMF, lower_lim, x, args=(x, *args), 
                                     epsrel=1e-8, limit=int(1e3), maxp1=int(1e3), limlst=int(1e3))[0] - M)
        sol = optimize.root_scalar(func, bracket=[lower_lim, upper_lim], rtol=1e-8)
        m_max = sol.root
        return k(m_max), m_max
    
    # Galaxy functions 
    def delta_tau(self, M_igal):
        '''
        Returns delta tau in Gyr for the downsizing relation as it is expressed in Recchi+09
        
        M_igal is expressed in Msun and ranges from 1e6 to 1e12
        '''
        return 8.16 * np.e**(-0.556 * np.log10(M_igal) + 3.401) + 0.027
        
    def SFR_func(self):
        '''SFR [Msun/yr] assuming the downsizing time (Thomas et al., 2005)'''
        return np.divide(self.M_igal, self.downsizing_time * 1e9)
    
    def metal_mass_fraction_func(self, mass_metals, mass_gas):
        r"""$Z$
        Metallicity defined as the mass fraction between metals and hydrogen"""
        return np.divide(mass_metals, mass_gas) #0.0142 

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
            return self.normalized(m, m**(-alpha_3), condition=m_max)
        else:
            return 0.
    
    def stellar_IMF(self, M_ecl, ECMF_weight=None):
        r"""Eq. (1) Returns the stellar IMF xi: $\xi_* = d N_* / d m $"""
        alpha_3 = self.alpha_3_func(M_ecl)
        k_star, m_max = self.normalization(self.initial_mass_function, M_ecl, 
                                           self.m_star_min, self.m_star_max, alpha_3)
        IMF_weighted_func = lambda m: k_star * self.initial_mass_function(m, alpha_3=alpha_3, m_max=m_max) * ECMF_weight
        IMF_func = lambda m: k_star * self.initial_mass_function(m, alpha_3=alpha_3, m_max=m_max)
        return k_star, m_max, IMF_func, IMF_weighted_func, alpha_3

    # embedded cluster mass functions
    def beta_func(self):
        r"""Eq. (11) """
        return -0.106 * np.log10(self.SFR) + 2
    
    def embedded_cluster_mass_function(self, M_ecl, M_max=None):
        r"""Eq. (8)"""
        if M_ecl>=self.M_ecl_min:
            return self.normalized(M_ecl, M_ecl**(-self.beta_func()), condition=M_max)
        else:
            return 0.
        
    def ECMF(self):
        '''duplicate of stellar_IMF !!!!!!! '''
        k_ecl, M_max = self.normalization(self.embedded_cluster_mass_function,
                                          self.SFR * self.delta_t, self.M_ecl_min, self.M_ecl_max)
        ECMF_func = lambda M_ecl: k_ecl * self.embedded_cluster_mass_function(M_ecl, M_max=M_max)
        ECMF_weighted_func = lambda M_ecl: self.weighted_func(M_ecl, ECMF_func)
        #print(f"The relative error is {np.divide(np.abs(integr.quad(ECMF_weighted_func, self.M_ecl_min, self.M_ecl_max)[0] - self.Mtot), self.Mtot):.3e}")
        return k_ecl, M_max, ECMF_func, ECMF_weighted_func

    # IGIMF functions
    def gwIMF(self, resolution=50):
        r"""Eq. (12)"""
        k_ecl, M_max, ECMF_func, ECM_weighted_func = self.ECMF()
        M_ecl_v = np.logspace(np.log10(self.M_ecl_min), np.log10(M_max), num=resolution)
        ECMF_v = np.array([ECMF_func(M) for M in M_ecl_v])
        gwIMF_df = pd.DataFrame([self.stellar_IMF(M_ecl_v[i], ECMF_weight=ECMF_v[i])  for i in range(len(M_ecl_v))], columns=['kstar', 'm_max', 'func', 'func_weighted', 'alpha_3'])
        gwIMF_df['ECMF_v'] = ECMF_v
        sIMF_func = gwIMF_df['func'].to_numpy()
        sIMF_v = np.array([sIMF_func[i](ECMF_v[i]) for i in range(len(ECMF_v))])
        ECintegr = integr.simpson(np.multiply(ECMF_v, sIMF_v), x=M_ecl_v)   
        return ECintegr#, M_pgal
    
    #def normalization_test(self):
    #    [integr.simpson(xi[i],x=m_v) for i,_ in enumerate(M_ecl_v)]
    #    return None

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