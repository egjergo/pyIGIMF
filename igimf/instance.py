import numpy as np
from igimf import utilities as util
#from igimf.optimal_sampling import OptimalSampling_Stellar, OptimalSampling_Clusters
from scipy import integrate as integr


class Parameters:
    '''
    Parameters employed in all the subclasses.
    The INPUT parameters define your galaxy/system properties at any given timestep.
    The DEFAULT parameters are known values from the literature
    
    INPUT
        metal_mass_fraction    [dimensionless] initial metallicity of the e.cl.
        SFR                    [Msun/yr] star formation rate
        
    DEFAULT PARAMETERS
        solar_metallicity    [dimensionless] M_Z_sun/M_sun (Asplund+09)
        metallicity          [dimensionless] [Z]
        delta_alpha          [dimensionless] (page 3, Yan et al. 2021)
        m_star_max           [Msun] stellar mass upper limit, Yan et al. (2017)
        m_star_min           [Msun] stellar mass lower limit, Yan et al. (2017)
        M_ecl_max            [Msun] most-massive ultra-compact-dwarf galaxy
        M_ecl_min            [Msun] I've taken the lower limit from Eq. (8)!!!!
        delta_t              [yr] duration of the SF epoch
        '''
    def __init__(self, SFR: float, metal_mass_fraction: float= None,
                solar_metallicity=0.0134, delta_alpha=63., delta_t=1e7,
                m_star_min=0.08, m_star_max = 150., 
                M_ecl_min=5., M_ecl_max = 1e10, suppress_warnings=False):
        vars = locals() 
        self.__dict__.update(vars)
        del self.__dict__["self"] 
        
        self.Mtot = self.SFR * self.delta_t 
        if metal_mass_fraction is not None:
            self.metallicity = np.log10(self.metal_mass_fraction/self.solar_metallicity)
        
        if suppress_warnings:
            import warnings
            warnings.filterwarnings('ignore')


class ECMF(Parameters):
    ''' Embedded Cluster Mass Function
    Only depends on SFR, but needs metal_mass_fraction to import Parameters,
    (this preserves a consistency when called inside IGIMF,
    but if an ECMF is needed by itself, pass it a dummy metal_mass_fraction)
    
    i.e., evaluate ECMF at time t where the galaxy is characterized by
    a SFR(t) and a Z(t) -- but only SFR(t) affects the ECMF.
    '''
    def __init__(self, SFR):
        super().__init__(SFR=SFR)
        self.beta_ECMF = self.beta_func()
        self.call_ECMF()

    def beta_func(self):
        r"""Eq. (11) ECMF slope"""
        return -0.106 * np.log10(self.SFR) + 2
    
    def embedded_cluster_MF(self, M_ecl, m_max=None):
        r"""Eq. (8) ECMF (not normalized)"""
        if M_ecl>=self.M_ecl_min:
            return util.normalization_check(M_ecl, M_ecl**(-self.beta_ECMF), 
                                   condition=m_max)
        else:
            return 0.
               
    def call_ECMF(self):
        '''ECMF (normalized)'''
        self.k_ecl, self.M_max = util.optimal_sampling_ECMF(
                        self.beta_ECMF, self.SFR * self.delta_t, 
                        self.M_ecl_min, self.M_ecl_max
                        )
        ECMF_func = lambda M_ecl: (self.k_ecl *
                                self.embedded_cluster_MF(M_ecl, m_max=self.M_max))
        ECMF_weighted_func = lambda M_ecl: util.mass_weighted_func(M_ecl, ECMF_func)
        self.ECMF_func = np.vectorize(ECMF_func)
        self.ECMF_weighted_func = np.vectorize(ECMF_weighted_func)
    

class StellarIMF_WIP(Parameters):
    ''' Computes Initial Mass Function for an Embedded cluster (e.cl.)
    at a given time t where the e.cl. is characterized by a SFR(t) and a Z(t)
    
    Depends on the total mass and metallicity of the e.cl., and implicitly
    (through the M_ecl) on the SFR.
    
    RETURNS
        all parameters that characterize the stellar IMF,
        the stellar IMF function,
        the mass-weighted stellar IMF function.
    '''
    def __init__(self, SFR: float, metal_mass_fraction: float, M_ecl:float,
                 sIMF_params:dict=None):
        
        super().__init__(metal_mass_fraction=metal_mass_fraction, SFR=SFR)
        
        if sIMF_params is None:
            sIMF_params = { # Kroupa (2001) default
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': self.m_star_min,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': self.m_star_max
                        }
        self.Kroupa01 = util.Kroupa01(sIMF_params=sIMF_params)
            
        self.M_ecl = M_ecl
        self.rho_cl = float(self.rho_cl_func())
        self.alpha_1 = float(self.alpha_1_func())
        self.alpha_2 = float(self.alpha_2_func())
        self.alpha_3 = self.alpha_3_func()
        self.IGIMF_params = {
                            'alpha1': self.alpha_1,
                            'alpha2': self.alpha_2,
                            'alpha3': self.alpha_3,
                            'Ml': self.m_star_min,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': self.m_star_max
        }
        a1, a2, a3 = util. IMF_normalization_constants(os_norm=1, norm_wrt=150, sIMF_params=self.IGIMF_params)
        self.IGIMF_params['a1'] = a1
        self.IGIMF_params['a2'] = a2
        self.IGIMF_params['a3'] = a3
        Mlim12 = self.IGIMF_params['Mlim12']
        Mlim23 = self.IGIMF_params['Mlim23']
        print(f'\nIf the power law is broken at 0.5 ({Mlim12=}) and 1 ({Mlim23=}) solar masses ({Mlim12==0.5} and {Mlim23==1.}),\n'+ 
              f'{a1/a2 = } should equal 2 ({a1/a2==2.}), and\n'+
              f'{a2/a3 = } should equal 1 ({a2/a3==1.})\n')
        
        self.call_stellar_IMF()
        
        
    def alpha_1_func(self):
        r"""Eq. (4) pt.1"""
        return (1.3 + self.delta_alpha * (self.metal_mass_fraction
                                          - self.solar_metallicity))
    
    def alpha_2_func(self):
        r"""Eq. (4) pt. 2 $\alpha_2 - \alpha_1 = 1$ always holds"""
        return 1 + self.alpha_1
        
    def rho_cl_func(self):
        r"""Eq. (7) core density of the molecular cloud 
        which forms the embedded star cluster
        In units of [Mstar/pc$^3$]
    
        For example, for M_ecl = 1000 Msun:
        >>> rho_cl(10**3)
        gives a core density of 4.79e4 Msun/pc$^3$"""
        return 10**(0.61 * np.log10(self.M_ecl) + 2.85)
    
    def _x_alpha_3_func(self):
        r"""Eq. (6)"""
        return (-0.14 * self.metallicity + 0.99 * np.log10(self.rho_cl/1e6))

    def alpha_3_func(self):
        r"""Eq. (5)"""
        x_alpha_3 = self._x_alpha_3_func()
        if x_alpha_3 < -0.87:
            return 2.3
        else:
            return -0.41 * x_alpha_3 + 1.94
        
    def initial_MF(self, m:float, m_max:float=None):
        r"""stellar IMF (with and without normalization)"""
        if m>=self.m_star_min:
            func = util.normalization_check(m, util.Kroupa01(sIMF_params=self.IGIMF_params), 
                                   condition=m_max)
            return func(m) if callable(func) else func
        else:
            return 0.
    
    def IMF(self, m_star_v, *args, **kwargs):
        IMF_func = np.vectorize(self.initial_MF, otypes=[float])
        return IMF_func(m_star_v)
        
    def initial_mass_function_OLD(self, m, alpha_3=None, m_max=None):
        '''stellar IMF (not normalized)'''
        if np.logical_and(m>=self.m_star_min, m<0.5):
            return m**(-self.alpha_1) * 2
        elif np.logical_and(m>=0.5, m<1.):
            return m**(-self.alpha_2)
        elif m>=1.:
            return util.normalized(m, m**(-alpha_3), condition=m_max)
        else:
            return 0.
              
    def call_stellar_IMF(self):
        '''ECMF (normalized)'''
        self.k_star, self.m_max = util.optimal_sampling_IMF(
                        #self.alpha_1, self.alpha_2, self.alpha_3, 
                        #self.m_star_min, self.m_star_max
                        self.M_ecl, self.IGIMF_params
                        )
        IMF_func = lambda m: (self.k_star * self.IMF(m, m_max=self.m_max))
        IMF_weighted_func = lambda m: util.mass_weighted_func(m, IMF_func)
        self.IMF_func = np.vectorize(IMF_func)
        self.IMF_weighted_func = np.vectorize(IMF_weighted_func)
        return 
        
    def stellar_IMF_OLD(self):
        r"""Eq. (1) Returns the normalized stellar IMF xi: 
        $\xi_* = d N_* / d m $"""
        self.k_star, self.m_max = util.normalization_IMF(self.alpha_1, 
                        self.alpha_2, self.alpha_3, self.M_ecl, 
                        self.m_star_min, self.m_star_max)
        self.IMF_func = lambda m: self.k_star * self.initial_mass_function(m, 
                                        alpha_3=self.alpha_3, m_max=self.m_max)
        self.IMF_weighted_func = lambda m: m * self.IMF_func(m)
    
    def eta_func(self, massive_threshold = 10):
        '''$\eta$ from the encyclopedia chapter'''
        min_mass=self.m_star_min
        max_mass=self.m_star_max
        nom = integr.quad(self.IMF_weighted_func, massive_threshold, max_mass)[0]
        den = integr.quad(self.IMF_weighted_func, min_mass, max_mass)[0]
        self.eta = np.divide(nom, den)
        
    def BD_func(self, mass, alpha_0=0.3, min_mass=0.001, max_mass=0.1):
        '''Brown dwarf power law'''
        unnormed_plaw = lambda M: np.power(M, -alpha_0)
        BD_integr = integr.quad(unnormed_plaw, min_mass, max_mass)[0]
        IMF_integr = integr.quad(self.IMF_func, self.m_star_min, self.m_star_max)[0]
        norm_factor = IMF_integr / (4.5 * BD_integr)
        return unnormed_plaw(mass) * norm_factor
        

class StellarIMF(Parameters):
    def __init__(self, SFR: float, metal_mass_fraction: float, M_ecl:float,
                 sIMF_params:dict=None):
        
        super().__init__(metal_mass_fraction=metal_mass_fraction, SFR=SFR)
        
        if sIMF_params is None:
            sIMF_params = { # Kroupa (2001) default
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': self.m_star_min,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': self.m_star_max
                        }
        self.Kroupa01 = util.Kroupa01(sIMF_params=sIMF_params)
            
        self.M_ecl = M_ecl
        self.rho_cl = float(self.rho_cl_func())
        self.alpha_1 = float(self.alpha_1_func())
        self.alpha_2 = float(self.alpha_2_func())
        self.alpha_3 = self.alpha_3_func()
        self.IGIMF_params = {
                            'alpha1': self.alpha_1,
                            'alpha2': self.alpha_2,
                            'alpha3': self.alpha_3,
                            'Ml': self.m_star_min,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': self.m_star_max
        }
        a1, a2, a3 = util. IMF_normalization_constants(os_norm=1, norm_wrt=150, sIMF_params=self.IGIMF_params)
        self.IGIMF_params['a1'] = a1
        self.IGIMF_params['a2'] = a2
        self.IGIMF_params['a3'] = a3
        Mlim12 = self.IGIMF_params['Mlim12']
        Mlim23 = self.IGIMF_params['Mlim23']
        print(f'\nIf the power law is broken at 0.5 ({Mlim12=}) and 1 ({Mlim23=}) solar masses ({Mlim12==0.5} and {Mlim23==1.}),\n'+ 
              f'{a1/a2 = } should equal 2 ({a1/a2==2.}), and\n'+
              f'{a2/a3 = } should equal 1 ({a2/a3==1.})\n')
        
        self.call_stellar_IMF()
        

    def alpha_1_func(self):
        r"""Eq. (4) pt.1"""
        # 1.3 + 63 * (1e7/1e9- 0.0142)
        return 1.3 + self.delta_alpha * (self.metal_mass_fraction - self.solar_metallicity)
    
    def alpha_2_func(self):
        r"""Eq. (4) pt. 2 $\alpha_2 - \alpha_1 = 1$ always holds"""
        return 1 + self.alpha_1
        
    def rho_cl_func(self):
        r"""Eq. (7) core density of the molecular cloud 
        which forms the embedded star cluster
        In units of [Mstar/pc^3]
    
        For example, for M_ecl = 1000 Msun:
        >>> rho_cl(10**3)
        gives a core density of 4.79e4 Msun/pc^3"""
        return 10**(0.61 * np.log10(self.M_ecl) + 2.85)
    
    def x_alpha_3_func(self):
        r"""Eq. (6)"""
        return (-0.14 * self.metallicity + 0.99 * np.log10(self.rho_cl/1e6))

    def alpha_3_func(self):
        r"""Eq. (5)"""
        x_alpha_3 = self.x_alpha_3_func()
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
        
    def call_stellar_IMF(self):
        r"""Eq. (1) Returns the stellar IMF xi: $\xi_* = d N_* / d m $"""
        self.k_star, self.m_max = util.normalization_IMF(self.alpha_1, self.alpha_2, self.alpha_3, self.M_ecl, 
                                           self.m_star_min, self.m_star_max)
        IMF_func = lambda m: self.k_star * self.initial_mass_function(m, alpha_3=self.alpha_3, m_max=self.m_max)
        IMF_weighted_func = lambda m: util.mass_weighted_func(m, IMF_func)
        self.IMF_func = np.vectorize(IMF_func)
        self.IMF_weighted_func = np.vectorize(IMF_weighted_func)
        return 

    def BD_func(self, mass, alpha_0=0.3, min_mass=0.001, max_mass=0.1):
        '''Brown dwarf power law'''
        unnormed_plaw = lambda M: np.power(M, -alpha_0)
        BD_integr = integr.quad(unnormed_plaw, min_mass, max_mass)[0]
        IMF_integr = integr.quad(self.IMF_func, self.m_star_min, self.m_star_max)[0]
        norm_factor = IMF_integr / (4.5 * BD_integr)
        return unnormed_plaw(mass) * norm_factor