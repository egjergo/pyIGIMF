import numpy as np
from scipy import optimize
import scipy.integrate as integr
import pandas as pd
import igimf.util as util


class Parameters:
    '''
    Parameters employed in all the subclasses
    
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
    def __init__(self, metal_mass_fraction: float, SFR: float,
                solar_metallicity=0.0134, delta_alpha=63.,
                m_star_max = 150.1, m_star_min=0.07, suppress_warnings=True,
                M_ecl_max = 1e10, M_ecl_min=5., delta_t=1e7):
        vars = locals() 
        self.__dict__.update(vars)
        del self.__dict__["self"] 
        self.SFR = SFR 
        self.Mtot = self.SFR * self.delta_t 
        self.metal_mass_fraction = metal_mass_fraction
        self.metallicity = np.log10(self.metal_mass_fraction
                                    / self.solar_metallicity)
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
    
    ECMF_normalized() defines:
        - k_ecl
        - M_max
        - ECMF_func
        - ECMF_weighted_func
    '''
    def __init__(self, metal_mass_fraction: float, SFR: float, *args, **kwargs):
        super().__init__(metal_mass_fraction, SFR)
        self.beta_ECMF = self.beta_func()
        self.ECMF_normalized()

    def beta_func(self):
        r"""Eq. (11) ECMF slope"""
        return -0.106 * np.log10(self.SFR) + 2
    
    def ECMF_not_normalized(self, M_ecl, M_max=None):
        r"""Eq. (8) ECMF (not normalized)"""
        if M_ecl >= self.M_ecl_min:
            return util.impose_upper_lim(M_ecl, 
                        M_ecl**(-self.beta_ECMF), condition=M_max)
        else:
            return 0.
               
    def ECMF_normalized(self):
        '''ECMF (normalized)'''
        k_ecl, M_max = util.normalization_ECMF(self.ECMF_not_normalized,
                        self.beta_ECMF, self.SFR * self.delta_t, 
                        self.M_ecl_min, self.M_ecl_max)
        self.k_ecl = k_ecl
        self.M_max = M_max
        ECMF_func = lambda M_ecl: (k_ecl *
                                self.ECMF_not_normalized(M_ecl, M_max=M_max))
        ECMF_weighted_func = lambda M_ecl: util.weighted_func(M_ecl, ECMF_func)
        self.ECMF_func = np.vectorize(ECMF_func)
        self.ECMF_weighted_func = np.vectorize(ECMF_weighted_func)
    