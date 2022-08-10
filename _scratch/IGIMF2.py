import numpy as np
from scipy import optimize
import scipy.integrate as integr
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d

class IGIMF:
    ''' Computes the Integrated Galaxy-wide Initial Mass Function of stars'''
    
    #def __init__(self, mass_metals, mass_gas, star_formation_rate, t):
    def __init__(self, mass_metals: float, mass_gas: float, 
                 M_pgal: float, downsizing_time: float, t: float) -> None:
        self.delta_t = 1e7 # [yr] duration of SF epoch
        self.solar_metallicity = 0.0142
        self.delta_alpha = 63 # (page 3)
        self.m_star_max = 150 # [Msun] stellar mass upper limit, Yan et al. (2017)
        self.m_star_min = 0.08 # [Msun]
        self.M_ecl_max = 1e10 # [Msun] most-massive ultra-compact-dwarf galaxy.
        self.M_ecl_min = 5 # [Msun] !!!! I've taken the lower limit from Eq. (8)
        #self.M_pgal = M_pgal # [Msun]
        #self.downsizing_time = downsizing_time # [yr]
        self.t = t
        self.SFR = 2 # !!!!!!!
        
    def IGIMF(self, t):
        metallicity = 0.001 # !!!!!!!
        Mtot = self.SFR * self.delta_t  # !!!!!!!
        
        def integrand(M_ecl, m):
            k_ecl, M_max, ECMF_func = self.ECMF(M_ecl, Mtot)
            
            return ECMF_func(M_ecl) #self.stellar_IMF(M_ecl, m, metallicity) * 
        
        return lambda m: integr.quad(integrand, self.M_ecl_min, self.M_ecl_max, args=(m))[0]
    
    def ECMF(self, M_ecl, Mtot):
        
        def beta_func():
            r"""Eq. (11) """
            return -0.106 * np.log10(self.SFR) + 2
        
        def embedded_cluster_mass_function(M_ecl, M_max=None):
            r"""Eq. (8)"""
            func = lambda M_ecl: M_ecl**(-beta_func())
            if M_ecl>=self.M_ecl_min:
                return self._normalized(M_ecl, func, condition=M_max)
            else:
                return 0.
            
        k_ecl, M_max = self._normalization(embedded_cluster_mass_function, 
                                          Mtot, self.M_ecl_min, self.M_ecl_max)
        ECMF_func = lambda M_ecl: k_ecl * embedded_cluster_mass_function(M_ecl, M_max=M_max)
        return k_ecl, M_max, ECMF_func
    
    #def stellar_IMF(self):
    #    return
    
    def _normalized(self, x, func, condition=None):
        ''' IMF behavior depending on whether or not it has been normalized '''
        if condition:
            if x <= condition:
                return func
            else:
                return 0.
        else:
            return func
        
    def _normalization(self, IMF, M, lower_lim, upper_lim, **kwargs) -> (float, float):
        r"""normalize the IMFs"""
        
        #IMF_ = lambda m: IMF(m)
        k = lambda x: np.reciprocal(integr.quad(IMF, x, upper_lim)[0])
        weighted_IMF = lambda m,x: np.multiply(m, IMF(m)) * k(x)
        
        func = lambda x: integr.quad(weighted_IMF, lower_lim, x, args=(x,))[0] - M
        sol = optimize.root_scalar(func, bracket=[lower_lim, upper_lim])
        m_max = sol.root
        return k(m_max), m_max