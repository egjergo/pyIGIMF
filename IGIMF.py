import util as u 
import numpy as np
from scipy import optimize
import scipy.integrate as integr
import pandas as pd

class IMF:
    #def __init__(self, mass_metals: float, mass_gas: float, 
    #             M_pgal: float, downsizing_time: float, t: float) -> None:
    def __init__(self) -> None:
        self.delta_t = 1e7 # [yr] duration of SF epoch
        self.solar_metallicity = 0.0142
        self.delta_alpha = 63 # (page 3)
        self.m_star_max = 150 # [Msun] stellar mass upper limit, Yan et al. (2017)
        self.m_star_min = 0.08 # [Msun]
        self.M_ecl_max = 1e10 # [Msun] most-massive ultra-compact-dwarf galaxy.
        self.M_ecl_min = 5 # [Msun] !!!! I've taken the lower limit from Eq. (8)
        self.mass_metals: float = 1e7 # [Msun]
        self.mass_gas: float = 1e9 # [Msun]
        self.M_pgal: float = 1e10 # [Msun]
        self.downsizing_time: float = 10 # [yr] 
        self.t: float = 1 # [Gyr]
        
        #self.M_pgal = M_pgal # [Msun]
        #self.downsizing_time = downsizing_time # [yr]
        
        #self.metal_mass_fraction = self.metal_mass_fraction_func(mass_metals, mass_gas)
        #self.SFR = 2 #self.SFR_func() # [Msun/yr] star_formation_rate
        #self.alpha_1 = self.alpha_1_func()
        #self.alpha_2 = self.alpha_2_func()
    
    def execute_normalization(self, IMF, M, lower_lim, upper_lim): 
        k, m_max = u.normalization(IMF, M, lower_lim, upper_lim)
        IMF_func = lambda m: k * IMF(m, m_max=m_max)
        return k, m_max, IMF_func
    

class Cluster_IMF(IMF):
    """
    Embedded cluster mass function (ECMF)
    """
    
    def __init__(self):
        super().__init__()
        
    def beta_func(self):
        """Eq. (11) """
        return -0.106 * np.log10(self.SFR) + 2

    def embedded_cluster_mass_function(self, M_ecl, m_max=None):
        r"""Eq. (8)"""
        func = M_ecl**(-self.beta_func())
        if M_ecl>=self.M_ecl_min:
            return u.normalized(M_ecl, func, condition=m_max)
        else:
            return 0.
        
    def ECMF(self):
        '''duplicate of stellar_IMF !!!!!!! '''
        k_ecl, M_max = u.normalization(self.embedded_cluster_mass_function,
                                          self.SFR * self.delta_t, self.M_ecl_min, self.M_ecl_max)
        ECMF_func = lambda M_ecl: k_ecl * self.embedded_cluster_mass_function(M_ecl, M_max=M_max)
        return k_ecl, M_max, ECMF_func
        
class Stellar_IMF(IMF):
    """
    Stellar IMF
    """
    def __init__(self, mass_metals, mass_gas, M_pgal, downsizing_time, t):
        super().__init__()
    

class IGIMF(IMF):
    """
    Integrated galaxy-wide IMF
    """
    def __init__(self, mass_metals, mass_gas, M_pgal, downsizing_time, t):
        super().__init__()


def main():
    return None

if __name__ == '__main__':
    main()