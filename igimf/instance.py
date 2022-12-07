"""
Based on Gjergo, Kroupa, Yan, Sorokin, Jerabkova, and Zhang (2023, ApJS)

The Parameters superclass is parent to "ECMF", "StellarIMF", and "IGIMF".
    |... It contains the input parameters shared by all subclasses.

The ECMF subclass is structured as follows:
    |-- ECMF (normalized)
    |-- |-- embedded_cluster_MF (not normalized)
    |-- |-- |-- beta

The StellarIMF subclass is structured as follows:
    |-- stellar_IMF (normalized)
    |-- |-- initial_mass_function (not normalized)
    |-- |-- |-- alpha_1_func
    |-- |-- |-- alpha_2_func
    |-- |-- |-- alpha_3_func
    |-- |-- |-- |-- rho_cl
    |-- |-- |-- |-- x_alpha_3

The IGIMF subclass:
|-- gwIMF_integrand_func, the integrand
|-- gwIMF (output)
"""
import numpy as np
from scipy import optimize
import scipy.integrate as integr
import pandas as pd
import igimf.util as util
#from ..igimf import util

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
                solar_metallicity=0.0134, delta_alpha=63,
                m_star_max = 150, m_star_min=0.07, suppress_warnings=True,
                M_ecl_max = 1e10, M_ecl_min=5, delta_t=1e7):
        self.solar_metallicity = solar_metallicity
        self.delta_alpha = delta_alpha
        self.m_star_max = m_star_max
        self.m_star_min = m_star_min 
        self.M_ecl_max = M_ecl_max
        self.M_ecl_min = M_ecl_min 
        self.delta_t = delta_t
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
    '''
    def __init__(self, SFR, metal_mass_fraction=None):
        if metal_mass_fraction is None:
            metal_mass_fraction = 0.1 * 0.0134
        super().__init__(metal_mass_fraction, SFR)
        self.beta_ECMF = self.beta_func()
        self.call_ECMF()

    def beta_func(self):
        r"""Eq. (11) ECMF slope"""
        return -0.106 * np.log10(self.SFR) + 2
    
    def embedded_cluster_MF(self, M_ecl, m_max=None):
        r"""Eq. (8) ECMF (not normalized)"""
        if M_ecl>=self.M_ecl_min:
            return util.normalized(M_ecl, M_ecl**(-self.beta_ECMF), 
                                   condition=m_max)
        else:
            return 0.
               
    def call_ECMF(self):
        '''ECMF (normalized)'''
        k_ecl, M_max = util.normalization_ECMF(self.embedded_cluster_MF,
                        self.beta_ECMF, self.SFR * self.delta_t, 
                        self.M_ecl_min, self.M_ecl_max)
        self.k_ecl = k_ecl
        self.M_max = M_max
        ECMF_func = lambda M_ecl: (k_ecl *
                                self.embedded_cluster_MF(M_ecl, m_max=M_max))
        ECMF_weighted_func = lambda M_ecl: util.weighted_func(M_ecl, ECMF_func)
        self.ECMF_func = np.vectorize(ECMF_func)
        self.ECMF_weighted_func = np.vectorize(ECMF_weighted_func)
    

class StellarIMF(Parameters):
    ''' Computes Initial Mass Function for an Embedded cluster (e.cl.)
    at a given time t where the e.cl. is characterized by a SFR(t) and a Z(t)
    
    Depends on the total mass and metallicity of the e.cl., and implicitly
    (through the M_ecl) on the SFR.
    
    RETURNS
        all parameters that characterize the stellar IMF,
        the stellar IMF function,
        the mass-weighted stellar IMF function.
    '''
    def __init__(self, M_ecl:float, metal_mass_fraction: float, SFR: float):
        super().__init__(metal_mass_fraction, SFR)
        self.M_ecl = M_ecl
        self.alpha_1 = float(self.alpha_1_func())
        self.alpha_2 = float(self.alpha_2_func())
        self.alpha_3 = self.alpha_3_func()
        self.stellar_IMF()
        self.IMF_func = np.vectorize(self.IMF_func)
        self.IMF_weighted_func = np.vectorize(self.IMF_weighted_func)

    def alpha_1_func(self):
        r"""Eq. (4) pt.1"""
        return (1.3 + self.delta_alpha * (self.metal_mass_fraction
                                          - self.solar_metallicity))
    
    def alpha_2_func(self):
        r"""Eq. (4) pt. 2 $\alpha_2 - \alpha_1 = 1$ always holds"""
        return 1 + self.alpha_1
        
    def rho_cl(self):
        r"""Eq. (7) core density of the molecular cloud 
        which forms the embedded star cluster
        In units of [Mstar/pc$^3$]
    
        For example, for M_ecl = 1000 Msun:
        >>> rho_cl(10**3)
        gives a core density of 4.79e4 Msun/pc$^3$"""
        return 10**(0.61 * np.log10(self.M_ecl) + 2.85)
    
    def x_alpha_3_func(self):
        r"""Eq. (6)"""
        return (-0.14 * self.metallicity + 0.99 * np.log10(self.rho_cl()/1e6))

    def alpha_3_func(self):
        r"""Eq. (5)"""
        x_alpha_3 = self.x_alpha_3_func()
        if x_alpha_3 < -0.87:
            return 2.3
        else:
            return -0.41 * x_alpha_3 + 1.94
        
    def initial_mass_function(self, m, alpha_3=None, m_max=None):
        '''stellar IMF (not normalized)'''
        if np.logical_and(m>=self.m_star_min, m<0.5):
            return m**(-self.alpha_1) * 2
        elif np.logical_and(m>=0.5, m<1.):
            return m**(-self.alpha_2)
        elif m>=1.:
            return util.normalized(m, m**(-alpha_3), condition=m_max)
        else:
            return 0.
        
    def stellar_IMF(self):
        r"""Eq. (1) Returns the normalized stellar IMF xi: 
        $\xi_* = d N_* / d m $"""
        self.k_star, self.m_max = util.normalization_IMF(self.alpha_1, 
                        self.alpha_2, self.alpha_3, self.M_ecl, 
                        self.m_star_min, self.m_star_max)
        self.IMF_func = lambda m: self.k_star * self.initial_mass_function(m, 
                                        alpha_3=self.alpha_3, m_max=self.m_max)
        self.IMF_weighted_func = lambda m: util.weighted_func(m, self.IMF_func)


class IGIMF(Parameters):
    ''' Computes the Integrated Galaxy-wide Initial Mass Function of stars
    at a given time t, where the galaxy is characterized by 
    
    a metal mass fraction Z(t) = M_Z/M_gas, and
    a star formation rate SFR(t) [Msun/yr]
    
    RETURNS
        IGIMF    (function(mstar)) [#/Msun] the galaxy-wide IMF function
                 from the IGIMF formalism. The IGIMF function accepts
                 one variable, the mstar (either float or array)
    '''
    def __init__(self, metal_mass_fraction: float, SFR: float, 
                 suppress_warnings=True) -> None:
        super().__init__(metal_mass_fraction, SFR)
        self.IGIMF_func = np.vectorize(self.gwIMF())

    def gwIMF_integrand_func(self, M_ecl, ECMF_func, m):
        stellar_IMF = StellarIMF(M_ecl, self.metal_mass_fraction, self.SFR)
        return stellar_IMF.IMF_func(m) * ECMF_func(M_ecl)
    
    def gwIMF(self):
        r"""Eq. (12)"""
        ecl_MF = ECMF(self.SFR, metal_mass_fraction=self.metal_mass_fraction)
        return lambda m: integr.quadrature(self.gwIMF_integrand_func, 
            self.M_ecl_min, ecl_MF.M_max, args=(ecl_MF.ECMF_func, m), 
            vec_func=False, rtol=1e-15)[0]  
    
def main():
    import json
    massfrac = float(input("What is the metal mass fraction?") or .1*0.0134)
    SFR = float(input(r"What is the SFR (in $M_{\odot}$/yr)?") or 1)
    Mtot = float(input('''What is the mass of the embedded cluster?
    Note: this computes a single instance of the stellar IMF''') or 1e10)
    stellar_IMF = StellarIMF(Mtot, massfrac, SFR)
    ecl_MF = ECMF(SFR,metal_mass_fraction=massfrac)
    print(f'\nAt a time t characterized by:\n{SFR=:.2e}\nand \n{Mtot=:.2e}\n')
    print(f"\ndictionary of an e.cl.'s stellar IMF\n")
    print(json.dumps(stellar_IMF.__dict__, default=str, indent=4))
    print(f"\n\ndictionary of a galaxy's ECMF\n")
    print(json.dumps(ecl_MF.__dict__, default=str, indent=4))
    
    t_IGIMF = IGIMF(massfrac, SFR)
    print(f"\n\ndictionary of a galaxy's IGIMF at a given time t\n")
    print(json.dumps(t_IGIMF.__dict__, default=str, indent=4))
    return massfrac, SFR, Mtot, stellar_IMF, ecl_MF, t_IGIMF
        
if __name__ == '__main__':
    massfrac, SFR, Mtot, stellar_IMF, ecl_MF, t_IGIMF = main()