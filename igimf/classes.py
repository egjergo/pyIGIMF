import numpy as np
from igimf import utilities as util
from scipy import integrate as integr    
from functools import lru_cache
from numpy.polynomial.legendre import leggauss


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
    def __init__(self, SFR: float=None, metal_mass_fraction: float= None,
                solar_metallicity=0.0142, delta_alpha=63., delta_t=1e7,
                m_star_min=0.08, m_star_max = 150., #alpha1slope='logistic',
                M_ecl_min=5., M_ecl_max = 1e10, suppress_warnings=False):
        vars = locals() 
        #print(f'In Parameters class {metal_mass_fraction=}')
        self.__dict__.update(vars)
        del self.__dict__["self"] 
        #print(f'In Parameters class {self.metal_mass_fraction=}')
        
        if SFR is not None:
            self.Mtot = self.SFR * self.delta_t 
        if metal_mass_fraction is not None:
            self.metallicity = np.log10(self.metal_mass_fraction/self.solar_metallicity)
        
        if suppress_warnings:
            import warnings
            warnings.filterwarnings('ignore')


class ECMF(Parameters):
    ''' Embedded Cluster Mass Function

    INPUT
        SFR                    [Msun/yr] star formation rate
        
    GENERATES
        beta_ECMF              [dimensionless] ECMF slope
        k_ecl                  [dimensionless] normalization constant
        M_max                  [Msun] most-massive ultra-compact-dwarf galaxy
        ECMF_func              [#/Msun^-1] normalized ECMF: dN/dM_ecl
        ECMF_weighted_func     [dimensionless] mass-weighted ECMF_func: dM/dM_ecl
    '''
    def __init__(self, SFR: float = None):
        super().__init__(SFR=SFR)
        self.beta_ECMF = self.beta_func()
        self.call_ECMF()

    def beta_func(self):
        r"""Gjergo+2025 Eq. (11) ECMF slope"""
        return -0.106 * np.log10(self.SFR) + 2
    
    def embedded_cluster_MF(self, M_ecl, m_max=None):
        r"""Eq. (8) ECMF (not normalized)"""
        if M_ecl>=self.M_ecl_min:
            return util.normalization_check(M_ecl, M_ecl**(-self.beta_ECMF), 
                                   condition=m_max)
        else:
            return 0.
               
    def call_ECMF(self):
        '''
        Returns the Embedded Cluster Mass Function
        
        GENERATES
            k_ecl                  [dimensionless] normalization constant
            M_max                  [Msun] most-massive ultra-compact-dwarf galaxy
            ECMF_func              [#/Msun^-1] normalized ECMF: dN/dM_ecl
            ECMF_weighted_func     [dimensionless] mass-weighted ECMF_func: dM/dM_ecl
        '''
        self.k_ecl, self.M_max = util.optimal_sampling_ECMF(
                        self.beta_ECMF, self.SFR * self.delta_t, 
                        self.M_ecl_min, self.M_ecl_max
                        )
        ECMF_func = lambda M_ecl: (self.k_ecl *
                                self.embedded_cluster_MF(M_ecl, m_max=self.M_max))
        ECMF_weighted_func = lambda M_ecl: util.mass_weighted_func(M_ecl, ECMF_func)
        self.ECMF_func = np.vectorize(ECMF_func)
        self.ECMF_weighted_func = np.vectorize(ECMF_weighted_func)
    

class StellarIMF(Parameters):
    ''' Computes Initial Mass Function for an Embedded cluster (e.cl.)
    at a given time t where the e.cl. is characterized by a SFR(t) and a Z(t)
    
    Depends on the total mass and average metallicity of the e.cl., and implicitly
    (through the M_ecl) on the SFR.

    INPUT
        SFR                    [Msun/yr] star formation rate
        metal_mass_fraction    [dimensionless] initial metallicity of the e.cl.
        sIMF_params            [dict] parameters of the IMF
        alpha1slope            [str] parameters of the IMF
    '''
    def __init__(self, M_ecl:float=None, SFR: float=None, metal_mass_fraction: float=None, 
                 sIMF_params:dict=None, alpha1slope:str=None):
        #print('Creating a new StellarIMF instance')
        #print(f'In StellarIMF class {SFR=}')
        #print(f'In StellarIMF class {metal_mass_fraction=}')
        super().__init__(metal_mass_fraction=metal_mass_fraction, SFR=SFR)
        self.alpha1slope = alpha1slope
        #print('Imported Parameters parent class')
        #print(f'In StellarIMF class {self.SFR=}')
        #print(f'In StellarIMF class {self.metal_mass_fraction=}')
        #print(f'In StellarIMF class {self.metallicity=}')
        if sIMF_params is None:
            sIMF_params = { # Kroupa (2001) default
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': self.m_star_min,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': self.m_star_max,
                            'M_ecl': M_ecl,
                            'metal_mass_fraction': metal_mass_fraction,
                            
                        }
        self.Kroupa01 = util.Kroupa01(sIMF_params=sIMF_params)
            
        self.M_ecl = M_ecl
        self.rho_cl = float(self.rho_cl_func())
        if self.alpha1slope == 'linear':
            self.alpha_1 = float(self.alpha_1_func())
        elif self.alpha1slope == 'logistic':
            self.alpha_1 = float(self.alpha_1_func_logistic())
        else:
            ValueError('alpha1slope incorrectly defined')
        self.alpha_2 = float(self.alpha_2_func())
        self.alpha_3 = self.alpha_3_func()
        self.IGIMF_params = {
                            'alpha1': self.alpha_1,
                            'alpha2': self.alpha_2,
                            'alpha3': self.alpha_3,
                            'Ml': self.m_star_min,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': self.m_star_max,
                            'M_ecl': M_ecl,
                            'metal_mass_fraction': metal_mass_fraction
        }
        a1, a2, a3 = util.IMF_normalization_constants(os_norm=1, norm_wrt=150, sIMF_params=self.IGIMF_params)
        self.IGIMF_params['a1'] = a1
        self.IGIMF_params['a2'] = a2
        self.IGIMF_params['a3'] = a3
        
        self.call_stellar_IMF()
        
    def alpha_1_func_logistic(self, maximum=None, Z_midpoint=None, growth_rate=None):
        r"""Gjergo+2025 Eq. (15) logistic alpha1(Z)"""
        if maximum is None: maximum = 1.3*2
        if Z_midpoint is None: Z_midpoint = self.solar_metallicity
        if growth_rate is None: growth_rate = 2/self.solar_metallicity
        Z = self.metal_mass_fraction
        return np.divide(maximum, 1 + np.exp(-np.multiply(growth_rate,Z-Z_midpoint)))
    
    def alpha_1_func(self):
        r"""Gjergo+2025 Eq. (5) linear alpha1(Z)"""
        return (1.3 + self.delta_alpha * (self.metal_mass_fraction
                                          - self.solar_metallicity))
        
    def alpha_1_func_Y24(self):
        '''Yan+2024
        https://ui.adsabs.harvard.edu/abs/2024ApJ...969...95Y/abstract'''
        return 79.4 * (self.metal_mass_fraction - np.power(10,-0.1)*self.solar_metallicity)
    
    def alpha_2_func(self):
        r"""Gjergo+2025 Eq. (5) 
        $\alpha_2 - \alpha_1 = 1$ always holds"""
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
        r"""Gjergo+2025 Eq. (7) alpha3 dependence on density and metallicity"""
        return (-0.14 * self.metallicity + 0.99 * np.log10(self.rho_cl/1e6))

    def alpha_3_func(self):
        r"""Gjergo+2025 Eq. (6) alpha3 slope"""
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
    
    def call_stellar_IMF(self):
        '''ECMF (normalized)'''
        self.k_star, self.m_max = util.optimal_sampling_IMF(
                        self.M_ecl, self.IGIMF_params
                        )
        IMF_func = lambda m: (self.k_star * self.IMF(m, m_max=self.m_max))
        IMF_weighted_func = lambda m: util.mass_weighted_func(m, IMF_func)
        self.IMF_func = np.vectorize(IMF_func)
        self.IMF_weighted_func = np.vectorize(IMF_weighted_func)
        return 
    
    def eta_func(self, massive_threshold = 10):
        '''eta from the encyclopedia chapter 
        (Kroupa, Gjergo, Jerankova & Yan, 2025)
        https://ui.adsabs.harvard.edu/abs/2024arXiv241007311K/abstract'''
        min_mass=self.m_star_min
        max_mass=self.m_star_max
        nom = integr.quad(self.IMF_weighted_func, massive_threshold, max_mass)[0]
        den = integr.quad(self.IMF_weighted_func, min_mass, max_mass)[0]
        self.eta = np.divide(nom, den)
        
    def BD_func(self, mass, alpha_0=0.3, min_mass=0.001, max_mass=0.1):
        '''Brown dwarf power law from the encyclopedia chapter 
        (Kroupa, Gjergo, Jerankova & Yan, 2025)
        https://ui.adsabs.harvard.edu/abs/2024arXiv241007311K/abstract'''
        unnormed_plaw = lambda M: np.power(M, -alpha_0)
        BD_integr = integr.quad(unnormed_plaw, min_mass, max_mass)[0]
        IMF_integr = integr.quad(self.IMF_func, self.m_star_min, self.m_star_max)[0]
        norm_factor = IMF_integr / (4.5 * BD_integr)
        return unnormed_plaw(mass) * norm_factor

class IGIMF(Parameters):
    def __init__(self, metal_mass_fraction=None, SFR=None, alpha1slope=None):
        """
        Initialize an instance of the IGIMF class.

        INPUT
            metal_mass_fraction     [dimensionless] Metal mass fraction (float)
            SFR                     [Msun/yr] Star formation rate (float)
            alpha1slope             [str] Slope of the alpha1 power-law function: 
                                    'linear' or 'logistic'

        GENERATES
            Mtot                    [Msun] Total stellar mass generated in a 
                                    galaxy over 1e7 years.
            m_max                   [Msun] most-massive star in the galaxy
            M_max                   [Msun] most-massive embedded cluster 
                                    produced in the galaxy over 1e7 years
            IGIMF_func              [#/Msun^-1] normalized IGIMF: dN/dm_star
        """
        super().__init__(metal_mass_fraction=metal_mass_fraction, SFR=SFR)
        self.alpha1slope = alpha1slope
        self.Mtot = self.SFR * self.delta_t
        self.ecl_MF = ECMF(SFR=self.SFR)
        self.M_max = self.ecl_MF.M_max
        self.ECMF_func = self.ecl_MF.ECMF_func
        self.stellar_IMF_cache = {}

        # m_max(M_ecl) is monotonic increasing, so the global maximum occurs at M_max
        self.m_max = self.get_stellar_IMF(self.M_max).m_max

        # Precompute Gauss–Legendre nodes and heavy objects once
        self._prepare_quadrature(n_nodes=64)

        # Precompute quadrature weights:  gl_pref * w_i * ECMF(M_i) * (dM/dx)_i
        self._quad_weights = self._gl_pref * self._gl_w * self._ECMF_vals * self._dMdx  # (Nnodes,)
        # Bind per-node stellar-IMF functions to avoid attribute lookups in hot path
        self._phi_funcs = [s.IMF_func for s in self._stellar_nodes]
        # Expose fast callable
        self.IGIMF_func = self._build_gwIMF_callable()

    @lru_cache(maxsize=128)
    def get_stellar_IMF(self, M_ecl):
        return StellarIMF(M_ecl=M_ecl, metal_mass_fraction=self.metal_mass_fraction,
                          alpha1slope=self.alpha1slope)

    def _prepare_quadrature(self, n_nodes=64):
        """Precompute Gauss–Legendre nodes on log10(M_ecl) and cache heavy objects."""
        a = np.log10(self.M_ecl_min)
        b = np.log10(self.M_max)
        xi, wi = leggauss(n_nodes)
        # Map [-1, 1] -> [a, b]
        x = 0.5 * (b - a) * xi + 0.5 * (b + a)
        M_nodes = np.power(10.0, x)
        # dM/dx = ln(10) * 10^x
        dMdx = np.log(10.0) * M_nodes
        # Interval scaling for Gauss–Legendre
        self._gl_pref = 0.5 * (b - a)
        self._gl_w = wi
        self._M_nodes = M_nodes
        self._dMdx = dMdx
        # Heavy precomputations reused for all m
        self._stellar_nodes = [self.get_stellar_IMF(M) for M in M_nodes]
        self._mmax_nodes = np.array([s.m_max for s in self._stellar_nodes], dtype=float)
        self._ECMF_vals = self.ECMF_func(M_nodes)

    def _gwIMF_fast(self, m):
        """Fast gwIMF(m) using precomputed nodes and Gauss–Legendre quadrature."""
        mask = m <= self._mmax_nodes
        if not np.any(mask):
            return 0.0
        vals = np.zeros_like(self._M_nodes, dtype=float)
        for i, s in enumerate(self._stellar_nodes):
            if mask[i]:
                vals[i] = float(s.IMF_func(m))
        integrand = vals * self._ECMF_vals * self._dMdx
        return self._gl_pref * np.dot(self._gl_w, integrand)

    def compute_m_max(self):
        """Compute the global maximum stellar mass"""
        def m_max_of_cluster(M_ecl):
            return self.get_stellar_IMF(M_ecl).m_max

        # Maximizing m_max_of_cluster to find global m_max
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            lambda M_ecl: -m_max_of_cluster(M_ecl),
            bounds=(self.M_ecl_min, self.M_max),
            method='bounded',
            options={'xatol': 1e-5}
        )

        return m_max_of_cluster(res.x)
        
    def _build_gwIMF_callable(self):
        """Return f(m) that evaluates the gwIMF for scalar or 1D array m."""
        weights = self._quad_weights                  # (Nnodes,)
        mmax    = self._mmax_nodes                    # (Nnodes,)
        phi_fns = self._phi_funcs                     # list of Nnodes callables

        def igimf(m):
            m_arr = np.atleast_1d(np.asarray(m, dtype=float))  # (Nm,)
            # Evaluate stellar IMFs at all nodes in one pass over nodes
            # Each fn is vectorized over m, so we get (Nnodes, Nm)
            phi_mat = np.vstack([fn(m_arr) for fn in phi_fns])
            # Mask out contributions where m > m_max(M_node)
            mask = (m_arr[None, :] <= mmax[:, None])
            phi_mat = np.where(mask, phi_mat, 0.0)
            # Gauss–Legendre: sum_i w_i * integrand_i(m)
            y = weights @ phi_mat                               # (Nm,)
            return float(y[0]) if np.ndim(m) == 0 else y

        return igimf
    