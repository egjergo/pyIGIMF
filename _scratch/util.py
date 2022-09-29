    
def weighted_func(M, func, *args, **kwargs):
    return np.multiply(M, func(M))

def normalized(x, func, condition=None, *args, **kwargs):
    ''' IMF behavior depending on whether or not it has been normalized '''
    if condition:
        if x <= condition:
            return func
        else:
            return 0.
    else:
        return func
        
def normalization(IMF, M, lower_lim, upper_lim, *args, **kwargs) -> (float, float):
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
    *args   other required arguments
    **kwargs    optional keyword arguments
        
    -----
        
    .. math::
    `M = \int_{\mathrm{lower_lim}}^{\mathrm{m_max}}
    m \, \mathrm{IMF}(m,...)\,\mathrm{d}m`
        
    .. math::
    `1 = \int_{\mathrm{m_max}}^{{\rm upper_lim}}{\mathrm{IMF}(m,...)} \,\mathrm{d}m`
    '''
    k = lambda x: np.reciprocal(integr.quad(IMF, x, upper_lim, args=(args))[0])
    def weighted_IMF(m, x, *args):
        return m * IMF(m, *args) * k(x)
    func = lambda x: (integr.quad(weighted_IMF, lower_lim, x, args=(x, *args))[0] - M)
    sol = optimize.root_scalar(func, bracket=[lower_lim, upper_lim], rtol=1e-8)
    m_max = sol.root
    return k(m_max), m_max

def get_norm(IMF, Mtot, min_val, max_val, *args, **kwargs):
    '''duplicate of stellar_IMF !!!!!!! '''
    k, M = util.normalization(IMF, Mtot, min_val, min_val, *args, **kwargs)
    IMF_func = lambda mass: k * IMF(mass, m_max=M, *args, **kwargs)
    IMF_weighted_func = lambda mass: self.weighted_func(mass, IMF_func, *args, **kwargs)
    return k, M, IMF_func, IMF_weighted_func