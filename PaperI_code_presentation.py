import numpy as np
import pandas as pd
from igimf import instance as inst
from igimf import util as utl
from igimf import plots as plts

class Vectors:
    '''
    Set up the vectors in order to compute the IGIMF instances
    across a range of (t, SFR(t), Z(t))
    '''
    def __init__(self, resolution=50):
        
        # Setup and parameters
        self.plots = plts.Plots()
        self.resolution = resolution
        par = inst.Parameters(0.01, 1) # dummy values to extract param. below:
        self.solar_metallicity = par.solar_metallicity
        self.m_star_max = par.m_star_max
        self.m_star_min = par.m_star_min
        
        # Vectors
        self.M_igal_v = np.logspace(6, 11,num=resolution)
        self.Mecl_v = np.logspace(np.log10(5),10,num=resolution)
        self.Z_massfrac_v = np.logspace(-9,-1,num=resolution)
        self.mstar_v = np.logspace(np.log10(self.m_star_min),
                                   np.log10(self.m_star_max), num=resolution)
        self.SFR_v = np.logspace(-6,4,num=resolution)
        self.metallicity_v = np.log10(self.Z_massfrac_v/self.solar_metallicity)
        
class SingleECMF(Vectors):
    """
    A single instance of ECMF.
    
    Requires either
    M_igal (to extract the SFR through Downsizing)
    or SFR directly.
    """
    def __init__(self, M_igal=None, SFR=None):
        super().__init__()
        if M_igal is not None and SFR is not None:
            raise ValueError("Define only one between M_igal and SFR")
        elif M_igal:
            self.M_igal = M_igal
            self.downsizing_obj = utl.Downsizing(M_igal)
            self.SFR = self.downsizing_obj.SFR
        elif SFR:
            self.SFR = SFR
        self.o_ECMF = inst.ECMF(self.SFR)
        self.__dict__.update(self.o_ECMF.__dict__)
        return None

    def ECMF_plot(self):
        ECMF_v = self.ECMF_func(self.Mecl_v)
        self.plots.ECMF_plot(self.Mecl_v, ECMF_v, self.SFR)

class ECMFbySFR(SingleECMF):
    '''
    Compute various ECMFs along various SFR(t) 
    (e.g., use SFR_v from the Vectors superclass)
    '''
    def __init__(self, M_igal=None, SFR=None):
        if M_igal is None and SFR is None:
            M_igal = 1e10 #M_igal = 1e10, a dummy
        super().__init__(M_igal=M_igal, SFR=SFR) 
        returned_lists = self.return_lists()
        self.beta_ECMF_list = returned_lists[0]
        self.MeclMax_list = returned_lists[1]
        self.ECMF_v_list = returned_lists[2]
        return None
    
    def return_lists(self):
        beta_ECMF_list, MeclMax_list, ECMF_v_list = [], [], []
        for S in self.SFR_v:
            o_ECMF = SingleECMF(SFR=S)
            beta_ECMF_list.append(o_ECMF.beta_ECMF)
            MeclMax_list.append(o_ECMF.M_max)
            ECMF_v = o_ECMF.ECMF_func(self.Mecl_v)
            ECMF_v_list.append(ECMF_v)
        return beta_ECMF_list, MeclMax_list, ECMF_v_list
    
    def all_plots(self):
        self.beta_ECMF_bySFR_plot()
        self.MeclMax_bySFR_plot()
        self.Mecl_power_beta_plot()
        self.ECMF_plots()
    
    def beta_ECMF_bySFR_plot(self):
        self.plots.beta_ECMF_bySFR_plot(self.SFR_v, self.beta_ECMF_list)
    
    def MeclMax_bySFR_plot(self):
        self.plots.MeclMax_bySFR_plot(self.SFR_v, self.MeclMax_list)
    
    def Mecl_power_beta_plot(self):
        self.plots.Mecl_power_beta_plot(self.Mecl_v, self.beta_ECMF_list)
    
    def ECMF_plots(self):
        self.plots.ECMF_plots(self.Mecl_v, self.ECMF_v_list, self.SFR_v)
   
           
class SingleStellarIMF(Vectors):
    def __init__(self, M_ecl, metal_mass_fraction, SFR):
        super().__init__()
        self.M_ecl = M_ecl
        self.metal_mass_fraction = metal_mass_fraction
        self.SFR = SFR
        self.o_IMF = inst.StellarIMF(M_ecl, metal_mass_fraction, SFR)
        self.__dict__.update(self.o_IMF.__dict__)
        self.IMF_v = self.IMF_func(self.mstar_v)
        return None
    
    def IMF_plot(self):
        self.plots.IMF_plot(self.mstar_v, self.IMF_v, self.M_ecl, 
                            self.metallicity, self.SFR)
   
class StellarIMFbyMtot(SingleStellarIMF):
    def __init__(self, metal_mass_fraction, SFR, M_ecl=1e5):
        super().__init__(M_ecl, metal_mass_fraction, SFR)
        self.return_lists()
        self.k_idx = np.where(np.array(self.k_IMF_list)<=1e10)[0]
    
    def return_lists(self):
        k_IMF_list, m_max_list, IMF_v_list = [], [], []
        for M in self.Mecl_v:
            imf = SingleStellarIMF(M, self.metal_mass_fraction, self.SFR)
            k_IMF_list.append(imf.k_star)
            m_max_list.append(imf.m_max)
            IMF_v_list.append(imf.IMF_v)
        self.k_IMF_list = k_IMF_list
        self.m_max_list = m_max_list
        self.IMF_v_list = IMF_v_list
    
    def IMF_plots(self):
        return self.plots.IMF_plots(self.mstar_v, self.IMF_v_list,
                        self.Mecl_v, self.k_idx, self.metal_mass_fraction)
        
        
class StellarIMFbyZbyMecl(SingleStellarIMF):
    def __init__(self, SFR, M_ecl=1e5, metal_mass_fraction=0.1*0.0134,
                 compute_IMF_by_Z=False):
        super().__init__(M_ecl, metal_mass_fraction, SFR)
        if compute_IMF_by_Z is False:
            import pickle
            self.IMF_Z_v_list = pickle.load(open('IMF_Z_v_list.pkl', 'rb'))
        else:
            self.IMF_Z_v_list = self.return_list()
    
    def return_list(self):
        IMF_Z_v_list = []
        for M in self.Mecl_v:
            IMF_v_list = []
            for Z in self.Z_massfrac_v:
                imf = SingleStellarIMF(M, Z, self.SFR)
                IMF_v_list.append(imf.IMF_func(self.mstar_v))
            IMF_Z_v_list.append(IMF_v_list)
        return IMF_Z_v_list 
    
    def sIMF_subplot(self):
        return self.plots.sIMF_subplot(self.metallicity_v, self.Mecl_v, 
                                    self.mstar_v, self.IMF_Z_v_list)
    
    def sIMF_subplot_Mecl(self):
        return self.plots.sIMF_subplot_Mecl(self.metallicity_v, self.Mecl_v,
                                            self.mstar_v, self.IMF_Z_v_list)


class InstanceIGIMF(Vectors):
    def __init__(self, metal_mass_fraction:float, SFR:float, computeV=False):
        super().__init__()
        self.o_IGIMF = inst.IGIMF(metal_mass_fraction, SFR)
        self.__dict__.update(self.o_IGIMF.__dict__)
        if computeV is True:
            self.IGIMF_v = self.o_IGIMF.IGIMF_func(self.mstar_v)

class IGIMFGrid(InstanceIGIMF):   
    '''Creates IGIMF grids'''  
    def __init__(self, metal_mass_fraction=0.1*0.0134, SFR=1):
        super().__init__(metal_mass_fraction, SFR)
        
    def by_Z_by_SFR(self):
        IGIMF_v_list = []
        for S in self.SFR_v:
            IGIMF_list = []
            for Z in self.Z_massfrac_v:
                igimf = InstanceIGIMF(Z, S, computeV=True)
                print(igimf.IGIMF_v)
                IGIMF_list.append(igimf.IGIMF_v)
            IGIMF_v_list.append(IGIMF_list)
        return IMF_Z_S_v_list 
    
    def by_Z_by_SFR_pickle(self):
        import pickle
        for S in self.SFR_v:
            IGIMF_list = []
            for Z in self.Z_massfrac_v:
                igimf = InstanceIGIMF(Z, S, computeV=True)
                pickle.dump(igimf.__dict__,open(f'grid/igimf_SFR{S}_Z{Z}.pkl','wb'))
        return None
     
if __name__ == '__main__':
    metal_mass_fraction = 1e-1 * 0.0134
    M_igal = 1e10
    M_ecl = 1e5
    o_igimf = SingleECMF(M_igal=M_igal)
    o_igimf.ECMF_plot()
    
    ecmf_by_SFR = ECMFbySFR(M_igal=M_igal)
    ecmf_by_SFR.all_plots()
    
    stellar_IMF = SingleStellarIMF(M_ecl, metal_mass_fraction, o_igimf.SFR)
    stellar_IMF.IMF_plot()
    
    sIMF_by_Mtot = StellarIMFbyMtot(metal_mass_fraction, o_igimf.SFR)
    sIMF_by_Mtot.IMF_plots()
    
    sIMF_by_Z = StellarIMFbyZbyMecl(o_igimf.SFR, compute_IMF_by_Z=True)
    sIMF_by_Z.sIMF_subplot()
    sIMF_by_Z.sIMF_subplot_Mecl()
    
    instance_IGIMF = InstanceIGIMF(metal_mass_fraction, o_igimf.SFR,
                                   computeV=True)
    print(instance_IGIMF.IGIMF_v)
    
    create_grid = IGIMFGrid()
    create_grid.by_Z_by_SFR_pickle()