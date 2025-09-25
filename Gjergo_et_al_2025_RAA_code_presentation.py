import os
import dill
import time
import numpy as np
import pandas as pd
from igimf import downsizing
from igimf import classes as inst
from igimf import plots as plts

class Vectors:
    '''
    Prepare the input needed to evaluate the various IGIMF classes at time t.

    This class constructs the mass grid and auxiliary arrays required to compute
    xi_IGIMF(m; SFR(t), Z(t)), where:
    - SFR(t): star-formation rate at time t [Msun/yr]
    - Z(t): metal mass fraction at time t (dimensionless)
    - m: stellar mass [Msun]
    '''
    def __init__(self, resolution=9, alpha1slope='logistic'):
        # res 9 for Mecl plots, 8 for Z plots
        # Setup and parameters
        self.alpha1slope = alpha1slope #'linear' # 'logistic'
        self.plots = plts.Plots()
        self.resolution = resolution
        par = inst.Parameters(metal_mass_fraction=0.01, SFR=1) # dummy values to extract param. below:
        self.solar_metallicity = par.solar_metallicity
        self.m_star_max = par.m_star_max
        self.m_star_min = par.m_star_min
        
        # Vectors
        ##self.M_igal_v = self.logspace_v_func(resolution, minlog=6, maxlog=11)
        self.Mecl_v = self.logspace_v_func(100, minlog=np.log10(5), maxlog=10)# for the mmax_mecl plot (k_Z_plot) # best for mmax-Mecl
        self.Mecl_v = self.logspace_v_func(9, minlog=1, maxlog=9) #np.logspace(np.log10(5),10,num=9)
        ##self.Mecl_v = self.logspace_v_func(resolution, minlog=1, maxlog=9) #np.logspace(np.log10(5),10,num=9)
        self.Mecl_v_plot = np.array([1.e1,1.e2,1.e3,1.e7])
        
        #self.Z_massfrac_v = self.logspace_v_func(8, minlog=-7, maxlog=0) # np.logspace(-9,-1,num=resolution) #best for alpha3
        self.Z_massfrac_v = self.logspace_v_func(9, minlog=-7, maxlog=1) # np.logspace(-9,-1,num=resolution) # necessary for mmax colorbar
        #self.Z_massfrac_v = self.logspace_v_func(17, minlog=-7, maxlog=1) # np.logspace(-9,-1,num=resolution) # best for mmax-Mecl
        #self.Z_massfrac_v = self.logspace_v_func(33, minlog=-7, maxlog=1) # np.logspace(-9,-1,num=resolution) # mmax vs SFR plot
        #self.Z_massfrac_v = np.linspace(0,1,num=9)
        #metallicity_v = np.linspace(0.5,1,num=6)
        #self.Z_massfrac_v = np.power(10, metallicity_v)
        
        self.Z_massfrac_v *= self.solar_metallicity # to make subplots labels consistent
        self.Z_massfrac_v_plot = np.power(10, [-4., -1., 0., 0.5]) * self.solar_metallicity
        self.Z_massfrac_v_highZ = np.power(10, [0.3, 0.5, 0.7, 1]) * self.solar_metallicity
        self.metallicity_v = np.log10(self.Z_massfrac_v/self.solar_metallicity)
        self.metallicity_v_plot = np.log10(self.Z_massfrac_v_plot/self.solar_metallicity)
        self.metallicity_v_highZ = np.log10(self.Z_massfrac_v_highZ/self.solar_metallicity)
        
        #self.SFR_v = self.logspace_v_func(37, minlog=-5., maxlog=4.)
        self.SFR_v = self.logspace_v_func(resolution, minlog=-5., maxlog=4.)
        
    def logspace_v_func(self, res, minlog=-1, maxlog=1):
        return np.logspace(minlog, maxlog, num=res)
      
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
            self.downsizing_obj = downsizing.Downsizing(M_igal)
            self.SFR = self.downsizing_obj.SFR.value
        elif SFR:
            self.SFR = SFR
        self.o_ECMF = inst.ECMF(SFR=self.SFR)
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
        self.k_v_list = returned_lists[3]
        self.M_ecl_list = returned_lists[4]
        return None
    
    def return_lists(self):
        beta_ECMF_list, MeclMax_list, ECMF_v_list, k_v_list, M_ecl_list = [], [], [], [], []
        for S in self.SFR_v:
            o_ECMF = SingleECMF(SFR=S)
            beta_ECMF_list.append(o_ECMF.beta_ECMF)
            MeclMax_list.append(o_ECMF.M_max)
            k_v_list.append(o_ECMF.k_ecl)
            M_ecl = np.logspace(np.log10(5), np.log10(o_ECMF.M_max), num=50, endpoint=True)
            M_ecl_list.append(M_ecl)
            ECMF_v = o_ECMF.ECMF_func(M_ecl)
            ECMF_v_list.append(ECMF_v)
        return beta_ECMF_list, MeclMax_list, ECMF_v_list, k_v_list, M_ecl_list
    
    def all_plots(self):
        #self.Meclmax_vs_SFR_observations()
        self.MeclMax_bySFR_plot()
        self.Mecl_power_beta_plot()
        self.ECMF_plots()
    
    def Meclmax_vs_SFR_observations(self):
        self.plots.Meclmax_vs_SFR_observations(self.SFR_v, self.MeclMax_list)
        
    def beta_ECMF_bySFR_plot(self):
        self.plots.beta_ECMF_bySFR_plot(self.SFR_v, self.beta_ECMF_list)
    
    def MeclMax_bySFR_plot(self):
        self.plots.MeclMax_bySFR_plot(self.SFR_v, self.MeclMax_list, self.k_v_list, self.beta_ECMF_list)
    
    def Mecl_power_beta_plot(self):
        self.plots.Mecl_power_beta_plot(self.M_ecl_list, self.beta_ECMF_list)
    
    def ECMF_plots(self):
        self.plots.ECMF_plots(self.M_ecl_list, self.ECMF_v_list, self.SFR_v)
          
class SingleStellarIMF(Vectors):
    def __init__(self, M_ecl:float=None, metal_mass_fraction:float=None):#, SFR:float=None):
        super().__init__()
        self.M_ecl = M_ecl
        self.metal_mass_fraction = metal_mass_fraction
        self.o_IMF = inst.StellarIMF(M_ecl=M_ecl, metal_mass_fraction=metal_mass_fraction, alpha1slope=self.alpha1slope)#, SFR=SFR)
        self.__dict__.update(self.o_IMF.__dict__)
        if self.o_IMF.m_max > 1.:
            self.m_star_v = np.sort(np.concatenate([[1.],np.logspace(np.log10(self.o_IMF.IGIMF_params['Ml']), np.log10(self.o_IMF.m_max), endpoint=True, num=100)]))
        else:
            self.m_star_v = np.logspace(np.log10(self.o_IMF.IGIMF_params['Ml']), np.log10(self.o_IMF.m_max), endpoint=True, num=100) 
        self.IMF_v = self.IMF_func(self.m_star_v)
        self.IMF_weighted_v = self.IMF_weighted_func(self.m_star_v)
        return None
    
    def IMF_plot(self):
        self.plots.IMF_plot(self.m_star_v, self.IMF_v, self.M_ecl, 
                            self.metallicity, self.SFR)
  
class StellarIMFbyMtot(SingleStellarIMF):
    def __init__(self, metal_mass_fraction, SFR, M_ecl=1e5):
        super().__init__(M_ecl, metal_mass_fraction)#, SFR)
        self.return_lists()
        
    def return_lists(self):
        k_IMF_list, m_max_list, IMF_v_list, m_star_v_list = [], [], [], []
        for M in self.Mecl_v:
            imf = SingleStellarIMF(M_ecl=M, metal_mass_fraction=self.metal_mass_fraction)#, SFR=self.SFR)
            k_IMF_list.append(imf.k_star)
            m_max_list.append(imf.m_max)
            IMF_v_list.append(imf.IMF_v)
            m_star_v_list.append(imf.m_star_v)
        self.k_IMF_list = k_IMF_list
        self.m_max_list = m_max_list
        self.IMF_v_list = IMF_v_list
        self.m_star_v_list = m_star_v_list
    
    def IMF_plots(self):
        return self.plots.IMF_plots(self.m_star_v_list, self.IMF_v_list,
                        self.Mecl_v, #self.k_idx, 
                        self.metallicity)
     
class StellarIMFbyZbyMecl(SingleStellarIMF):
    def __init__(self, M_ecl=1e5, metal_mass_fraction=0.1*0.0142):
        super().__init__(M_ecl, metal_mass_fraction)#, SFR)
        self.IMF_Z_v_list_M, self.mw_IMF_Z_v_list_M, self.k_Z_v_list_M, self.m_max_Z_v_list_M, self.m_star_Z_v_list_M = self.return_list_Mecl_variation()
        self.IMF_Z_v_list_Z, self.mw_IMF_Z_v_list_Z, self.k_Z_v_list_Z, self.m_max_Z_v_list_Z, self.m_star_Z_v_list_Z = self.return_list_Z_variation()
        self.IMF_Z_v_list_all, self.mw_IMF_Z_v_list_all, self.k_Z_v_list_all, self.m_max_Z_v_list_all, self.m_star_Z_v_list_all, self.alpha3_Z_v_list_all, self.alpha1_Z_v_list_all = self.return_list_all_variation()
        self.k_v_list, self.m_max_v_list, self.alpha3_v_list, self.alpha1_v_list = self.mmax_Mecl_slightly_supersolar(Z=2*self.solar_metallicity)
        
    def return_list_Mecl_variation(self):
        IMF_Z_v_list = []
        mw_IMF_Z_v_list = []
        k_Z_v_list = []
        m_max_Z_v_list = []
        m_star_Z_v_list = []
        for Mval in self.Mecl_v_plot:
            IMF_v_list = []
            mw_IMF_v_list = []
            k_v_list = []
            m_max_v_list = []
            m_star_v_list = []
            for Z in self.Z_massfrac_v:
                imf = SingleStellarIMF(M_ecl=Mval, metal_mass_fraction=Z)
                m_star_v_list.append(np.append(imf.m_star_v, imf.m_star_v[-1]+0.01))
                IMF_v_list.append(np.append(imf.IMF_v, 0.))
                mw_IMF_v_list.append(np.append(imf.IMF_weighted_v, 0.))
                k_v_list.append(imf.k_star)
                m_max_v_list.append(imf.m_max)
            IMF_Z_v_list.append(IMF_v_list)
            mw_IMF_Z_v_list.append(mw_IMF_v_list)
            k_Z_v_list.append(k_v_list)
            m_max_Z_v_list.append(m_max_v_list)   
            m_star_Z_v_list.append(m_star_v_list)
        return IMF_Z_v_list, mw_IMF_Z_v_list, k_Z_v_list, m_max_Z_v_list, m_star_Z_v_list
    
    def return_list_Z_variation(self):
        IMF_Z_v_list = []
        mw_IMF_Z_v_list = []
        k_Z_v_list = []
        m_max_Z_v_list = []
        m_star_Z_v_list = []
        for Mval in self.Mecl_v:
            IMF_v_list = []
            mw_IMF_v_list = []
            k_v_list = []
            m_max_v_list = []
            m_star_v_list = []
            for Z in self.Z_massfrac_v_plot:
                imf = SingleStellarIMF(M_ecl=Mval, metal_mass_fraction=Z)#, SFR=None)
                m_star_v_list.append(np.append(imf.m_star_v, imf.m_star_v[-1]+0.01))
                IMF_v_list.append(np.append(imf.IMF_v, 0.))
                mw_IMF_v_list.append(np.append(imf.IMF_weighted_v, 0.))
                k_v_list.append(imf.k_star)
                m_max_v_list.append(imf.m_max)
            IMF_Z_v_list.append(IMF_v_list)
            mw_IMF_Z_v_list.append(mw_IMF_v_list)
            k_Z_v_list.append(k_v_list)
            m_max_Z_v_list.append(m_max_v_list)   
            m_star_Z_v_list.append(m_star_v_list)
        return IMF_Z_v_list, mw_IMF_Z_v_list, k_Z_v_list, m_max_Z_v_list, m_star_Z_v_list

    def return_list_all_variation(self):
        IMF_Z_v_list = []
        mw_IMF_Z_v_list = []
        k_Z_v_list = []
        m_max_Z_v_list = []
        m_star_Z_v_list = []
        alpha3_Z_v_list = []
        alpha1_Z_v_list = []
        for Z in self.Z_massfrac_v:
            IMF_v_list = []
            mw_IMF_v_list = []
            k_v_list = []
            m_max_v_list = []
            m_star_v_list = []
            alpha3_v_list = []
            alpha1_v_list = []
            for Mval in self.Mecl_v:
                imf = SingleStellarIMF(M_ecl=Mval, metal_mass_fraction=Z)
                m_star_v_list.append(np.append(imf.m_star_v, imf.m_star_v[-1]+0.01))
                IMF_v_list.append(np.append(imf.IMF_v, 0.))
                mw_IMF_v_list.append(np.append(imf.IMF_weighted_v, 0.))
                k_v_list.append(imf.k_star)
                m_max_v_list.append(imf.m_max)
                alpha3_v_list.append(imf.IGIMF_params['alpha3'])
                alpha1_v_list.append(imf.IGIMF_params['alpha1'])
            IMF_Z_v_list.append(IMF_v_list)
            mw_IMF_Z_v_list.append(mw_IMF_v_list)
            k_Z_v_list.append(k_v_list)
            m_max_Z_v_list.append(m_max_v_list)   
            m_star_Z_v_list.append(m_star_v_list)
            alpha3_Z_v_list.append(alpha3_v_list)
            alpha1_Z_v_list.append(alpha1_v_list)
        return IMF_Z_v_list, mw_IMF_Z_v_list, k_Z_v_list, m_max_Z_v_list, m_star_Z_v_list, alpha3_Z_v_list, alpha1_Z_v_list
    
    def mmax_Mecl_slightly_supersolar(self, Z=2*0.0142):
        k_v_list = []
        m_max_v_list = []
        alpha3_v_list = []
        alpha1_v_list = []
        for Mval in self.Mecl_v:
            imf = SingleStellarIMF(M_ecl=Mval, metal_mass_fraction=Z)
            k_v_list.append(imf.k_star)
            m_max_v_list.append(imf.m_max)
            alpha3_v_list.append(imf.IGIMF_params['alpha3'])
            alpha1_v_list.append(imf.IGIMF_params['alpha1'])
        return k_v_list, m_max_v_list, alpha3_v_list, alpha1_v_list
    
    def Encyclopedia_main_plot(self, Mecl_v=[1e3,1e6]):
        return self.plots.Encyclopedia_main_plot(Mecl_v, alpha1slope=self.alpha1slope)
        
    def sIMF_subplot(self):
        '''stellarIMF_subplots_Zcolorbar
        i.e., 4 panels each with different M_ecl'''
        return self.plots.sIMF_subplot(self.metallicity_v, self.Mecl_v_plot, 
                                    self.m_star_Z_v_list_M, self.IMF_Z_v_list_M, alpha1slope=self.alpha1slope)    
    
    def sIMF_subplot_norm1(self):
        '''stellarIMF_subplots_Zcolorbar
        i.e., 4 panels each with different M_ecl'''
        return self.plots.sIMF_subplot_norm1(self.metallicity_v, self.Mecl_v_plot, 
                                    self.m_star_Z_v_list_M, self.IMF_Z_v_list_M, alpha1slope=self.alpha1slope)    
        
    def mw_sIMF_subplot(self):
        return self.plots.mw_sIMF_subplot(self.metallicity_v, self.Mecl_v_plot, 
                                    self.m_star_Z_v_list_M, self.mw_IMF_Z_v_list_M, alpha1slope=self.alpha1slope)
        
    def mw_sIMF_subplot_norm1(self):
        return self.plots.mw_sIMF_subplot_norm1(self.metallicity_v, self.Mecl_v_plot, 
                                    self.m_star_Z_v_list_M, self.mw_IMF_Z_v_list_M, alpha1slope=self.alpha1slope)
    
    def sIMF_subplot_Mecl(self):
        return self.plots.sIMF_subplot_Mecl(self.metallicity_v_plot, self.Mecl_v,
                                            self.m_star_Z_v_list_Z, self.IMF_Z_v_list_Z, alpha1slope=self.alpha1slope)
        
    def sIMF_subplot_Mecl_supersolar(self):
        return self.plots.sIMF_subplot_Mecl_supersolar(self.metallicity_v_highZ, self.Mecl_v, solar_metallicity=self.solar_metallicity)
        
    def mw_sIMF_subplot_Mecl(self):
        return self.plots.mw_sIMF_subplot_Mecl(self.metallicity_v_plot, self.Mecl_v, 
                                    self.m_star_Z_v_list_Z, self.mw_IMF_Z_v_list_Z, alpha1slope=self.alpha1slope)
        
    def k_Z_plot(self):
        return self.plots.k_Z_plot(self.Z_massfrac_v, self.k_Z_v_list_all, self.m_max_Z_v_list_all, self.Mecl_v,
                                   self.k_v_list, self.m_max_v_list,  solar_metallicity=self.solar_metallicity,
                alpha1slope=self.alpha1slope)
    
    def k_Z_alpha3_plot(self):
        return self.plots.k_Z_alpha3_plot(self.Z_massfrac_v, self.m_max_Z_v_list_all, self.Mecl_v, 
                                   self.alpha3_v_list, self.alpha3_Z_v_list_all, solar_metallicity=self.solar_metallicity,
                alpha1slope=self.alpha1slope)
        
    def k_Z_alpha3alpha1_plot(self):
        return self.plots.k_Z_alpha3alpha1_plot(self.Z_massfrac_v,self. m_max_Z_v_list_all, self.Mecl_v, self.m_max_v_list,  
                 self.alpha3_v_list, self.alpha3_Z_v_list_all, self.alpha1_v_list, self.alpha1_Z_v_list_all, solar_metallicity=0.0142,
                 alpha1slope=self.alpha1slope)
        
    def alpha1_Z_plot(self):
        return self.plots.alpha1_Z_plot()
        
    def alpha3_plot(self):
        from sklearn.model_selection import ParameterGrid
        parameter_space = {'M_ecl': self.Mecl_v, 'Zmassfrac': self.Z_massfrac_v}
        dict_list = list(ParameterGrid(parameter_space))
        alpha3_v = []
        rho_space = []
        for pair in dict_list:
            imf = SingleStellarIMF(M_ecl=pair['M_ecl'], metal_mass_fraction=pair['Zmassfrac'])#, SFR=None)
            rho_space.append(np.log10(imf.rho_cl))
            alpha3_v.append(imf.alpha_3)
        parameter_space['log10_rho_cl'] = np.unique(rho_space)
        parameter_space['[Z]'] = self.metallicity_v
        
        return self.plots.alpha3_plot(alpha3_v, parameter_space)


    def alpha1_plot(self):
        from sklearn.model_selection import ParameterGrid
        parameter_space = {'M_ecl': self.Mecl_v, 'Zmassfrac': self.Z_massfrac_v}
        dict_list = list(ParameterGrid(parameter_space))
        alpha1_v = []
        rho_space = []
        for pair in dict_list:
            imf = SingleStellarIMF(M_ecl=pair['M_ecl'], metal_mass_fraction=pair['Zmassfrac'])#, SFR=None)
            rho_space.append(np.log10(imf.rho_cl))
            alpha1_v.append(imf.alpha_1)
        parameter_space['log10_rho_cl'] = np.unique(rho_space)
        parameter_space['[Z]'] = self.metallicity_v
    
        return self.plots.alpha1_plot(alpha1_v, parameter_space, resolution=self.resolution, alpha1slope=self.alpha1slope)
    
    def mmax_plot(self):
        from sklearn.model_selection import ParameterGrid
        parameter_space = {'M_ecl': self.Mecl_v, 'Zmassfrac': self.Z_massfrac_v}
        dict_list = list(ParameterGrid(parameter_space))
        mmax_v = []
        rho_space = []
        for pair in dict_list:
            imf = SingleStellarIMF(M_ecl=pair['M_ecl'], metal_mass_fraction=pair['Zmassfrac'])#, SFR=None)
            rho_space.append(np.log10(imf.rho_cl))
            mmax_v.append(imf.m_max)
        parameter_space['log10_rho_cl'] = np.unique(rho_space)
        parameter_space['[Z]'] = self.metallicity_v
    
        return self.plots.mmax_plot(mmax_v, parameter_space, alpha1slope=self.alpha1slope)

class InstanceIGIMF(Vectors):
    def __init__(self, metal_mass_fraction:float=None, SFR:float=None, computeV=False):
        '''
        Constructor for the InstanceIGIMF class.
        The IGIMF is evaluated for the paired inputs (SFR(t), Z(t)) and returned as a
        function of m over the configured mass grid.

        Parameters
        ----------
        metal_mass_fraction : float
            Metal mass fraction.
        SFR : float
            Star formation rate.
        computeV : bool
            Whether to compute the IMF vector.

        '''
        super().__init__()
        self.o_IGIMF = inst.IGIMF(metal_mass_fraction=metal_mass_fraction, SFR=SFR, alpha1slope=self.alpha1slope)
        self.__dict__.update(self.o_IGIMF.__dict__)
        if np.isnan(self.o_IGIMF.m_max): 
            self.o_IGIMF.m_max = self.o_IGIMF.m_star_max
        self.m_star_v = np.logspace(np.log10(0.08), np.log10(self.o_IGIMF.m_max), endpoint=True, num=100)
        
        if computeV is True:
           self.IGIMF_v = self.o_IGIMF.IGIMF_func(self.m_star_v)


class IGIMFGrid(Vectors):   
    '''Creates IGIMF grids'''  
    def __init__(self, folder_path=None, folder_panels=None):
        super().__init__()
        if folder_path == None:
            folder_path = f'./grid/alpha1_{self.alpha1slope}/' 
        self.folder_path=folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        if folder_panels == None:
            folder_panels = f'./grid/panelplot_{self.alpha1slope}/' 
        self.folder_panels=folder_panels
        if not os.path.exists(folder_panels):
            os.makedirs(folder_panels)
    
    def by_Z_by_SFR_dill(self):
        print('\nStarting to create the IGIMF grids by SFR and [Z]')
        print(f'Saved in the folder \n{self.folder_path}\n')
        Slen = len(self.SFR_v)
        Zlen = len(self.Z_massfrac_v)
        for i,S in enumerate(self.SFR_v):
            for j,Z in enumerate(self.Z_massfrac_v):
                print (f'Metals at {j+1}/{Zlen}, SFR at {i+1}/{Slen}\nwith values [Z]={np.log10(Z/self.solar_metallicity):.2e} and {S=:.2e}, respectively')
                igimf = InstanceIGIMF(metal_mass_fraction=Z, SFR=S, computeV=True)
                df = {}
                df['SFR'] = S
                df['metal_mass_fraction'] = Z
                df['M_max'] = igimf.M_max
                df['m_max'] = igimf.m_max
                m_star_v = np.logspace(np.log10(igimf.m_star_min), np.log10(igimf.m_max), endpoint=True, num=100)
                df['mass_star'] = m_star_v
                df['IGIMF'] = igimf.IGIMF_func(m_star_v)
                df['IGIMF_func'] = igimf.IGIMF_func
                df = pd.DataFrame(df)
                dill.dump(df,open(f'{self.folder_path}igimf_SFR{S}_Z{Z}.pkl','wb'))
        return None
    

    def by_Z_by_SFR_dill_plot(self):
        #SFR = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4]
        #metallicity = [1e-7, 1e-5, 1e-3, 1e-2, 0.1, 1, 10]
        #df = {'SFR':[], 'metal_mass_fraction':[], 'mass_star':[], 'IGIMF':[]}
        #SFR = [1.e-5, 3.e-2, 1., 1.e3]
        #metal_mass_fraction = [1.e-4, 1.e-1, 10**-0.5, 1., 2., 10**0.5, 5, 10]
        SFR = np.logspace(-5, 3, 40)  
        metal_mass_fraction = np.logspace(-4, np.log10(10), 50)  
        Slen = len(SFR)
        Zlen = len(metal_mass_fraction)
        for i,S in enumerate(SFR):
            for j,Z in enumerate(np.multiply(metal_mass_fraction, self.solar_metallicity)):
                print (f'Metals at {j+1}/{Zlen}, SFR at {i+1}/{Slen}\nwith values [Z]={np.log10(Z/self.solar_metallicity):.2e} and {S=:.2e}, respectively')
                igimf = InstanceIGIMF(metal_mass_fraction=Z, SFR=S, computeV=True)
                df = {}
                df['SFR'] = S
                df['metal_mass_fraction'] = Z
                df['M_max'] = igimf.M_max
                df['m_max'] = igimf.m_max
                m_star_v = np.logspace(np.log10(igimf.m_star_min), np.log10(igimf.m_max), endpoint=True, num=100)
                df['mass_star'] = m_star_v
                df['IGIMF'] = igimf.IGIMF_func(m_star_v)
                df['IGIMF_func'] = igimf.IGIMF_func
                df = pd.DataFrame(df)
                dill.dump(df,open(f'grid/panelplot_{self.alpha1slope}/igimf_SFR{S}_Z{Z}.pkl','wb'))
        return None


if __name__ == '__main__':
    metal_mass_fraction = 1e-1 * 0.0142
    M_igal = 1e10
    M_ecl = 1e5
    o_ecmf = SingleECMF(M_igal=M_igal)
    o_ecmf.plots.Cook23_plot()
    o_ecmf.plots.Kroupa_canonical_plot()
        
    ecmf_by_SFR = ECMFbySFR(M_igal=M_igal)
    ecmf_by_SFR.all_plots()
    
    print('\n Creating a StellarIMFbyZbyMecl')
    sIMF_by_Z = StellarIMFbyZbyMecl()
    sIMF_by_Z.alpha1_Z_plot()
    sIMF_by_Z.sIMF_subplot_Mecl()
    sIMF_by_Z.sIMF_subplot_Mecl_supersolar()
    sIMF_by_Z.mw_sIMF_subplot()
    sIMF_by_Z.k_Z_plot()
    sIMF_by_Z.k_Z_alpha3alpha1_plot()
    sIMF_by_Z.mmax_plot()
    sIMF_by_Z.alpha3_plot()
    sIMF_by_Z.alpha1_plot()

    print('\n Creating an InstanceIGIMF')
    start_time = time.time()
    instance_IGIMF = InstanceIGIMF(metal_mass_fraction=metal_mass_fraction, SFR=o_ecmf.SFR, computeV=True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Process took {elapsed:.4f} seconds.")
    fig = plts.plt.figure()
    plts.plt.loglog(instance_IGIMF.m_star_v, instance_IGIMF.IGIMF_v)
    plts.plt.show(block=False)
    
    # print('\n Creating IGIMFGrid')    
    # create_grid = IGIMFGrid()
    # create_grid.by_Z_by_SFR_dill() 
    # create_grid.by_Z_by_SFR_dill_plot()
    