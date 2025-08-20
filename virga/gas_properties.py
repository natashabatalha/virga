import numpy as np

def TiO2(mw_atmos, mh=1, gas_mmr = None):
    """Defines properties for TiO2 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio.
        None points to the default value of : 1.69e-7
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    mean molecular weight of gas,
    gas mass mixing ratio 
    density of gas cgs
    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in TiO2 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 1.69e-7 * mh
    gas_mw = 80
    gas_mmr = gas_mmr * (gas_mw/mw_atmos) 
    rho_p =  4.25
    return gas_mw, gas_mmr, rho_p

def CH4(mw_atmos , mh = 1,gas_mmr = None):
    """Defines properties for CH4 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio.
        None points to the default value of : 4.9e-4
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    mean molecular weight of gas,
    gas mass mixing ratio 
    density of gas cgs
    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in CH4 Routine. Consult your local theorist to determine next steps.")
    #ch4 is linearly dependent on m/h at cold temperatures but scales as m^0 at hotter temps where co is dominant 
    #but we can assume a linear dependence here 
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 4.9e-4 * mh

    gas_mw = 16.0
    # Lodders(2003)
    gas_mmr = gas_mmr * (gas_mw/mw_atmos) 
    #V.G. Manzhelii and A.M. Tolkachev, Sov. Phys. Solid State 5, 2506 (1964)
    rho_p =  0.49   #solid (T = 213 K)
    return gas_mw, gas_mmr, rho_p

def NH3(mw_atmos, mh = 1, gas_mmr = None):
    """Defines properties for NH3 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 1.34e-4
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    mean molecular weight of gas,
    gas mass mixing ratio 
    density of gas cgs
    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in NH3 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 1.34e-4 * mh

    gas_mw = 17.
    #Lodders(2003)
    gas_mmr = gas_mmr * (gas_mw/mw_atmos) 
    #V.G. Manzhelii and A.M. Tolkachev, Sov. Phys. Solid State 5, 2506 (1964)
    rho_p =  0.84  #solid (T = 213 K)
    return gas_mw, gas_mmr, rho_p

def H2O(mw_atmos, mh = 1, gas_mmr = None):
    """Defines properties for H2O as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 7.54e-4
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    mean molecular weight of gas,
    gas mass mixing ratio 
    density of gas cgs
    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in H2O Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 7.54e-4 * mh
    gas_mw = 18.
    #Lodders(2003)
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)  
    #V.G. Manzhelii and A.M. Tolkachev, Sov. Phys. Solid State 5, 2506 (1964)
    rho_p =  0.93   #solid (T = 213 K)
    return gas_mw, gas_mmr, rho_p

def Fe(mw_atmos,  mh = 1,gas_mmr = None):
    """Defines properties for Fe as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 5.095e-5
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    mean molecular weight of gas,
    gas mass mixing ratio 
    density of gas cgs

    Notes
    -----
    .. [1] Morley C.~V., Mukherjee S., Marley M.~S., Fortney J.~J., Visscher C., Lupu R., Gharib-Nezhad E., et al., 2024, ApJ, 975, 59. doi:10.3847/1538-4357/ad71d5

    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in Fe Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr =  5.095e-5 * mh#5.78e-5 * mh #
    gas_mw = 55.845
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)
    #Lodders and Fegley (1998)
    rho_p =  7.875  
    return gas_mw, gas_mmr, rho_p

def KCl(mw_atmos, mh = 1, gas_mmr = None):
    """Defines properties for KCl as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs
    """ 
    #if mh != 1: raise Exception("Alert: No M/H Dependence in Fe Routine. Consult your local theorist to determine next steps.")

    gas_mw = 74.5
    #mh dependence from Morley12
    #mh_values = np.array([0,0.3,0.5,0.7,1.0,1.7,2.0,2.5])
    #gas_mmrs = np.array([2.2627E-07, 2*2.2627E-07, 3.1*2.2627E-07,5*2.2627E-07, 2.1829E-06,  8.1164E-06,2*8.1164E-06, 3*2*8.1164E-06])

    if isinstance(gas_mmr, type(None)):
        gas_mmr = 2.55e-7 * mh #np.interp(np.log10(mh), mh_values, gas_mmrs) * (gas_mw/mw_atmos)
    
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)
    #source unknown
    rho_p =  1.99
    return gas_mw, gas_mmr, rho_p

def MgSiO3(mw_atmos, mh = 1, gas_mmr = None):
    """Defines properties for MgSiO3 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 2.75e-3
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Morley C.~V., Mukherjee S., Marley M.~S., Fortney J.~J., Visscher C., Lupu R., Gharib-Nezhad E., et al., 2024, ApJ, 975, 59. doi:10.3847/1538-4357/ad71d5

    """ 
    gas_mw = 100.4
    #if mh != 1: raise Exception("Alert: No M/H Dependence in MgSiO3 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr =  2.92e-5 *  mh # # gao 202 nat astro 60.3e-6 
    
    gas_mmr =  gas_mmr * (gas_mw/mw_atmos)
    #Lodders and Fegley (1998)
    rho_p =  3.192
    return gas_mw, gas_mmr, rho_p

def Mg2SiO4(mw_atmos, mh = 1,gas_mmr = None,):
    """Defines properties for Mg2SiO4 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 3.58125e-05
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Morley C.~V., Mukherjee S., Marley M.~S., Fortney J.~J., Visscher C., Lupu R., Gharib-Nezhad E., et al., 2024, ApJ, 975, 59. doi:10.3847/1538-4357/ad71d5

    """ 
    #if mh != 1: raise Exception("Alert: No M/H Dependence in Mg2SiO4 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr =  3.063e-5 * mh #59.36e-6 * mh
    gas_mw = 140.7
    #NEW FORSTERITE (from Lodders et al. table, 1000mbar, 1900K)
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)  
    #Lodders and Fegley (1998)
    rho_p =  3.214  
    return gas_mw, gas_mmr, rho_p

def MnS(mw_atmos, mh=1,gas_mmr = None): 
    """Defines properties for MnS as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 6.32e-7
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Lodders, K. Solar System Abundances of the Elements. Astrophys. Space Sci.
Proc. 16, 379 (2010)
    """ 
    if isinstance(gas_mmr, type(None)):
        gas_mmr =  5.41e-7 * mh

    gas_mw = 87.00

    gas_mmr = gas_mmr * (gas_mw/mw_atmos) 

    #Lodders and Fegley (2003) (cvm)
    rho_p =  4.0
    return gas_mw, gas_mmr, rho_p

def ZnS(mw_atmos, mh=1, gas_mmr=None):
    """Defines properties for ZnS as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 8.40e-8
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Lodders, K. Solar System Abundances of the Elements. Astrophys. Space Sci.
Proc. 16, 379 (2010)
    """ 
    if isinstance(gas_mmr, type(None)):
        gas_mmr =  0.076e-6 * mh #lodders 2010

    gas_mw = 97.46

    gas_mmr = gas_mmr * (gas_mw/mw_atmos) 

    #Lodders and Fegley (2003) (cvm)
    rho_p =  4.04   
    return gas_mw, gas_mmr, rho_p

def Cr(mw_atmos, mh=1,gas_mmr=None):
    """Defines properties for Cr as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of :
        1xSolar = 8.87e-7
        10xSolar = 8.6803E-06
        50xSolar = 4.1308E-05
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Lodders, K. Solar System Abundances of the Elements. Astrophys. Space Sci.
    Proc. 16, 379 (2010)    
    """     
    gas_mw = 51.996
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 8.87e-7 *  mh
        #if mh==1:
        #    gas_mmr = 8.87e-7 * (gas_mw/mw_atmos) 
        #elif mh==10:
        #    gas_mmr = 8.6803E-06 * (gas_mw/mw_atmos) 
        #elif mh==50:
        #    gas_mmr = 4.1308E-05 * (gas_mw/mw_atmos) 
        #else: 
        #    raise Exception("Chromium (Cr) gas properties are only available for 1, 10, and 50xSolar")
    
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)   

    #Lodders and Fegley (2003) (cvm)
    rho_p =  7.15
    return gas_mw, gas_mmr, rho_p

    
def Al2O3(mw_atmos, mh = 1, gas_mmr = None):
    """Defines properties for Al2O3 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 2.51e-6
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Gao, Peter, et al. "Aerosol composition of hot giant exoplanets dominated by silicates and hydrocarbon hazes." Nature Astronomy (2020): 1-6.
    .. [2] Lodders, Katharina. "Solar system abundances of the elements." Principles and perspectives in cosmochemistry. Springer, Berlin, Heidelberg, 2010. 379-417.
    .. [3] Morley C.~V., Mukherjee S., Marley M.~S., Fortney J.~J., Visscher C., Lupu R., Gharib-Nezhad E., et al., 2024, ApJ, 975, 59. doi:10.3847/1538-4357/ad71d5
    """     
    #if mh != 1: raise Exception("Alert: No M/H Dependence in Al2O3 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        #gas_mmr = 4.937e-6 #gao 2020 value 
        gas_mmr = 2.489e-6 * mh #diamond back

    gas_mw = 101.961
    #NEW FORSTERITE (from Lodders et al. table, 1000mbar, 1900K)
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)  
    #Lodders and Fegley (1998)
    rho_p =  3.987
    return gas_mw, gas_mmr, rho_p

def Na2S(mw_atmos, mh = 1, gas_mmr = None):
    """Defines properties for Na2S as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 3.97e-6 
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs

    Notes
    -----
    .. [1] Lodders, K. Solar System Abundances of the Elements. Astrophys. Space Sci.
Proc. 16, 379 (2010)
    """     
    #if mh != 1: raise Exception("Alert: No M/H Dependence in Na2S Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 3.34e-6 * mh
    gas_mw = 78.05
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)  
    #Lodders and Fegley (1998)
    rho_p =  1.856
    return gas_mw, gas_mmr, rho_p

def CaTiO3(mw_atmos, mh = 1, gas_mmr=None):
    """Defines properties for CaTiO3 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 2.51e-6
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs
    """     
    #if mh != 1: raise Exception("Alert: No M/H Dependence in CaTiO3 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 2.51e-6 * mh 
    gas_mw = 135.745
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)
    rho_p =  3.987
    return gas_mw, gas_mmr, rho_p

def CaAl12O19(mw_atmos, mh = 1,gas_mmr=None):
    """Defines properties for CaAl12O19 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio
        None points to the default value of : 8.80e-7
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    Mean molecular weight of gas,
    Gas mass mixing ratio 
    Density of gas cgs
    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in CaAl12O19 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 8.80e-7 * mh 
    gas_mw = 667.843
    gas_mmr = gas_mmr * (gas_mw/mw_atmos)
    rho_p =  7.15
    return gas_mw, gas_mmr, rho_p

def SiO2(mw_atmos, mh=1, gas_mmr = None):
    """Defines properties for TiO2 as condensible
    
    Parameters 
    ----------
    mw_atmos : float 
        Mean molecular weight of the atmosphere amu
    gas_mmr : float , optional
        Gas mass mixing ratio.
        None points to the default value of : 1.69e-7
    mh : float , optional
        Metallicity, Default is 1=1xSolar
    
    Returns
    -------
    mean molecular weight of gas,
    gas mass mixing ratio 
    density of gas cgs
    """
    #if mh != 1: raise Exception("Alert: No M/H Dependence in SiO2 Routine. Consult your local theorist to determine next steps.")
    if isinstance(gas_mmr, type(None)):
        gas_mmr = 60.3e-6 * mh
    gas_mw = 60
    gas_mmr = gas_mmr * (gas_mw/mw_atmos) 
    rho_p =  2.648
    return gas_mw, gas_mmr, rho_p