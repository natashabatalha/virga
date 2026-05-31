import numpy as np

def CaTiO3(temp,p,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    p : float
        Pressure (dyne/cm^2)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)
    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Wakeford, Hannah R., et al. "High temperature condensate clouds in super-hot Jupiter atmospheres." Monthly Notices of the Royal Astronomical Society (2016): stw2639.
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for CaTiO3")
    mh = np.log10(mh)
    #calculated from wakeford 2017
    pvap_catio3 = 1e6 * 10.0 ** (-72160./temp + 30.24 - 1*np.log10(p/1e6) - 2*mh) 
    return pvap_catio3

def CaAl12O19(temp,p,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    p : float
        Pressure (dyne/cm^2)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)
    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Wakeford, Hannah R., et al. "High temperature condensate clouds in super-hot Jupiter atmospheres." Monthly Notices of the Royal Astronomical Society (2016): stw2639.
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for CaAl12O19")
    mh = np.log10(mh)
    #calculated from wakeford 2017
    pvap_caal12o19 = 1e6 * 10.0 ** (16.46 -44021./temp  - 0.083*np.log10(p/1e6) - 1.67*mh)
    return pvap_caal12o19

def TiO2(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Marley M.~S., Saumon D., Visscher C., Lupu R., Freedman R., Morley C., Fortney J.~J., et al., 2021, ApJ, 920, 85. doi:10.3847/1538-4357/ac141d
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for TiO2")
    mh = np.log10(mh)
    #return 1e6 * 10. ** (9.5489 - 32456.8678/temp) #Gao 2020 
    return 1e6 * 10. ** (13.95 - 38266/temp - mh) #Marley et al. 2021

def Al2O3(temp,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Wakeford, Hannah R., et al. "High temperature condensate clouds in super-hot Jupiter atmospheres." Monthly Notices of the Royal Astronomical Society (2016): stw2639.
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for Al2O3")
    mh = np.log10(mh)
    #return np.exp(-73503./temp + 22.01)*1e6 #Kozasa et al. Ap J. 344 325
    #calculated from wakeford 2017
    #pvap_al2o3 = 1e6 * 10.0 ** (17.7 - 45892.6/temp - 1.66*mh) #wakeford et al 2017
    pvap_al2o3 = 1e6 * 10.0 ** (15.24 - 41481/temp - 1.66*mh) #diamondback
    return pvap_al2o3

def Fe(temp,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr. "Atmospheric chemistry in giant planets, brown dwarfs, and low-mass dwarf stars. III. Iron, magnesium, and silicon." The Astrophysical Journal 716.2 (2010): 1060.
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for Fe")
    mh = np.log10(mh)
    #EXPRESSION from Channon Visscher, correspondance on 6/3/11, added 7/27/11 (cvm)
    pvap_fe = 10.0**(7.09-20995./temp)
    pvap_fe = pvap_fe * 1e6   # convert from bars to dyne/cm^2
    return pvap_fe

def Mg2SiO4(temp, p, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    p : float, ndarray
        Pressure dyne/cm^2
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr. "Atmospheric chemistry in giant planets, brown dwarfs, and low-mass dwarf stars. III. Iron, magnesium, and silicon." The Astrophysical Journal 716.2 (2010): 1060.
    """
    mh = np.log10(mh)
    #Another new expression from Channon Visscher, correspondance on 10/6/11
    #includes total pressure dependence and met dep. 
    pvap_mg2sio4 = 10.0**(-32488./temp + 14.88 - 0.2*np.log10(p/1e6) 
            - 1.4*mh) * 1e6 #convered from bars to dynes/cm2
    return pvap_mg2sio4

def MgSiO3(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2
    
    Notes
    -----
    .. [1] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr. "Atmospheric chemistry in giant planets, brown dwarfs, and low-mass dwarf stars. III. Iron, magnesium, and silicon." The Astrophysical Journal 716.2 (2010): 1060.
    """
    mh = np.log10(mh)
    #MgSiO3 vapor pressure above cloud
    #the one that is in A&M is this : np.exp(-58663./temp + 25.37)
    #this is a new one from Channon Visscher
    pvap_mgsio3 = 10.0**(13.43 - 28665.0/temp - mh)
    #convert bars -> dynes/cm^2
    pvap_mgsio3 = 1e6 * pvap_mgsio3 
    return pvap_mgsio3

def SiO2(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2
    
    Notes
    -----

    """

    mh = np.log10(mh)

    #VIRGA placeholder expressions:
    pvap_sio2 = 10.0**(13.168 - 28265/temp - mh) # V1 undepleted gas source
    pvap_sio2 = 10.0**(13.360 - 28265/temp - mh) # V1 depleted gas source
    pvap_sio2 = 1e6 * pvap_sio2 # convert bars -> dynes/cm^2
    return pvap_sio2

def Cr(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf atmospheres." The Astrophysical Journal 756.2 (2012): 172.
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for Cr")
    mh = np.log10(mh)
    #Cr vapor pressure above cloud 
    pvap_cr_bars = 10.0**(7.49-20592./temp)
    #Then convert from bars to dynes/cm^2    
    pvap_cr = pvap_cr_bars*1e6   
    return pvap_cr

def MnS(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf atmospheres." The Astrophysical Journal 756.2 (2012): 172.
    .. [2] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr. "Atmospheric chemistry in giant planets, brown dwarfs, and low-mass dwarf stars. II. Sulfur and phosphorus." The Astrophysical Journal 648.2 (2006): 1181.
    """
    mh = np.log10(mh)
    #Mn vapor pressure above cloud 
    pvap_mns_bars = 10.0**(11.5315-23810./temp - mh)
    #Then convert from bars to dynes/cm^2    
    pvap_mns = pvap_mns_bars*1e6 
    return  pvap_mns

def Na2S(temp,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf atmospheres." The Astrophysical Journal 756.2 (2012): 172.
    .. [2] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr. "Atmospheric chemistry in giant planets, brown dwarfs, and low-mass dwarf stars. II. Sulfur and phosphorus." The Astrophysical Journal 648.2 (2006): 1181.
    """
    mh = np.log10(mh)
    #Na vapor pressure above cloud 
    #metallicityMH=0.0
    pvap_na2s_bars = 10.0**(8.5497-13889./temp-0.5*mh)
    #Then convert from bars to dynes/cm^2    
    pvap_na2s = pvap_na2s_bars*1e6  
    return pvap_na2s

def ZnS(temp,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes
    -----
    .. [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf atmospheres." The Astrophysical Journal 756.2 (2012): 172.
    .. [2] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr. "Atmospheric chemistry in giant planets, brown dwarfs, and low-mass dwarf stars. II. Sulfur and phosphorus." The Astrophysical Journal 648.2 (2006): 1181.
    """
    mh = np.log10(mh)
    #Zn vapor pressure above cloud 
    #pvap_zns_bars = 10.0**(12.8117-15873./temp - mh)
    #Then convert from bars to dynes/cm^2    
    # pvap_zns = pvap_zns_bars*1e6

    #Elspeth 5 polynomial Barin data fit (ZnS -> ZnS[s]), from mini_cloud
    pvap_zns = np.exp(-4.75507888e4/temp + 3.66993865e1 - 2.49490016e-3*temp
            + 7.29116854e-7*temp**2 - 1.12734453e-10*temp**3)

    return pvap_zns

def KCl(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes 
    -----
    .. [1] Morley, C.~V., Fortney, J.~J., Marley, M.~S., Visscher, C., Saumon, D., Leggett, S.~K. 2012. Neglected Clouds in T and Y Dwarf Atmospheres. The Astrophysical Journal 756. doi:10.1088/0004-637X/756/2/172
    .. [2] Lodders, K. 1999. Alkali Element Chemistry in Cool Dwarf Atmospheres. The Astrophysical Journal 519, 793–801. doi:10.1086/307387
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for KCl")
    mh = np.log10(mh)
    pvap_kcl_bars = 10.0**(7.6106 - 11382./temp)
    #Then convert from bars to dynes/cm^2    
    pvap_kcl = pvap_kcl_bars*1e6  
    return pvap_kcl

def H2O(temp,do_buck = True,mh = 1, do_murphy = True):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)
    do_buck : bool 
        True means use Buck 1981 expresssion, False means use 
        Wexler's. Only used if do_murphy is False.
    do_murphy : bool
        True means use Murphy & Koop (2005) expression from mini_cloud.
        Default is True.

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes 
    -----
    .. [1] Lodders, K. & Fegley, B. 1998, The planetary scientist's companion / Katharina Lodders, Bruce Fegley.  New York : Oxford University Press, 1998. QB601 .L84 1998
    .. [2] Buck, Arden L. "New equations for computing vapor pressure and enhancement factor." Journal of Applied Meteorology and Climatology 20.12 (1981): 1527-1532.
    .. [3] Flatau, Piotr J., Robert L. Walko, and William R. Cotton. "Polynomial fits to saturation vapor pressure." Journal of Applied Meteorology 31.12 (1992): 1507-1513.
    .. [4] Murphy, D. M. & Koop, T. 2005. Review of the vapour pressures of ice and supercooled water for atmospheric applications. Quarterly Journal of the Royal Meteorological Society 131, 1539-1565. doi:10.1256/qj.04.94
    """
    # if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for H2O")
    mh = np.log10(mh)
    temp = np.asarray(temp, dtype=float)
    if temp.ndim == 0: temp = np.array([temp.item()])
    pvap_h2o = np.zeros(len(temp))

    if do_murphy:
        t_ice = np.where(temp <= 273.16)
        t_liq = np.where((temp > 273.16) & (temp < 1048))
        t_high = np.where(temp >= 1048.0)
        if len(temp[t_ice]) > 0:
            t = temp[t_ice]
            ln_p = 9.550426 - 5723.265/t + 3.53068*np.log(t) - 0.00728332*t
            pvap_h2o[t_ice] = np.exp(ln_p) * 10.0
        if len(temp[t_liq]) > 0:
            t = temp[t_liq]
            ln_p = (54.842763 - 6763.22/t - 4.210*np.log(t) + 0.000367*t
                    + np.tanh(0.0415*(t - 218.8))
                    * (53.878 - 1331.22/t - 9.44523*np.log(t) + 0.014025*t))
            pvap_h2o[t_liq] = np.exp(ln_p) * 10.0
        if len(temp[t_high]) > 0:
            t = 1048.0
            ln_p = (54.842763 - 6763.22/t - 4.210*np.log(t) + 0.000367*t
                    + np.tanh(0.0415*(t - 218.8))
                    * (53.878 - 1331.22/t - 9.44523*np.log(t) + 0.014025*t))
            pvap_h2o[t_high] = np.exp(ln_p) * 10.0            
        if len(pvap_h2o) == 1 : pvap_h2o = pvap_h2o[0]
        return pvap_h2o

    #define constants used in Buck's expressions
    #Buck, 1981 (J. Atmos. Sci., 20, p. 1527)
    BAL = 6.1121e3 
    BBL = 18.729 
    BCL = 257.87 
    BDL = 227.3 
    BAI = 6.1115e3 
    BBI = 23.036 
    BCI = 279.82 
    BDI = 333.7 

    #define constants used in Wexler formulas
    #(see Flatau et al., 1992, J. Appl. Meteor. p. 1507)

    GG0 =-0.29912729e4
    GG1 =-0.60170128e4
    GG2 = 0.1887643854e2
    GG3 =-0.28354721e-1
    GG4 = 0.17838301e-4
    GG5 =-0.84150417e-9
    GG6 = 0.44412543e-12
    GG7 = 0.28584870e1

    HH0 = -0.58653696e4
    HH1 =  0.2224103300e2
    HH2 =  0.13749042e-1
    HH3 = -0.34031775e-4
    HH4 =  0.26967687e-7
    HH5 =  0.6918651

    t_low = np.where( temp < 273.16 )

    #Branch on temperature for liquid or ice
    if len(temp[t_low])>0:
        if do_buck:
            tc = temp[t_low] - 273.16
            pvap_h2o[t_low] = BAI * np.exp( (BBI - tc/BDI)*tc / (tc + BCI) )
        else: 
            t = temp[t_low]
            pvap_h2o[t_low] = 10*np.exp( 1.0/t* 
                        (HH0+(HH1+HH5*np.log(t)+
                        (HH2+(HH3+HH4*t)*t)*t)*t))

    #saturation vapor pressure over water
    t_med = np.where((temp>=273.16) & (temp<1048))
    if len(temp[t_med])>0:
        if do_buck: 
            tc = temp[t_med] - 273.16
            pvap_h2o[t_med] = BAL * np.exp( (BBL - tc/BDL)*tc / (tc + BCL) )
        else: 
            t = temp[t_med]
            pvap_h2o[t_med] = 10*np.exp( (1.0/(t*t))* 
                    ( GG0+(GG1+(GG2+GG7*np.log(t)+
                    ( GG3+(GG4+(GG5+GG6*t)*t)*t)*t)*t)*t ) )

    #anything greater than 1048 K is fixed at 600 bars
    t_high = np.where(temp>=1048)
    if len(temp[t_high])>0:
        pvap_h2o[t_high] = 600.0e6

    if len(pvap_h2o) == 1 : pvap_h2o = pvap_h2o[0]
    return pvap_h2o

def NH3(temp, mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes 
    -----
    .. [1] Lodders, K. & Fegley, B. 1998, The planetary scientist's companion / Katharina Lodders, Bruce Fegley.  New York : Oxford University Press, 1998. QB601 .L84 1998
    .. [2] Fray, N. & Schmitt, B. 2009. Sublimation of ices of astrophysical interest: A bibliographic review. Planetary and Space Science 57, 2053-2080. doi:10.1016/j.pss.2009.09.011
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for NH3")
    mh = np.log10(mh)

    temp = np.asarray(temp, dtype=float)
    if temp.ndim == 0: temp = np.array([temp.item()])

    #Fray & Schmitt (2009), from mini_cloud
    pvap_nh3 = np.exp(15.96 - 3537.0/temp - 3.310e4/temp**2
            + 1.742e6/temp**3 - 2.995e7/temp**4) * 1e6 # convert from bars to dyne/cm^2

    #Lodders & Fegley / Virga piecewise expression:
    #pvap_nh3 = np.zeros(len(temp))
    #tlow = np.where(temp<195.4)[0]
    #thigh = np.where(temp>=195.4)[0]
    #if len(tlow) > 0: pvap_nh3[tlow] = 10**(6.900 - 1588/temp[tlow])
    #if len(thigh) > 0: pvap_nh3[thigh] = 10**(5.201 - 1248/temp[thigh])
    #pvap_nh3 = pvap_nh3*1e6 # convert from bars to dyne/cm^2

    if len(pvap_nh3) == 1 : pvap_nh3 = pvap_nh3[0]
    return pvap_nh3

def CH4(temp,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes 
    -----
    .. [1] Lodders, K. & Fegley, B. 1998, The planetary scientist's companion / Katharina Lodders, Bruce Fegley.  New York : Oxford University Press, 1998. QB601 .L84 1998
    .. [2] Fray, N. & Schmitt, B. 2009. Sublimation of ices of astrophysical interest: A bibliographic review. Planetary and Space Science 57, 2053-2080. doi:10.1016/j.pss.2009.09.011
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for CH4")
    mh = np.log10(mh)

    #Fray & Schmitt (2009), from mini_cloud
    pvap_ch4 = np.exp(1.051e1 - 1.110e3/temp - 4.341e3/temp**2
            + 1.035e5/temp**3 - 7.910e5/temp**4) * 1e6 # convert from bars to dyne/cm^2

    #Lodders & Fegley (1998) expression:
    #AMR = 16.043 / 8.3143
    #TCRIT = 90.68
    #PCRIT = .11719 
    #AS = 2.213 - 2.650
    #AL = 2.213 - 3.370 
    #ALS = 611.10
    #ALV = 552.36
    #ic = 0
    #if temp>TCRIT : ic = 1
    #A, B, C = np.zeros(2),np.zeros(2),np.zeros(2)
    #C[0] = - AMR * AS
    #C[1] = - AMR * AL
    #B[0] = - AMR * ( ALS + AS * TCRIT )
    #B[1] = - AMR * ( ALV + AL * TCRIT )
    #A[0] = PCRIT * TCRIT ** ( -C[0] ) * np.exp( -B[0] / TCRIT )
    #A[1] = PCRIT * TCRIT ** ( -C[1] ) * np.exp( -B[1] / TCRIT )
    #pvap_ch4 = A[ic] * temp**C[ic] * np.exp( B[ic] / temp )
    #pvap_ch4 = pvap_ch4*1e6 # convert from bars to dyne/cm^2

    return pvap_ch4

def H2S(temp,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes 
    -----
    .. [1] Fray, N. & Schmitt, B. 2009. Sublimation of ices of astrophysical interest: A bibliographic review. Planetary and Space Science 57, 2053-2080. doi:10.1016/j.pss.2009.09.011
    """
    #if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for H2S")
    mh = np.log10(mh)
    pvap_h2s = np.exp(12.98 - 2.707e3/temp) * 1e6 # convert from bars to dyne/cm^2
    return pvap_h2s
