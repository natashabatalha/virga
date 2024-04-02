import numpy as np

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
    """
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for TiO2")

    return 1e6 * 10. ** (9.5489 - 32456.8678/temp)

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
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for Cr")
    mh = np.log10(mh)
    #Cr vapor pressure above cloud 
    pvap_cr_bars = 10.0**(7.49-20592./temp)
    #Then convert from bars to dynes/cm^2    
    pvap_cr = pvap_cr_bars*1e6   
    return pvap_cr

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
    pvap_zns_bars = 10.0**(12.8117-15873./temp - mh)
    #Then convert from bars to dynes/cm^2    
    pvap_zns = pvap_zns_bars*1e6   
    return pvap_zns

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
    .. [1] Lodders, K. \& Fegley, B.\ 1998, The planetary scientist's companion / Katharina Lodders, Bruce Fegley.  New York : Oxford University Press, 1998. QB601 .L84 1998
    """
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for NH3")
    mh = np.log10(mh)
    #NH3 vapor pressure above cloud
    #pvap_nh3 = np.exp(-86596./temp**2 - 2161./temp + 10.53)

    if isinstance(temp,float):
        temp = np.array([temp])
        pvap_nh3 = np.array([0])
        convert_arr = True
    else:
        pvap_nh3 = np.zeros(len(temp))
        convert_arr = False
    tlow = np.where(temp<195.4)[0]
    thigh = np.where(temp>=195.4)[0]
    if len(tlow) > 0: pvap_nh3[tlow] = 10**(6.900 - 1588/temp[tlow])
    if len(thigh) > 0: pvap_nh3[thigh] = 10**(5.201 - 1248/temp[thigh])
    # convert from bars to dyne/cm^2
    pvap_nh3 = pvap_nh3*1e6   
    if convert_arr: 
        pvap_nh3 = pvap_nh3[0]
    return pvap_nh3

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
    .. [1] Morley, C.~V., Fortney, J.~J., Marley, M.~S., Visscher, C., Saumon, D., Leggett, S.~K.\ 2012.\ Neglected Clouds in T and Y Dwarf Atmospheres.\ The Astrophysical Journal 756. doi:10.1088/0004-637X/756/2/172
    .. [2] Lodders, K.\ 1999.\ Alkali Element Chemistry in Cool Dwarf Atmospheres.\ The Astrophysical Journal 519, 793–801. doi:10.1086/307387
    """
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for KCl")
    mh = np.log10(mh)
    pvap_kcl_bars = 10.0**(7.6106 - 11382./temp)
    #Then convert from bars to dynes/cm^2    
    pvap_kcl = pvap_kcl_bars*1e6  
    return pvap_kcl

def H2O(temp,do_buck = True,mh = 1 ):
    """Computes vapor pressure curve
    
    Parameters 
    ----------
    temp : float, ndarray 
        Temperature (K)
    mh : float 
        NON log metallicity relative to solar (1=1Xsolar)
    do_buck : bool 
        True means use Buck 1981 expresssion, False means use 
        Wexler's

    Returns
    -------
    vapor pressure in dyne/cm^2

    Notes 
    -----
    .. [1] Lodders, K. \& Fegley, B.\ 1998, The planetary scientist's companion / Katharina Lodders, Bruce Fegley.  New York : Oxford University Press, 1998. QB601 .L84 1998
    .. [2] Buck, Arden L. "New equations for computing vapor pressure and enhancement factor." Journal of Applied Meteorology and Climatology 20.12 (1981): 1527-1532.
    .. [3] Flatau, Piotr J., Robert L. Walko, and William R. Cotton. "Polynomial fits to saturation vapor pressure." Journal of Applied Meteorology 31.12 (1992): 1507-1513.
    """
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for H2O")
    mh = np.log10(mh)
    if isinstance(temp, float): temp=np.array([temp]) 
    pvap_h2o = np.zeros(len(temp))
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
            tc = temp - 273.16
            pvap_h2o[t_low] = BAI * np.exp( (BBI - tc/BDI)*tc / (tc + BCI) )
        else: 
            pvap_h2o[t_low] = 10*np.exp( 1.0/temp* 
                        (HH0+(HH1+HH5*np.log(temp)+
                        (HH2+(HH3+HH4*temp)*temp)*temp)*temp))

    #saturation vapor pressure over water
    t_med = np.where((temp>=273.16) & (temp<1048))
    if len(temp[t_med])>0:
        if do_buck: 
            tc = temp - 273.16
            pvap_h2o[t_med] = BAL * np.exp( (BBL - tc/BDL)*tc / (tc + BCL) )
        else: 
            pvap_h2o = 10*exp( (1.0/(temp*temp))* 
                    ( GG0+(GG1+(GG2+GG7*np.log(temp)+
                    ( GG3+(GG4+(GG5+GG6*temp)*temp)*temp)*temp)*temp)*temp ) )

    #anything greater than 1048 K is fixed at 600 bars
    t_high = np.where(temp>=1048)
    if len(temp[t_high])>0:
        pvap_h2o[t_high] = 600.0e6

    if len(pvap_h2o) == 1 : pvap_h2o = pvap_h2o[0]
    return pvap_h2o

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
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for Fe")
    mh = np.log10(mh)
    #EXPRESSION from Channon Visscher, correspondance on 6/3/11, added 7/27/11 (cvm)
    pvap_fe = 10.0**(7.09-20995./temp)
    pvap_fe = pvap_fe * 1e6   # convert from bars to dyne/cm^2
    return pvap_fe

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
    .. [1] Lodders, K. \& Fegley, B.\ 1998, The planetary scientist's companion / Katharina Lodders, Bruce Fegley.  New York : Oxford University Press, 1998. QB601 .L84 1998
    """
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for CH4")
    mh = np.log10(mh)

    #     AMR   -- molecular weight / ideal gas constant
    #     TCRIT -- triple point temperature
    #     PCRIT --    "     "   pressure
    #     AS    -- specific heat at constant pressure ( gas - solid )
    #     AL    --    "      "   "     "        "     ( gas - liquid )
    #     ALS   -- latent heat of sublimation
    #     ALV   --   "     "   "  vaporization


    AMR = 16.043 / 8.3143
    TCRIT = 90.68
    PCRIT = .11719 
    AS = 2.213 - 2.650
    AL = 2.213 - 3.370 
    ALS = 611.10
    ALV = 552.36 

    #ic=0: temperature below triple point
    #ic=1: temperature above triple point

    ic = 0
    if temp>TCRIT : ic = 1

    A, B, C = np.zeros(2),np.zeros(2),np.zeros(2)
    C[0] = - AMR * AS
    C[1] = - AMR * AL
    B[0] = - AMR * ( ALS + AS * TCRIT )
    B[1] = - AMR * ( ALV + AL * TCRIT )
    A[0] = PCRIT * TCRIT ** ( -C[0] ) * np.exp( -B[0] / TCRIT )
    A[1] = PCRIT * TCRIT ** ( -C[1] ) * np.exp( -B[1] / TCRIT )

    pvap_ch4 = A[ic] * temp**C[ic] * np.exp( B[ic] / temp )
    pvap_ch4= pvap_ch4*1e6    # convert from bars to dyne/cm^2

    return pvap_ch4

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
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for Al2O3")
    mh = np.log10(mh)
    #Kozasa et al. Ap J. 344 325
    #return np.exp(-73503./temp + 22.01)*1e6
    #calculated from wakeford 2017
    pvap_al2o3 = 1e6 * 10.0 ** (17.7 - 45892.6/temp - 1.66*mh)
    return pvap_al2o3

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
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for CaTiO3")
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
    if mh != 1 : raise Exception("Warning: no M/H Dependence in vapor pressure curve for CaAl12O19")
    mh = np.log10(mh)
    #calculated from wakeford 2017
    pvap_caal12o19 = 1e6 * 10.0 ** (16.46 -44021./temp  - 0.083*np.log10(p/1e6) - 1.67*mh)
    return pvap_caal12o19

def SiO2(temp, mh = 1 ):
    """PLACEHODER
    Not suited yet for public use 
    Computes vapor pressure curve
    
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
    .. [1] MADE UP
    """
    mh = np.log10(mh)
    
    #V1 undepleted gas source
    pvap_sio2 = 10.0**(13.168 - 28265/temp - mh)

    #V1 depleted gas source 
    #13.360 - 28265/T(K) - mh

    #convert bars -> dynes/cm^2
    pvap_sio2 = 1e6 * pvap_sio2 
    return pvap_sio2


