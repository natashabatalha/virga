import gas_properties
import astropy.constants as c
import astropy.units as u
import pandas as pd
import numpy as np
import pvaps
import os
from scipy import optimize 
from root_functions import advdiff, vfall,vfall_find_root,qvs_below_model
import PyMieScatt as ps
from calc_mie import fort_mie_calc, calc_new_mieff

def justdoit(atmo, directory = None, do_optics=False, fort_calc_mie=False, rmin = 1e-5, nradii = 40):
    """
    Top level program to run eddysed. Requires running `Atmosphere` class 
    before running this. 

    atmo : class 
        `Atmosphere` class 
    directory : str, optional 
        Directory string that describes where refrind files are 
    do_optics : bool, optional
        If True, computes mie optical properties 
    fort_calc_mie : bool, optional 
        If True uses the original converted fortran to python version of the Mie code. 
        If False, uses the python module PyMieScatt, which is MUCHH faster. 
        Default is False.
    rmin : float 
        Minimum particle radius size in cm, Default = 1e-5 cm
    nradii : int
        Number of radii for which to compute mie properties. Default=40.
    """

    mmw = atmo.mmw
    mh = atmo.mh
    condensibles = atmo.condensibles

    ngas = len(condensibles)

    gas_mw = np.zeros(ngas)
    gas_mmr = np.zeros(ngas)
    rho_p = np.zeros(ngas)
    
    #### First we need to either grab or compute Mie coefficients #### 
    for i, igas in zip(range(ngas),condensibles) : 

        #Get gas properties including gas mean molecular weight,
        #gas mixing ratio, and the density
        run_gas = getattr(gas_properties, igas)
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh)
        
        #If do optics is true, then compute the Mie from scratch
        #files will be saved in `directory`
        if do_optics:
            # obtaining refractive index data for each gas
            wave_in,nn,kk = get_refrind(igas,directory)
            nwave = len(wave_in)

            #Setup up a particle size grid on first run and calculate single-particle scattering
            if i == 0:
                radius, rup, dr = get_r_grid(rmin, nradii)
                qext=np.zeros(shape=(nwave,nradii,ngas))
                qscat = np.zeros(shape=(nwave,nradii,ngas))
                cos_qscat=np.zeros(shape=(nwave,nradii,ngas))

            #get extinction, scattering, and asymmetry
            #all of these are  [nwave by nradii]
            qext_gas, qscat_gas, cos_qscat_gas = calc_new_mieff(wave_in, nn,kk, radius, rup, fort_calc_mie = False)
        else: 
            #Otherwise, get mie files that are already saved in 
            #directory
            #eventually we will replace this with nice database 
            qext_gas, qscat_gas, cos_qscat_gas, nwave, radius = get_mie(igas,directory)
            radius, rup, dr = get_r_grid(rmin, nradii)

            if i==0: 
                nradii = len(radius)
                qext = np.zeros((nwave,nradii,ngas))
                qscat = np.zeros((nwave,nradii,ngas))
                cos_qscat = np.zeros((nwave,nradii,ngas))

        #add to master matrix that contains the per gas Mie stuff
        qext[:,:,i], qscat[:,:,i], cos_qscat[:,:,i] = qext_gas, qscat_gas, cos_qscat_gas

    #Next, calculate size and concentration
    #of condensates in balance between eddy diffusion and sedimentation

    #qc = condensate mixing ratio, qt = condensate+gas mr, rg = mean radius,
    #reff = droplet eff radius, ndz = column dens of condensate, 
    #qc_path = vertical path of condensate

    qc, qt, rg, reff, ndz, qc_path = eddysed(atmo.t_top, atmo.p_top, atmo.t, atmo.p, 
        condensibles, gas_mw, gas_mmr, rho_p , mmw, atmo.g, atmo.kz, atmo.fsed, mh)

    #Finally, calculate spectrally-resolved profiles of optical depth, single-scattering
    #albedo, and asymmetry parameter.    
    opd, w0, g0, opd_gas = calc_optics(nwave, qc, qt, rg, reff, ndz,radius,dr,qext, qscat,cos_qscat)

    return opd, w0, g0, opd_gas

def calc_optics(nwave, qc, qt, rg, reff, ndz,radius,dr,qext, qscat,cos_qscat):
    """
    Calculate spectrally-resolved profiles of optical depth, single-scattering
    albedo, and asymmetry parameter.

    Parameters
    ----------
    nwave : int 
        Number of wave points 
    qc : ndarray
        Condensate mixing ratio 
    qt : ndarray 
        Gas + condensate mixing ratio 
    rg : ndarray 
        Geometric mean radius of condensate 
    reff : ndarray
        Effective (area-weighted) radius of condensate (cm)
    ndz : ndarray
        Column of particle concentration in layer (#/cm^2)
    radius : ndarray
        Radius bin centers (cm)
    dr : ndarray
        Width of radius bins (cm)
    qscat : 
        Scattering efficiency
    qext : ndarray
        scattering efficiency
    cos_qscat : 
        qscat-weighted <cos (scattering angle)>

    Returns
    -------
    opd : ndarray 
        extinction optical depth due to all condensates in layer
    w0 : ndarray 
        single scattering albedo
    g0 : ndarray 
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    opd_gas : ndarray
        cumulative (from top) opd by condensing vapor as geometric conservative scatterers
    """

    PI=np.pi
    nz = qc.shape[0]
    ngas = qc.shape[1]
    nrad = len(radius)

    opd_layer = np.zeros((nz, ngas))
    scat_gas = np.zeros((nz,nwave,ngas))
    ext_gas = np.zeros((nz,nwave,ngas))
    cqs_gas = np.zeros((nz,nwave,ngas))
    opd = np.zeros((nz,nwave))
    opd_gas = np.zeros((nz,ngas))
    w0 = np.zeros((nz,nwave))
    g0 = np.zeros((nz,nwave))

    for iz in range(nz):
        for igas in range(ngas):
            # Optical depth for conservative geometric scatterers 
            if ndz[iz,igas] > 0:
                r2 = rg[iz,igas]**2 * np.exp( 2*np.log( sig_all )**2 )
                opd_layer[iz,igas] = 2.*PI*r2*ndz[iz,igas]

                #  Calculate normalization factor (forces lognormal sum = 1.0)
                rsig = sig_all
                norm = 0.
                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / ( np.sqrt(2.*PI)*rr*np.log(rsig) )
                    arg2 = -np.log( rr/rg[iz,igas] )**2 / ( 2*np.log(rsig)**2 )
                    norm = norm + arg1*np.exp( arg2 )
                # normalization 

                norm = ndz[iz,igas] / norm

                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / ( np.sqrt(2.*PI)*np.log(rsig) )
                    arg2 = -np.log( rr/rg[iz,igas] )**2 / ( 2*np.log(rsig)**2 )
                    pir2ndz = norm*PI*rr*arg1*np.exp( arg2 )                    

                    for iwave in range(nwave): 
                        scat_gas[iz,iwave,igas] = (scat_gas[iz,iwave,igas] + 
                                                    qscat[iwave,irad,igas]*pir2ndz)
                        ext_gas[iz,iwave,igas] = (ext_gas[iz,iwave,igas] + 
                                                    qext[iwave,irad,igas]*pir2ndz)
                        cqs_gas[iz,iwave,igas] = (cqs_gas[iz,iwave,igas] + 
                                                    cos_qscat[iwave,irad,igas]*pir2ndz)

                    #TO DO ADD IN CLOUD SUBLAYER KLUGE LATER 

    #Sum over gases and compute spectral optical depth profile etc
    for iz in range(nz):
        for iwave in range(nwave): 
            opd_scat = 0.
            opd_ext = 0.
            cos_qs = 0.
            for igas in range(ngas):
                opd_scat = opd_scat + scat_gas[iz,iwave,igas]
                opd_ext = opd_ext + ext_gas[iz,iwave,igas]
                cos_qs = cos_qs + cqs_gas[iz,iwave,igas]

                if( opd_scat > 0. ):
                    opd[iz,iwave] = opd_ext
                    w0[iz,iwave] = opd_scat / opd_ext
                    g0[iz,iwave] = cos_qs / opd_scat
                    
    #cumulative optical depths for conservative geometric scatterers
    opd_tot = 0.

    for igas in range(ngas):
        opd_gas[0,igas] = opd_layer[0,igas]

        for iz in range(1,nz):
            opd_gas[iz,igas] = opd_gas[iz-1,igas] + opd_layer[iz,igas]

    return opd, w0, g0, opd_gas

def eddysed(t_top, p_top,t_mid, p_mid, condensibles, gas_mw, gas_mmr,rho_p,
    mw_atmos,gravity, kz,fsed, mh, do_virtual=True, supsat=0):
    """
    Given an atmosphere and condensates, calculate size and concentration
    of condensates in balance between eddy diffusion and sedimentation.

    Parameters
    ----------
    t_top : ndarray
        Temperature at each layer (K)
    p_top : ndarray
        Pressure at each layer (dyn/cm^2)
    t_mid : ndarray
        Temperature at each midpoint (K)
    p_mid : ndarray 
        Pressure at each midpoint (dyn/cm^2)
    condensibles : ndarray or list of str
        List or array of condensible gas names
    gas_mw : ndarray
        Array of gas mean molecular weight from `gas_properties`
    gas_mmr : ndarray 
        Array of gas mixing ratio from `gas_properties`
    rho_p : float 
        density of condensed vapor (g/cm^3)
    mw_atmos : float 
        Mean molecular weight of the atmosphere
    gravity : float 
        Gravity of planet cgs
    kz : float or ndarray
        Kzz in cgs, either float or ndarray depending of whether or not 
        it is set as input
    fsed : float 
        Sedimentation efficiency, unitless
    mh : float 
        Atmospheric metallicity in NON log units (e.g. 1 for 1x solar)
    do_virtual : bool,optional 
        include decrease in condensate mixing ratio below model domain
    supsat : float, optional
        Default = 0 , Saturation factor (after condensation)

    Returns
    -------
    qc : ndarray 
        condenstate mixing ratio (g/g)
    qt : ndarray 
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  cm 
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray 
        number column density of condensate (cm^-3)
    qc_path : ndarray 
        vertical path of condensate 
    """
    t_bot = t_top[-1]
    p_bot = p_top[-1]
    ngas =  len(condensibles)
    nz = len(t_mid)
    qc = np.zeros((nz,ngas))
    qt  = np.zeros((nz, ngas))
    rg = np.zeros((nz, ngas))
    reff = np.zeros((nz, ngas))
    ndz = np.zeros((nz, ngas))
    qc_path = np.zeros(ngas)

    for i, igas in zip(range(ngas), condensibles):

        q_below = gas_mmr[i]


        #include decrease in condensate mixing ratio below model domain
        if do_virtual: 
            qvs_factor = (supsat+1)*gas_mw[i]/mw_atmos
            get_pvap = getattr(pvaps, igas)
            if igas == 'Mg2SiO4':
                pvap = get_pvap(t_bot, p_bot, mh=np.log10(mh))
            else:
                pvap = get_pvap(t_bot, mh=np.log10(mh))

            qvs = qvs_factor*pvap/p_bot   

            print('q_below',q_below)
            print('qvs',qvs)

            if qvs <= q_below :   
                print('ENTERING VIRTUAL',igas) 
                #find the pressure at cloud base 
                #   parameters for finding root 
                p_lo = p_bot
                p_hi = p_bot * 1e2

                #temperature gradient 
                dtdlnp = ( t_top[-2] - t_bot ) / np.log10( p_bot/p_top[-2] )

                #   load parameters into qvs_below common block
                print(dtdlnp,t_bot,p_bot,pvap)

                qv_dtdlnp = dtdlnp
                qv_p = p_bot
                qv_t = t_bot
                qv_gas_name = igas
                qv_factor = qvs_factor

                p_base = optimize.root_scalar(qvs_below_model, 
                    bracket=[p_lo, p_hi], method='brentq', 
                args=(qv_dtdlnp,qv_p, qv_t,qv_factor ,qv_gas_name,mh,q_below))
                p_base = p_base.root 
                t_base = t_bot + np.log10( p_bot/p_base )*dtdlnp
                
                print(p_base,t_base)

                #   Calculate temperature and pressure below bottom layer
                #   by adding a virtual layer 

                p_layer = 0.5*( p_bot + p_base )
                t_layer = t_bot + np.log10( p_bot/p_layer )*dtdlnp

                #we just need to overwrite 
                #q_below from this output for the next routine
                qc, qt, rg, reff,ndz,q_below = layer( igas, rho_p[i], t_mid[iz], p_mid[iz], 
                    t_top[iz],t_top[iz+1], p_top[iz], p_top[iz+1],
                     kz_in, gravity, mw_atmos, gas_mw[i], q_below, supsat, fsed
                 )

        for iz in range(nz-1,-1,-1): #goes from BOA to TOA
            if not isinstance(kz,(float,int)): 
                kz_in = kz[iz]
            else: 
                kz_in = kz

            qc[iz,i], qt[iz,i], rg[iz,i], reff[iz,i],ndz[iz,i],q_below = layer( igas, rho_p[i], t_mid[iz], p_mid[iz], 
                t_top[iz],t_top[iz+1], p_top[iz], p_top[iz+1],
                 kz_in, gravity, mw_atmos, gas_mw[i], q_below, supsat, fsed
             )

            qc_path[i] = (qc_path[i] + qc[iz,i]*
                            ( p_top[iz+1] - p_top[iz] ) / gravity)
 
    return qc, qt, rg, reff, ndz, qc_path

def layer(gas_name,rho_p, t_layer, p_layer, t_top, t_bot, p_top, p_bot,
    kz, gravity, mw_atmos, gas_mw, q_below, supsat, fsed):
    """
    Calculate layer condensate properties by iterating on optical depth
    in one model layer (convering on optical depth over sublayers)

    gas_name : str 
        Name of condenstante 
    rho_p : float 
        density of condensed vapor (g/cm^3)
    t_layer : float 
        Temperature of layer mid-pt (K)
    p_layer : float 
        Pressure of layer mid-pt (dyne/cm^2)
    t_top : float 
        Temperature at top of layer (K)
    t_bot : float 
        Temperature at botton of layer (K)
    p_top : float 
        Pressure at top of layer (dyne/cm2)
    p_bot : float 
        Pressure at botton of layer 
    kz : float 
        eddy diffusion coefficient (cm^2/s)
    gravity : float 
        Gravity of planet cgs 
    mw_atmos : float 
        Molecular weight of the atmosphere 
    gas_mw : float 
        Gas molecular weight 
    q_below : float 
        total mixing ratio (vapor+condensate) below layer (g/g)
    supsat : float 
        Super saturation factor
    fsed : float
        Sedimentation efficiency (unitless) 

    Returns
    -------
    qc_layer : ndarray 
        condenstate mixing ratio (g/g)
    qt_layer : ndarray 
        gas + condensate mixing ratio (g/g)
    rg_layer : ndarray
        geometric mean radius of condensate  cm 
    reff_layer : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : ndarray 
        number column density of condensate (cm^-3)
    q_below : ndarray 
        total mixing ratio (vapor+condensate) below layer (g/g)
    """
    #   universal gas constant (erg/mol/K)

    R_GAS = 8.3143e7
    AVOGADRO = 6.02e23
    K_BOLTZ = R_GAS / AVOGADRO
    PI = np.pi 
    #   Number of levels of grid refinement used 
    nsub = 1

    #   diameter of atmospheric molecule (cm) (Rosner, 2000)
    #   (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
    d_molecule = 2.827e-8

    #   Depth of the Lennard-Jones potential well for the atmosphere 
    # Used in the viscocity calculation (units are K) (Rosner, 2000)
    #   (78.6 for air, 71.4 for N2, 59.7 for H2)
    eps_k = 59.7

    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos

    #specific gas constant for cloud (erg/K/g)
    r_cloud = R_GAS/ gas_mw

    #   specific heat of atmosphere (erg/K/g)
    c_p = 7./2. * r_atmos

    #   pressure thickness of layer
    dp_layer = p_bot - p_top
    dlnp = np.log( p_bot/p_top )

    #   temperature gradient 
    dtdlnp = ( t_top - t_bot ) / dlnp
    lapse_ratio = ( t_bot - t_top ) / dlnp / ( 2./7.*t_layer )

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / ( r_atmos * t_layer )

    #   atmospheric scale height (cm)
    scale_h = r_atmos * t_layer / gravity    

    #   convective mixing length scale (cm): no less than 1/10 scale height
    # Eqn. 6 in A & M 01 
    mixl = np.max( [0.10, lapse_ratio ]) * scale_h


    #   scale factor for eddy diffusion: 1/3 is baseline
    scalef_kz = 1./3.

    #   convective velocity scale (cm/s) from mixing length theory
    w_convect = kz / mixl 

    #   atmospheric number density (molecules/cm^3)
    n_atmos = p_layer / ( K_BOLTZ*t_layer )

    #   atmospheric mean free path (cm)
    mfp = 1. / ( np.sqrt(2.)*n_atmos*PI*d_molecule**2 )

    # atmospheric viscosity (dyne s/cm^2)
    # EQN B2 in A & M 2001, originally from Rosner+2000
    # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
    visc = (5./16.*np.sqrt( PI*K_BOLTZ*t_layer*(mw_atmos/AVOGADRO)) /
        ( PI*d_molecule**2 ) /
        ( 1.22 * ( t_layer / eps_k )**(-0.16) ))

    #   --------------------------------------------------------------------
    #   Top of convergence loop    
    converge = False
    while not converge: 

        #   Zero cumulative values
        qc_layer = 0.
        qt_layer = 0.
        ndz_layer = 0.
        opd_layer = 0.        

        #   total mixing ratio and pressure at bottom of sub-layer

        qt_bot_sub = q_below
        p_bot_sub = p_bot

        #SUBALYER 
        dp_sub = dp_layer / nsub

        for isub in range(nsub): 
            qt_below = qt_bot_sub
            p_top_sub = p_bot_sub - dp_sub
            dz_sub = scale_h * np.log( p_bot_sub/p_top_sub )
            #print('dz',scale_h , p_bot_sub,p_top_sub )
            p_sub = 0.5*( p_bot_sub + p_top_sub )
            t_sub = t_bot + np.log( p_bot/p_sub )*dtdlnp

            qt_top, qc_sub, qt_sub, rg_sub, reff_sub,ndz_sub= calc_qc(
                    gas_name, supsat, t_sub, p_sub,r_atmos, r_cloud,
                        qt_below, mixl, dz_sub, gravity,mw_atmos,mfp,visc,
                        rho_p,w_convect,fsed)


            #   vertical sums
            qc_layer = qc_layer + qc_sub*dp_sub/gravity
            qt_layer = qt_layer + qt_sub*dp_sub/gravity
            ndz_layer = ndz_layer + ndz_sub

            if reff_sub > 0.:
                opd_layer = (opd_layer + 
                                    1.5*qc_sub*dp_sub/gravity/(rho_p*reff_sub))
    
            #   Increment values at bottom of sub-layer

            qt_bot_sub = qt_top
            p_bot_sub = p_top_sub

        #    Check convergence on optical depth
        if nsub_max == 1 :
            converge = True
        elif  nsub == 1 : 
            opd_test = opd_layer
        elif (opd_layer == 0.) or (nsub >= nsub_max): 
            converge = True
        elif ( abs( 1. - opd_test/opd_layer ) <= 1e-2 ) : 
            converge = True
        else: 
            opd_test = opd_layer
        
        nsub = nsub * 2
    #   Update properties at bottom of next layer

    q_below = qt_top

    #Get layer averages

    if opd_layer > 0. : 
        reff_layer = 1.5*qc_layer / (rho_p*opd_layer)
        lnsig2 = 0.5*np.log( sig_all )**2
        rg_layer = reff_layer*np.exp( -5*lnsig2 )
    else : 
        reff_layer = 0.
        rg_layer = 0.

    qc_layer = qc_layer*gravity / dp_layer
    qt_layer = qt_layer*gravity / dp_layer

    return qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer,q_below

def calc_qc(gas_name, supsat, t_layer, p_layer
    ,r_atmos, r_cloud, q_below, mixl, dz_layer, gravity,mw_atmos
    ,mfp,visc,rho_p,w_convect, fsed):
    """
    Calculate condensate optical depth and effective radius for a layer,
    assuming geometric scatterers. 

    gas_name : str 
        Name of condenstante 
    supsat : float 
        Super saturation factor 
    t_layer : float 
        Temperature of layer mid-pt (K)
    p_layer : float 
        Pressure of layer mid-pt (dyne/cm^2)
    r_atmos : float 
        specific gas constant for atmosphere (erg/K/g)
    r_cloud : float 
        specific gas constant for cloud species (erg/K/g)     
    q_below : float 
        total mixing ratio (vapor+condensate) below layer (g/g)
    mxl : float 
        convective mixing length scale (cm): no less than 1/10 scale height
    dz_layer : float 
        Altitude of layer cm 
    gravity : float 
        Gravity of planet cgs 
    mw_atmos : float 
        Molecular weight of the atmosphere 
    mfp : float 
        atmospheric mean free path (cm)
    visc : float 
        atmospheric viscosity (dyne s/cm^2)
    rho_p : float 
        density of condensed vapor (g/cm^3)
    w_convect : float    
        convective velocity scale (cm/s)
    fsed : float 
        Sedimentation efficiency (unitless)

    Returns
    -------
    qt_top : float 
        gas + condensate mixing ratio at top of layer(g/g)
    qc_layer : float 
        condenstate mixing ratio (g/g)
    qt_layer : float 
        gas + condensate mixing ratio (g/g)
    rg_layer : float
        geometric mean radius of condensate  cm 
    reff_layer : float
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : float 
        number column density of condensate (cm^-3)
    """

    get_pvap = getattr(pvaps, gas_name)
    if gas_name == 'Mg2SiO4':
        pvap = get_pvap(t_layer, p_layer)
    else:
        pvap = get_pvap(t_layer)

    fs = supsat + 1 


    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / ( r_atmos * t_layer )    

    #   mass mixing ratio of saturated vapor (g/g)
    qvs = fs*pvap / ( (r_cloud) * t_layer ) / rho_atmos
    
    #   --------------------------------------------------------------------
    #   Layer is cloud free

    if( q_below < qvs ):

        qt_layer = q_below
        qt_top   = q_below
        qc_layer = 0.
        rg_layer = 0.
        reff_layer = 0.
        ndz_layer = 0.

    else:

        #   --------------------------------------------------------------------
        #   Cloudy layer: first calculate qt and qc at top of layer,
        #   then calculate layer averages

        #   range of mixing ratios to search (g/g)
        qhi = q_below
        qlo = qhi / 1e3

        #   precision of advective-diffusive solution (g/g)
        #delta_q = q_below / 1000.

        #   load parameters into advdiff common block

        ad_qbelow = q_below
        ad_qvs = qvs
        ad_mixl = mixl
        ad_dz = dz_layer
        ad_rainf = fsed

        #   Find total vapor mixing ratio at top of layer
        qt_top = optimize.root_scalar(advdiff, bracket=[qlo, qhi], method='brentq', 
                args=(ad_qbelow,ad_qvs, ad_mixl,ad_dz ,ad_rainf))


        qt_top = qt_top.root

        #   Use trapezoid rule (for now) to calculate layer averages
        #   -- should integrate exponential
        qt_layer = 0.5*( q_below + qt_top )


        #   Find total condensate mixing ratio
        qc_layer = np.max( [0., qt_layer - qvs] )

        #   --------------------------------------------------------------------
        #   Find <rw> corresponding to <w_convect> using function vfall()

        #   range of particle radii to search (cm)
        rlo = 1.e-10
        rhi = 10.

        #   precision of vfall solution (cm/s)
        rw_layer = optimize.root_scalar(vfall_find_root, bracket=[rlo, rhi], method='brentq', 
                args=(gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p,w_convect))

        #fall velocity particle radius 
        rw_layer = rw_layer.root
        
        #   geometric std dev of lognormal size distribution
        lnsig2 = 0.5*np.log( sig_all )**2
        #   sigma floor for the purpose of alpha calculation
        sig_alpha = np.max( [1.1, sig_all] )    

        if fsed > 1 :

            #   Bulk of precip at r > rw: exponent between rw and rw*sig
            alpha = (np.log(
                            vfall( rw_layer*sig_alpha,gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p ) 
                            / w_convect )
                                / np.log( sig_alpha ))

        else:

            #   Bulk of precip at r < rw: exponent between rw/sig and rw
            alpha = (np.log(
                            w_convect / vfall( rw_layer/sig_alpha,gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p) )
                                / np.log( sig_alpha ))

        #     EQN. 13 A&M 
        #   geometric mean radius of lognormal size distribution
        rg_layer = (fsed**(1./alpha) *
                    rw_layer * np.exp( -(alpha+6)*lnsig2 ))

        #   droplet effective radius (cm)
        reff_layer = rg_layer*np.exp( 5*lnsig2 )

        #      EQN. 14 A&M
        #   column droplet number concentration (cm^-2)
        ndz_layer = (3*rho_atmos*qc_layer*dz_layer /
                    ( 4*np.pi*rho_p*rg_layer**3 ) * np.exp( -9*lnsig2 ))

    return qt_top, qc_layer,qt_layer, rg_layer,reff_layer,ndz_layer 

class Atmosphere():
    def __init__(self,condensibles,fsed = 0.5, mh=1,mmw=2.2) :
        """
        Parameters
        ----------
        condensibles : list of str
            list of gases for which to consider as cloud species 
        fsed : float 
            Sedimentation efficiency. Jupiter ~3-6. Hot Jupiters ~ 0.1-1.
        mh : float 
            metalicity 
        mmw : float 
            MMW of the atmosphere 
    
        """
        self.mh = mh
        self.mmw = mmw
        self.condensibles = condensibles
        self.fsed = fsed

    def get_pt(self, df = None, filename=None,kz_min=1e5, **pd_kwargs):
        """
        Read in file or define dataframe. 
    
        Parameters
        ----------
        df : dataframe or dict
            Dataframe with "pressure"(bars),"temperature"(K). Should have at least two 
            columns with names "pressure" and "temperature". Can also include 'kz' in CGS units. 
        filename : str 
            Filename read in. Will be read in with pd.read_csv and should 
            result in two named headers "pressure"(bars),"temperature"(K). Can also include 'kz' in 
            CGS units. Use pd_kwargs to ensure file is read in properly.
        kz_min : float, optional
            Minimum Kz value. This will reset everything below kz_min to kz_min. 
            Default = 1e5 cm2/s
        pd_kwargs : kwargs
            Pandas key words for file read in. 
            If reading old style eddysed files, you would need: 
            skiprows=3, delim_whitespace=True, header=None, names=["ind","pressure","temperature","kz"]
        """
        if not isinstance(df, type(None)):
            if isinstance(df, dict): df = pd.DataFrame(df)
            df = df.sort_values('pressure')
        elif not isinstance(filename, type(None)):
            df = pd.read_csv(filename, **pd_kwargs)
            df = df.sort_values('pressure')

        self.pressure = np.array(df['pressure'])
        self.temperature = np.array(df['temperature'])
        if 'kz' in df.keys(): 
            if df.loc[df['kz']<kz_min].shape[0] > 0:
                df.loc[df['kz']<kz_min] = kz_min
                print('Overwriting some Kz values to minimum value set by kz_min') 
            self.kz = np.array(df['kz'])
        else:
            self.kz = np.nan

        r_atmos = 8.3143e7 / self.mmw

        # itop=iz = [0:-1], ibot = [1:]
        #convert bars to dyne/cm^2 
        self.p_top = self.pressure*1e6
        self.t_top = self.temperature 

        dlnp = np.log( self.p_top[1:] / self.p_top[0:-1] )#ag

        #take pressures at midpoints of layers
        self.p = 0.5*( self.p_top[1:] + self.p_top[0:-1]) #ag
        dtdlnp = ( self.t_top[0:-1] - self.t_top[1:] ) / dlnp
        self.t = self.t_top[1:] + np.log( self.p_top[1:]/self.p )*dtdlnp

        self.scale_h = r_atmos * self.t / self.g

        self.dz_pmid = self.scale_h * np.log( self.p_top[1:]/self.p )
        self.dz_layer = self.scale_h * dlnp

        self.z_top = np.concatenate(([0],np.cumsum(self.dz_layer[::-1])))[::-1]

        self.z = self.z_top[1:]+self.dz_pmid

    def get_gravity(self, gravity=None, gravity_unit=None, radius=None, radius_unit=None, mass = None, mass_unit=None):
        """
        Get gravity based on mass and radius, or gravity inputs 

        Parameters
        ----------
        gravity : float 
            (Optional) Gravity of planet 
        gravity_unit : astropy.unit
            (Optional) Unit of Gravity
        radius : float 
            (Optional) radius of planet MUST be specified for thermal emission!
        radius_unit : astropy.unit
            (Optional) Unit of radius
        mass : float 
            (Optional) mass of planet 
        mass_unit : astropy.unit
            (Optional) Unit of mass    
        """
        if (mass is not None) and (radius is not None):
            m = (mass*mass_unit).to(u.g)
            r = (radius*radius_unit).to(u.cm)
            g = (c.G.cgs * m /  (r**2)).value
            self.g = g
            self.gravity_unit = 'cm/(s**2)'
        elif gravity is not None:
            g = (gravity*gravity_unit).to('cm/(s**2)')
            g = g.value
            self.g = g
            self.gravity_unit = 'cm/(s**2)'
        else: 
            raise Exception('Need to specify gravity or radius and mass + additional units')


    def get_kz(self,df = None, constant=None,kz_min = 1e5): 
        """
        Define Kz in CGS. Should be on same grid as pressure. This overwrites whatever was 
        defined in get_pt ! Users can define kz by: 
            1) Defining a DataFrame with keys 'pressure' (in bars), and 'kz'
            2) Defining constant kz 

        Parameters
        ----------
        df : pandas.DataFrame, dict
            Dataframe or dictionary with 'kz' as one of the fields. 
        
        """

        if not isinstance(df, type(None)):
            #reset to minimun value if specified by the user
            if df.loc[df['kz']<kz_min].shape[0] > 0:
                df.loc[df['kz']<kz_min] = kz_min
                print('Overwriting some Kz values to minimum value set by kz_min') 
            self.kz = np.array(df['kz'])
            #make sure pressure and kz are the same size 
            if len(self.kz) != len(self.pressure) : 
                raise Exception('Kzz and pressure are not the same length')

        elif not isinstance(constant, type(None)):
            self.kz = constant
            if self.kz<kz_min:
                self.kz = kz_min
                print('Overwriting kz constant value to minimum value set by kz_min')


        #   vertical eddy diffusion coefficient (cm^2/s)
        #   from Gierasch and Conrath (1985)
        # we are discontinuing this formalism
        # self.kz = (scalef_kz * scale_h * (mixl/scale_h)**(4./3.) * #when dont know kz
        #  ( ( r_atmos*chf ) / ( rho_atmos*c_p ) )**(1./3.)) #when dont know kz

def get_mie(gas, directory):
    """
    Get Mie parameters from old files
    """
    df = pd.read_csv(os.path.join(directory,gas+".mieff"),names=['wave','qscat','qext','cos_qscat'], delim_whitespace=True)
    
    nwave = int( df.iloc[0,0])
    nradii = int(df.iloc[0,1])

    #get the radii (all the rows where there the last three rows are nans)
    radii = df.loc[np.isnan(df['qscat'])]['wave'].values

    df = df.dropna()

    assert len(radii) == nradii , "Number of radii specified in header is not the same as number of radii"
    assert nwave*nradii == df.shape[0] , "Number of wavelength specified in header is not the same as number of waves in file"

    wave = df['wave'].values.reshape((nradii,nwave)).T
    qscat = df['qscat'].values.reshape((nradii,nwave)).T
    qext = df['qext'].values.reshape((nradii,nwave)).T
    cos_qscat = df['cos_qscat'].values.reshape((nradii,nwave)).T

    return qext,qscat, cos_qscat, nwave, radii

def get_refrind(igas,directory): 
    """
    Reads reference files with wavelength, and refractory indecies. 
    This function relies on input files being structured as a 4 column file with 
    columns: index, wavelength (micron), nn, kk 

    Parameters
    ----------
    igas : str 
        Gas name 
    directory : str 
        Directory were reference files are located. 
    """
    filename = os.path.join(directory ,igas+".refrind")
    #put skiprows=1 in loadtxt to skip first line
    idummy, wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), unpack=True, usecols=[0,1,2,3])#[:-1]

    return wave_in,nn,kk

def get_r_grid(r_min=1e-5, n_radii=40):
    """
    Get spacing of radii to run Mie code

    r_min : float 
        Minimum radius to compute (cm)

    n_radii : int
        Number of radii to compute 
    """
    vrat = 2.2 
    pw = 1. / 3.
    f1 = ( 2.0*vrat / ( 1.0 + vrat) )**pw
    f2 = (( 2.0 / ( 1.0 + vrat ) )**pw) * (vrat**(pw-1.0))

    radius = r_min * vrat**(np.linspace(0,n_radii-1,n_radii)/3.)
    rup = f1*radius
    dr = f2*radius

    return radius, rup, dr



#INPUTS REQUIRED 

condensibles = ['H2O','Fe', 'Al2O3' ,'Na2S','NH3', 'KCl',
                 'MnS','ZnS','Cr' 'MgSiO3' ,'Mg2SiO4'  ]

#minimum eddy diffusion coefficient in cm^2/s
kzz_min = 1e5 

#geometric standard deviation of lognormal size distribution 
sig_all = 2.0 

#Maximimum subgridding to arrive at solution 
nsub_max = 2*64 #why two of these... 

#ramp up optical depth below cloud base in calc_optics()
do_subcloud = False 

#   saturation factor (after condensation)
supsat = 0

#rain factor 
rainf_all = 0.5

rmin = 1e-5

nradii = 40 

