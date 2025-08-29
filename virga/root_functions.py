import numpy as np
from . import  pvaps
from . import  gas_properties
from scipy.stats import lognorm
from scipy.integrate import quad, simps
from scipy import optimize

def advdiff(qt, ad_qbelow=None,ad_qvs=None, ad_mixl=None,ad_dz=None ,ad_rainf=None,
        zb=None, b=None, eps=None, param='const'):
    """
    Calculate divergence from advective-diffusive balance for 
    condensate in a model layer

    All units are cgs
    
    A. Ackerman Feb-2000

    Parameters
    ----------
    qt : float 
        total mixing ratio of condensate + vapor (g/g)
    ad_qbelow : float 
        total mixing ratio of vapor in underlying layer (g/g)
    ad_qvs : float 
        saturation mixing ratio (g/g)
    ad_mixl : float 
        convective mixing length (cm)
    ad_dz : float 
        layer thickness (cm) 
    ad_rainf : float
        rain efficiency factor 
    zb : float
        altitude at bottom of layer
    b : float
        denominator of fsed exponential (if param is 'exp')
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)

    Returns
    -------
    ad_qc : float 
        mixing ratio of condensed condensate (g/g)
    """
    #   All vapor in excess of saturation condenses
    if param == 'const':
        ad_qc = np.max([ 0., qt - ad_qvs ])

        # Eqn. 7 in A & M 
        #   Difference from advective-diffusive balance 
        advdif = ad_qbelow*np.exp( - ad_rainf*ad_qc*ad_dz / ( qt*ad_mixl ) )
        #print(advdif, ad_qc, ad_dz ,ad_mixl,qt )
    elif param == 'exp':
        fsed = ad_rainf; mixl = ad_mixl; z = ad_dz
        qc = (ad_qbelow - ad_qvs) * np.exp( - b * fsed / mixl * np.exp(zb/b) 
                            * (np.exp(z/b) -1) + eps*z/b)
        advdif = qc + ad_qvs

    advdif = advdif - qt
    return advdif

def vfall(r, grav, mw_atmos, mfp, visc, t, p, rhop, aggregates, Df, N_mon, r_mon, k0):
    
    '''
    New vfall function written by SEM and MGL, based on the methods of:

        - "Aggregate Cloud Particle Effects in Exoplanet Atmospheres", Vahidinia et al. (2024)
        - "A Condensation–coalescence Cloud Model for Exoplanetary Atmospheres: Formulation and Test Applications to Terrestrial and Jovian Clouds", Ohno and Okuzumi (2017)
        - "Theoretical modeling of mineral cloud formation on super-Earths", K. Ohno (2024)
        
    We adapt the above to include a broad range of aggregates and atmospheric conditions. The general outline of the code is:
    
        - Find the characteristic radius (R_c) of the particle:

            - SPHERES: R_c = radius of the sphere

            - AGGREGATES: R_c = radius of gyration
        
        - Determine the Knudsen number:
        
            - If Kn > 10 we are in the Free molecular regime (if the mean free path of the gas is large compared to the particle size).
                
                - Here, use equation 46 of Vahidinia et al. (2024) to adapt the fall velocity of spheres for aggregates of fractal dimension d_f

            - If Kn < 10 we are in the Continuum regime (the gas can be treated a a continuous fluid)

                - Here, use equation 23 of Ohno and Okuzumi (2017) to calculate vfall for aggregates of any type, with the addition of the Beta slip correction
                  factor (Eq 3.11 of Ohno, 2024). This equation (Eq. 23) is valid for all three of the stokes, slip and turbulent regimes, so there is no need to calculate 
                  Reynolds number. It is not strictly designed to work for really low fractal dimensions (e.g. DCLA aggregates with d_f = 1.2), so consider the effects 
                  an upper bound for these if used (though aggregates in high atmospheres are not usually in the continuum regime anyway, they are in the Free molecular 
                  regime. The use of mobility radius for linear DCLA aggregates was explored, but R_gyr was chosen to remain consistent and continuous as d_f changes.
            
    Parameters
    ----------
    r : float
        particle radius (cm)
    grav : float 
        acceleration of gravity (cm/s^2)
    mw_atmos : float 
        atmospheric molecular weight (g/mol)
    mfp : float 
        atmospheric molecular mean free path (cm)
    visc : float 
        atmospheric dynamic viscosity (dyne s/cm^2) see Eqn. B2 in A&M
    t : float 
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    rhop : float 
        density of particle (g/cm^3)
    aggregates : bool, optional
        Turns on aggregate functionality. set to False for default spheres.
    Df : float, optional
        fractal dimension of aggregate particle.
    N_mon: float, optional
        number of monomers in aggregate
    r_mon : float, optional
        the monomer radius (i.e., the smaller sub-particles) for aggregate particles (cm).
    
    Returns
    -------
    v_fall : float
        terminal velocity of particle (cm/s)
    '''

    R_GAS = 8.3143e7 

    # calculate the characteristic radius of the particle
    if aggregates:

        # if N_mon was prescribed by the user, calculate r_mon (or vice versa)
        if N_mon is not None: 
            N_mon_prescribed=1 # N_mon is a fixed value, prescribed by the user
            original_N_mon = N_mon # record the original set value to send to the 'find_N_mon_and_r_mon' function below
        else:
            N_mon_prescribed=0 # N_mon needs to be calculated for each radius
            original_N_mon = 0 # original number of monomers not set

        # calculate N_mon and r_mon for this radius
        N_mon, r_mon = find_N_mon_and_r_mon(N_mon_prescribed, r, original_N_mon, r_mon)

        if r <= r_mon: # the aggregate radius can never be less than the monomer particle size, so if this is the case, just use spheres
            # SPHERES
            r_c = r # use the radius of the sphere as the characteristic radius of the particle
            rho_c = rhop # use the bulk density of the monomers as the characteristic density

        else:
            # AGGREGATES
            
            if k0<=0: # if fractal prefractor k0 is left unprescribed by user, calculate it here (in the case of fixed r_mon, k0 can change as the number of monomers grows, so it needs to be calculated here, for each new radius)
                k0 = calculate_k0(N_mon, Df)

            R_gyr = (N_mon/k0)**(1/Df)*r_mon # calculate radius of gyration
            r_c = R_gyr # use the radius of gyration as the characteristic radius of the particle
            rho_c = rhop*N_mon*(r_mon/R_gyr)**3 # calculate characteristic aggregate density using R_gyr as the characteristic radius (derived from Ohno thesis)
    else:
        # SPHERES
        r_c = r # use the radius of the sphere as the characteristic radius of the particle
        rho_c = rhop # use the bulk density of the monomers as the characteristic density
   
    knudsen = mfp / r_c # calculate knudsen number; remember to use the CHARACTERISTIC radius here to determine the regime
    rho_atmos = p / ( (R_GAS/mw_atmos) * t ) # calculate density of the atmosphere
    
    if knudsen > 10: # Free molecular regime 
        N_A = 6.02e23 # avogadro's number
        k_b = 1.38e-16 # Boltzmann constant (in cgs units)
        sound_speed = np.sqrt(3*k_b*t/(mw_atmos/N_A)) # calculate thermal speed (speed of sound)
        
        vfall_r = ((2.0/3.0)*grav*rhop/(sound_speed*rho_atmos))*r # find vfall for solid spheres in the Free Molecular regime
        
        if aggregates and (r > r_mon):
            vfall_r = vfall_r * (R_gyr/r_mon)**(((2*Df)-6)/3) # if finding vfall for aggregates, apply equation 46 of Vahidinia et al. (2024) (works in the Free Molecular regime only)

    else: # Continuum regime: use Eq. 23 of Ohno and Okuzuki (2017), but with the added Beta slip correction factor as in (Eq. 3.11) of "Theoretical modeling of mineral cloud formation on super-Earths" (2024), K. Ohno
        beta_slip = 1. + knudsen*(1.257 + 0.4*np.exp(-1.1/knudsen)) # calculate the Beta slip factor
        vfall_r = (2.0*beta_slip*grav*(r_c**2)*rho_c/(9.0*visc)) * (1.0 + (0.45*grav*(r_c**3)*rho_atmos*rho_c/(54.0*visc**2))**(2.0/5.0) )**(-5.0/4.0)

    # For aggregates (especially where d_f<2), check that the terminal velocity of the aggregate > terminal velocity of a single monomer.
    # If this condition is not true, set the monomer terminal velocity as a lower limit for the aggregate's terminal velocity. 
    # This is because the aggregate's cross section is probably being over-approximated, and even for the most linear fractals it should not
    # exceed the sum of monomer cross sections -- see Tazaki (2021) Fig. 4

    if aggregates:
        if r>r_mon: # if the radius is at least as large as 1 monomer (otherwise we just consider small spheres, not aggregates, so the rule below does not apply)
            knudsen = mfp / r_mon # # calculate knudsen number for the monomer; using the monomer radius here to determine the regime
            
            if knudsen > 10: # Free molecular regime 
                N_A = 6.02e23 # avogadro's number
                k_b = 1.38e-16 # Boltzmann constant (in cgs units)
                sound_speed = np.sqrt(3*k_b*t/(mw_atmos/N_A)) # calculate thermal speed (speed of sound) (this may not have been done above, as aggregates may not have been in the Kn>10 regime)
                vfall_monomer = ((2.0/3.0)*grav*rhop/(sound_speed*rho_atmos))*r_mon # find vfall for single monomers in the Free Molecular regime
                
            else: # Continuum regime: use Eq. 23 of Ohno and Okuzuki (2017), but with the added Beta slip correction factor as in (Eq. 3.11) of "Theoretical modeling of mineral cloud formation on super-Earths" (2024), K. Ohno
                beta_slip = 1. + knudsen*(1.257 + 0.4*np.exp(-1.1/knudsen)) # calculate the Beta slip factor
                vfall_monomer = (2.0*beta_slip*grav*(r_mon**2)*rhop/(9.0*visc)) * (1.0 + (0.45*grav*(r_mon**3)*rho_atmos*rhop/(54.0*visc**2))**(2.0/5.0) )**(-5.0/4.0)

            if(vfall_r < vfall_monomer): # if aggregate vfall < monomer vfall
                vfall_r = vfall_monomer # set monomer vfall as lower limit

    return vfall_r

def vfall_legacy(r, grav,mw_atmos,mfp,visc,
              t,p, rhop):
    """
    Calculate fallspeed for a spherical particle at one layer in an
    atmosphere, depending on Reynolds number for Stokes flow.

    For Re_Stokes < 1, use Stokes velocity with slip correction
    For Re_Stokes > 1, use fit to Re = exp( b1*x + b2*x^2 )
     where x = log( Cd Re^2 / 24 )
     where b2 = -0.1 (curvature term) and 
     b1 from fit between Stokes at Re=1, Cd=24 and Re=1e3, Cd=0.45

    and Precipitation, Reidel, Holland, 1978) and Carlson, Rossow, and
    Orton (J. Atmos. Sci. 45, p. 2066, 1988)

    all units are cgs
    
    A. Ackerman Feb-2000

    Parameters
    ----------
    r : float
        particle radius (cm)
    grav : float 
        acceleration of gravity (cm/s^2)
    mw_atmos : float 
        atmospheric molecular weight (g/mol)
    mfp : float 
        atmospheric molecular mean free path (cm)
    visc : float 
        atmospheric dynamic viscosity (dyne s/cm^2) see Eqn. B2 in A&M
    t : float 
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    rhop : float 
        density of particle (g/cm^3)
    """

    # the drag coefficient for a reynolds number of 1000
    # which is appropriate for oblate spheroids 
    # Fig. 10-36 in Pruppacher & Klett 1978
    cdrag = 0.45 

    #In order to solve the drag problem we fit y=log(reynolds)
    #as a function of x=log(cdrag * reynolds**2)
    #if you assume that at reynolds= 1, cdrag=24 and 
    #reynolds=1000, cdrag=0.45 you get the following fit: 
    # y = 0.8 * x - 0.1 * x**2
    #Full explanation: see A & M Appendix B between eq. B2 and B3
    #Simply though, this allows us to get terminal fall velocity from 
    #reynolds number
    b1 = 0.8 
    b2 = -0.01 



    R_GAS = 8.3143e7 

    #calculate constants need to get Knudsen and Reynolds numbers
    knudsen = mfp / r
    rho_atmos = p / ( (R_GAS/mw_atmos) * t )
    drho = rhop - rho_atmos

    #Cunningham correction (slip factor for gas kinetic effects)
    #Cunningham, E., "On the velocity of steady fall of spherical particles through fluid medium," Proc. Roy. Soc. A 83(1910)357
    #Cunningham derived a value of 1.26 in the stone ages. In reality, this number is 
    #a function of the knudsen number. Various studies have derived 
    #different value for this number (see this citation
    #https://www.researchgate.net/publication/242470948_A_Novel_Slip_Correction_Factor_for_Spherical_Aerosol_Particles
    #Within the range of studied values, this 1.26 number changes particle sizes by a few microns
    #That is A OKAY for the level of accuracy we need. 
    beta_slip = 1. + 1.26*knudsen 

    #Stokes terminal velocity (low Reynolds number)
    #EQN B1 in A&M 
    #visc is eqn. B2 in A&M but is computed in `calc_qc`
    #also eqn 10-104 in Pruppacher & klett 1978
    vfall_r = beta_slip*(2.0/9.0)*drho*grav*r**2 / visc

    #compute reynolds number for low reynolds number case
    reynolds = 2.0*r*rho_atmos*vfall_r / visc

    #if reynolds number is between 1-1000 we are in turbulent flow 
    #limit
    if ((reynolds > 1) and (reynolds<=1e3)): #:#(reynolds >1e-2) and (reynolds <= 300)
        #OLD METHODLOGY
        #correct drag coefficient for turbulence (x = Cd Re^2 / 24)
        #x = np.log( reynolds )
        #y = b1*x + b2*x**2

        #compute cd * N_re^2 by equating drag and gravitational force  
        cd_nre2 = 32.0 * r**3.0 * drho * rho_atmos * grav / (3.0 * visc ** 2 ) 
        #coefficients from EQN 10-111 in Pruppachar & Klett 1978
        #they are an empirical fit to Figure 10-9
        xx = np.log(cd_nre2)
        b0,b1,b2,b3,b4,b5,b6 = -0.318657e1, 0.992696, -.153193e-2, -.987059e-3, -.578878e-3, 0.855176e-4, -0.327815e-5
        y = b0 + b1*xx**1 + b2*xx**2 + b3*xx**3 + b4*xx**4 + b5*xx**5 + b6*xx**6

        reynolds = np.exp(y)
        vfall_r = visc*reynolds / (2.*r*rho_atmos)

    if reynolds >1e3 :# 300
        #when Reynolds is greater than 1000, we can just use 
        #an asymptotic value that is independent of Reynolds number
        #Eqn. B3 from A&M 01
        vfall_r = beta_slip*np.sqrt( 8.*drho*r*grav / (3.*cdrag*rho_atmos) )

    return vfall_r 


def vfall_aggregates(r, grav, mw_atmos, t, p, rhop, D=2, Ragg=None):
    """
    DEPRECATED, SEE CURRENT IMPLEMENTATION IN 'VFALL' FUNCTION
    Calculate fallspeed for a particle at one layer in an
    atmosphere, assuming low Reynolds number and in the molecular regime (Epstein drag), 
    i.e., where Knudsen number (mfp/r) is high (>>1). 

    User chooses whether particle is spherical or an aggregate.
    If aggregrate, the monomer size is given by r and the outer effective radius of the aggregrate is Ragg.
    
    all units are cgs
    
    Parameters
    ----------
    r : float
        monomer particle radius (cm)
    mw_atmos : float 
        atmospheric molecular weight (g/mol)
    t : float 
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    rhop : float 
        density of particle (g/cm^3)
    grav : float 
        acceleration of gravity (cm/s^2)
    Ragg : float
        aggregate particle effective radius (cm). (Defaults to being spherical monomers).
    D : float
        fractal number (Default is 2 because function reduces to monomers at this value).
    """
    R_GAS = 8.3143e7  #universial gas constant; cgs units cm3-bar/mole-K
    k = 1.38e-16  #boltzmann contant in cgs units -  cm2 g s-2 K-1 (ergs/K)
    
    N_avo = 6.022e23 
    
    mass = mw_atmos/N_avo 
    
    rho_atmos = (mw_atmos*p) / (RGAS*t) #atmospheric density
    drho = rhop - rho_atmos
    v_thermal = np.sqrt((3*k*t)/mass) #root mean speed of the gas 
    
    #the stopping time of the particle
    t_stop_epstein_r = (2.0/3.0) * (r*drho) / (rho_atmos*v_thermal) 

    if isinstance(Ragg, type(None)): Ragg = r
    
    vfall_epstein_agg_r = t_stop_epstein_r * grav * (Ragg/r)**(D-2)

    return vfall_epstein_agg_r

def vfall_aggregrates_ohno(r, grav,mw_atmos,mfp, t, p, rhop, ad_qc, D=2):
    """
    Calculates fallspeed for a fractal aggegrate particle as performed
    by Ohno et al., 2020, with an outer 
    effective radius of R_agg made up of monomers of size r, at one layer in an
    atmosphere. ***Requires characteristic radius ragg to have D > or equal 2, i.e., not very fluffy aggregrates.

    Essentially a more explicit version of the regular "vfall_aggregates" function, 
    and is not dependent on being in free molecular (Esptein) regime
    
    all units are cgs
    
    Parameters
    ----------
    r : float
        monomer particle radius (cm)
    grav : float 
        acceleration of gravity (cm/s^2)
    rhop : float
        density of monomer particle (g/cm^3)
    t : float 
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    mw_atmos : float 
        atmospheric molecular weight (g/mol)
    mfp : float 
        atmospheric molecular mean free path (cm)
    ad_qc : float 
        mixing ratio of condensed condensate (g/g)
    D : float
        fractal number (Default is 2).
    gas_mw : float
        condensate gas mean molecular weight (g/mol)
    """
    #Define some constants
    R_GAS = 8.3143e7  #universial gas constant; cgs units cm3-bar/mole-K
    k = 1.38e-16  #boltzmann contant in cgs units -  cm2 g s-2 K-1 (ergs/K)
    N_avo = 6.022e23 #avogadro's number

    #determine the number density of monomer particles
    N = ad_qc * N_avo/ gas_mw

    #calculate the aggregrate radius based on the fractal dimension, monomer radius, and number density of monomers
    
    k0 = 0.716 * (1-D) + np.sqrt(3)# from Tazaki 2021 
    Ragg = r * (N/k0)**(1/D) 

    mass = mw_atmos/N_avo
    rho_atmos = (mw_atmos*p) / (RGAS*t) #atmospheric density

    rho_agg = rhop *  N * (r/Ragg)**3
    
    drho = rho_agg - rho_atmos

    kn = mfp / Ragg #Knudsen number
    beta = 1.0 + (1.26*kn) #Cunningham correction (slip factor for gas kinetic effects)
    v_thermal = np.sqrt((8*k*t)/(mass*np.pi)) #thermal speed of the gas 

    #if kn > 10:
    #    visc = (1.0/3.0)*rho_atmos*v_thermal*mfp #viscosity of the atmosphere, appropriate for large Kn (Esptein)
    #elif kn < :
    visc = 5.877e-6 * np.sqrt(t) #in dyne/cm^2 with t in K (via Woitke & Helling 2003)

    vfall_stokes = (2.0/9.0) * beta * grav * ((Ragg)**2) * (drho/visc) 
    v_bracket = (1.0 + (((0.45/54.0) * (grav/((visc)**2)) * ((Ragg)**3) * rho_atmos * rho_agg)**(2./5.)))**(-5.0/4.0)
    
    vfall_r_ohno = vfall_stokes  * v_bracket
   
    return vfall_r_ohno

def vfall_find_root(r,grav=None,mw_atmos=None,mfp=None,visc=None,
              t=None,p=None, rhop=None,w_convect=None,aggregates=False,Df=None,N_mon=None,r_mon=None,k0=None):
    """
    This is used to find F(X)-y=0 where F(X) is the fall speed for 
    a spherical particle and `y` is the convective velocity scale. 
    When the two are balanced we arrive at our correct particle radius.


    Therefore, it is the same as the `vfall` function but with the 
    subtraction of the convective velocity scale (cm/s) w_convect. 
    """
    vfall_r = vfall(r, grav,mw_atmos,mfp,visc,t,p,rhop,aggregates,Df,N_mon,r_mon,k0)

    #print(f" p = {p:10.5f}     r = {r:10.5e} cm      w_convect = {w_convect:10.5e}   v_fall = {vfall_r:10.5e}     Sum = {vfall_r - w_convect:10.5f}")
    
    return vfall_r - w_convect

def force_balance(vf, r, grav, mw_atmos, mfp, visc, t, p, rhop, gas_kinetics=True):
    """"
    Define force balance for spherical particles falling in atmosphere, namely equate
    gravitational and viscous drag forces.

    Viscous drag assumed to be quadratic (Benchaita et al. 1983) and is a function of 
    the Reynolds number dependent drag coefficient.
    Drag coefficient taken from Khan-Richardson model (Richardson et al. 2002) and
    is valid for 1e-2 < Re < 1e5.

    Parameters
    ----------
    vf : float
        particle sedimentation velocity (cm/s)
    r : float
        particle radius (cm)
    grav : float 
        acceleration of gravity (cm/s^2)
    mw_atmos : float 
        atmospheric molecular weight (g/mol)
    mfp : float 
        atmospheric molecular mean free path (cm)
    visc : float 
        atmospheric dynamic viscosity (dyne s/cm^2) see Eqn. B2 in A&M
    t : float 
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    rhop : float 
        density of particle (g/cm^3)

    """

    R_GAS = 8.3143e7 
    rho_atmos = p / ( (R_GAS/mw_atmos) * t )
    # coefficients for drag coefficient taken from Khan-Richardson model (Richardson et al. 2002)
    # valid for 1e-2 < Re < 1e5
    a1 = 1.849; b1 = -0.31
    a2 = 0.293; b2 = 0.06
    b3 = 3.45    

    # include gas kinetic effects through slip factor 
    knudsen = mfp / r
    beta_slip = 1. + 1.26*knudsen 
    if gas_kinetics:
        vf = vf/beta_slip
                 
    # Reynolds number
    Re = rho_atmos * vf * 2 * r /  visc

    # Khan-Richardson approximation for drag coefficient is valid for 1e-2 < Re < 1e5
    RHS = (a1 * Re**b1 + a2 * Re**b2)**b3 * rho_atmos * np.pi * r**2 * vf**2 
       
    # gravitational force
    LHS = 4 * np.pi * r**3 * (rhop-rho_atmos) * grav / 3 

    return LHS - RHS

def solve_force_balance(solve_for, temp, grav, mw_atmos, mfp, visc, t, p, rhop, lo, hi):
    """
    This can be used to equate gravitational and viscous drag forces to find either
        (a) fallspeed for a spherical particle of given radius, or
        (b) the radius of a particle falling with a particular speed
    at one layer in an atmosphere.

    Parameters
    ----------
    solve_for : str
        property we are solving force balance for
        either "vfall" or "rw" 
    temp : float
        the other property needed to solve the force balance for solve_for
        ie if solve_for="vfall", temp will be a radius (cm)
           if solve_for="rw", temp will be w_convect (cm/s)
    grav : float 
        acceleration of gravity (cm/s^2)
    mw_atmos : float 
        atmospheric molecular weight (g/mol)
    mfp : float 
        atmospheric molecular mean free path (cm)
    visc : float 
        atmospheric dynamic viscosity (dyne s/cm^2) see Eqn. B2 in A&M
    t : float 
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    rhop : float 
        density of particle (g/cm^3)
    lo : float
        lower bound for root-finder
    hi : float
        upper bound for root-finder

    Returns
    ______
    soln.root : float
        the value of the parameter solve_for
    """

    def force_balance_new(u):
        if solve_for == "vfall":
            return force_balance(u, temp, grav, mw_atmos, mfp, visc, t, p, rhop) 
        elif solve_for == "rw":
            return force_balance(temp, u, grav, mw_atmos, mfp, visc, t, p, rhop) 

    soln = optimize.root_scalar(force_balance_new, bracket=[lo, hi], method='brentq') 

    return soln.root

def qvs_below_model(p_test, qv_dtdlnp=None, qv_p=None, 
    qv_t=None, qv_factor=None, qv_gas_name=None,mh=None,q_below=None):
    """
    Calculates the pressure of saturation mixing ratio for a gas 
    that is extrapolated below the model domain. 
    This is specifically used if the gas has saturated 
    below the model grid
    """
    #  Extrapolate temperature lapse rate to test pressure

    t_test = qv_t + np.log( qv_p / p_test )*qv_dtdlnp
    
    #  Compute saturation mixing ratio
    get_pvap = getattr(pvaps, qv_gas_name)
    if qv_gas_name in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
        pvap_test = get_pvap(t_test, p_test, mh=mh)
    else:
        pvap_test = get_pvap(t_test,mh=mh)

    fx = qv_factor * pvap_test / p_test 
    return np.log(fx) - np.log(q_below)


def find_cond_t(t_test, p_test = None, mh=None, mmw=None, gas_name=None, gas_mmr=None):    
    """
    Root function used used to find condensation temperature. E.g. 
    the temperature when  

    log p_vap = log partial pressure of gas 

    Parameters
    ----------
    t_test : float 
        Temp (K)
    p_test : float 
        Pressure bars 
    mh : float 
        NON log mh .. aka MH=1 for solar 
    mmw : float 
        mean molecular weight (2.2 for solar)
    gas_name : str 
        gas name, case sensitive 
    """
    pvap_fun = getattr(pvaps, gas_name)
    gas_p_fun = getattr(gas_properties, gas_name)
    #get gas mixing ratio 
    gas_mw, gas_mmr ,rho = gas_p_fun(mmw,mh=mh , gas_mmr=gas_mmr)
    #get vapor pressure and correct for masses of atmo and gas 
    if gas_name in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
        pv = gas_mw/mmw*pvap_fun(t_test,p_test, mh=mh)/1e6 #dynes to bars 
    else:
        pv = gas_mw/mmw*pvap_fun(t_test, mh=mh)/1e6 #dynes to bars 
    #get partial pressure
    partial_p = gas_mmr*p_test*mh 
    if pv == 0:
        pv = 1e-30 #dummy small
    return np.log10(pv) - np.log10(partial_p)

def moment(n, s, loc, scale, dist="lognormal"):
    """
    Calculate moment of size distribution.
    Will be extended to include more than just lognormal

    Parameters
    ----------
    n : float
        nth moment to be calculated
    s : float 
        std dev 
    loc : float
        Shift in distribution
    scale: float
        Scale distribution
    dist: str
        Continuous random variable for particle size distribution

    Returns
    -------
    moment_out : float 
        nth moment of distribution
    """

    #   for non-integer (any) n, must approximate moment integral numerically
    #   the following code works but it is slow 

    def pdf(r, s, loc, scale, dist):
        if dist=="lognormal":
            #return lognorm.pdf(r, s, loc, scale)
            return np.exp(- np.log((r-loc)/scale)**2 / (2*s**2)) / (r * s * np.sqrt(2. * np.pi))

    def func(r, n, s, loc, scale, dist):
        return r**n * pdf(r, s, loc, scale, dist)

    lbd = 0.
    ubd = np.inf
    moment_out = quad(func, lbd, ubd, epsabs=0, args=(n, s, loc, scale, dist))[0]

    #   for quicker result, take int(n) and use lognorm.moment, however, it will be slow for int(n) >= 2
    #n = int(round(n))
    #moment_out = lognorm.moment(n, s, loc, scale)
    return moment_out

def find_rg(rg, fsed, rw, alpha, s, loc=0., dist="lognormal"):
    """
    Root function used used to find the geometric mean radius of 
    lognormal size distribution.

    Useful if we consider more complicated size distributions for which
    analytical expressions for moments are not easily obtained.

    Parameters
    ----------
    rg : float 
        Geometric mean radius
    fsed : float 
        Sedimentation efficiency
    rw : float 
        Fall velocity particle radius 
    alpha : float 
        Exponent in power-law approximation for particle fall-speed
    s : float 
        s = log(sigma) where sigma is the geometric std dev of lognormal distn 
    loc : float
        Shift in lognormal distribution
    dist: str
        Continuous random variable for particle size distribution
    """

    return fsed - moment(3+alpha, s, loc, rg, dist) / rw**alpha / moment(3, s, loc, rg, dist)

def calculate_k0(N_mon, Df):
    """

    Find fractal prefactor k0 using Eq 15 of Moran & Lodge (2025)
    
    Parameters
    ----------
    N_mon : int
        Number of monomers in aggregate
    Df : float 
        Fractal dimension of aggregate
    
    Returns
    -------
    k0 : float
        Fractal prefactor k0
    
    """

    if (N_mon<=100):   
        # the Tazaki equation below is only valid for BCCA aggregates, and at low monomer numbers (e.g. N = 2 monomers) this is not true. 
        # Therefore, for N<=100 monomers, we smoothly transition to k0 = 1 for all d_f value (see Moran et al. 2025 for derivation)
        C1 = (1.448-0.716*Df)/99
        C2 = 1.0 - C1
        k0 = C1 * N_mon + C2

    else:
        # for monomer numbers > 100, find k0 using BCCA aggregate Eq. 2 of "Analytic expressions for geometric cross-sections of fractal dust aggregates", Tazaki (2021), MNRAS 504, 2811–2821
        k0 = 0.716*(1.0 - Df) + np.sqrt(3)
    
    return k0

def find_N_mon_and_r_mon(N_mon_prescribed, radius, original_N_mon, r_mon):
    """

    Find the following variables for a given radius. One value should be prescribed by the user, but both
    can change according to the growth model in Figure 2 of Moran & Lodge (2025):

        - N_mon: Number of monomers
        - r_mon: monomer radius
    
    Parameters
    ----------
    N_mon_prescribed : int
        Value of 1 if the user prescribed N_mon, 0 if they prescribed r_mon instead
    radius : float
        compact radius of aggregate (units:cm)
    original_N_mon : int 
        original number of monomers prescribed by user
    r_mon : float
        monomer radius (units:cm)
    
    Returns
    -------
    N_mon : int 
        number of monomers
    r_mon : float
        monomer radius (units:cm)
    
    """

    if (N_mon_prescribed==1):
        N_mon = original_N_mon # first, test the original N_mon value set by the user (this may have been changed on the last loop, so reset it here)
        r_mon = radius / np.cbrt(N_mon) # if we have provided N_mon, calculate r_mon (through conservation of volume between an aggregate and a compact sphere)

        # check whether this value of N and r would mean monomers are smaller than atoms (our absolute lowest physical limit for monomers)
        if (r_mon<1e-8):
            r_mon = 1e-8 # if so, set r_mon = atomic radius in cm (10^-10 m)...
            N_mon = (radius / r_mon)**3 # ...and decrease N_mon from the prescribed value -- calculate how many of these tiny monomers would be needed to achieve the desired compact particle radius
    
    else:
        N_mon = (radius / r_mon)**3 # alternatively, if we have provided r_mon, calculate N_mon (through conservation of volume between aggregate and compact sphere)
    
    return N_mon, r_mon