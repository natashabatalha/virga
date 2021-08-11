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
    if param is 'const':
        ad_qc = np.max([ 0., qt - ad_qvs ])

        # Eqn. 7 in A & M 
        #   Difference from advective-diffusive balance 
        advdif = ad_qbelow*np.exp( - ad_rainf*ad_qc*ad_dz / ( qt*ad_mixl ) )
        #print(advdif, ad_qc, ad_dz ,ad_mixl,qt )
    elif param is 'exp':
        fsed = ad_rainf; mixl = ad_mixl; z = ad_dz
        qc = (ad_qbelow - ad_qvs) * np.exp( - b * fsed / mixl * np.exp(zb/b) 
                            * (np.exp(z/b) -1) + eps*z/b)
        advdif = qc + ad_qvs

    advdif = advdif - qt
    return advdif



def vfall(r, grav,mw_atmos,mfp,visc,
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

def vfall_find_root(r, grav=None,mw_atmos=None,mfp=None,visc=None,
              t=None,p=None, rhop=None,w_convect=None ):
    """
    This is used to find F(X)-y=0 where F(X) is the fall speed for 
    a spherical particle and `y` is the convective velocity scale. 
    When the two are balanced we arrive at our correct particle radius.


    Therefore, it is the same as the `vfall` function but with the 
    subtraction of the convective velocity scale (cm/s) w_convect. 
    """
    vfall_r = vfall(r, grav,mw_atmos,mfp,visc,
              t,p, rhop )

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
        if solve_for is "vfall":
            return force_balance(u, temp, grav, mw_atmos, mfp, visc, t, p, rhop) 
        elif solve_for is "rw":
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
    if qv_gas_name == 'Mg2SiO4':
        pvap_test = get_pvap(t_test, p_test, mh=mh)
    else:
        pvap_test = get_pvap(t_test,mh=mh)

    fx = qv_factor * pvap_test / p_test 
    return np.log(fx) - np.log(q_below)


def find_cond_t(t_test, p_test = None, mh=None, mmw=None, gas_name=None):    
    """
    Root function used used to find condenstation temperature. E.g. 
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
    gas_mw, gas_mmr ,rho = gas_p_fun(mmw,mh=mh)
    #get vapor pressure and correct for masses of atmo and gas 
    if gas_name == 'Mg2SiO4':
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

