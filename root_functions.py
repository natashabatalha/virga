import numpy as np
def advdiff(qt, ad_qbelow=None,ad_qvs=None, ad_mixl=None,ad_dz=None ,ad_rainf=None):
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

	Returns
	-------
	ad_qc : float 
		mixing ratio of condensed condensate (g/g)
	"""
	#   All vapor in excess of saturation condenses (supsat=0)
	ad_qc = np.max([ 0., qt - ad_qvs ])

	#   Difference from advective-diffusive balance 
	advdif = ad_qbelow*np.exp( - ad_rainf*ad_qc*ad_dz / ( qt*ad_mixl ) ) - qt

	return advdif



def vfall(r, grav=None,mw_atmos=None,mfp=None,visc=None,
			  t=None,p=None, rhop=None ):
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
		atmospheric dynamic viscosity (dyne s/cm^2)
	t : float 
		atmospheric temperature (K)
	p  : float
		atmospheric pressure (dyne/cm^2)
	rhop : float 
		density of particle (g/cm^3)

	NOTE: WHAT ARE THESE NUMBERS?!?
	"""
	b1 = 0.8           #Ackerman
	b1 = 0.86 # Rossow
	b1 = 0.72 #Carlson
	b2 = -0.01 
	cdrag = 0.45 #Ackerman
	cdrag = 0.2 #Rossow
	cdrag = 2.0 #Carlson

	R_GAS = 8.3143e7 

	#calculate vfall based on Knudsen and Reynolds numbers

	knudsen = mfp / r
	rho_atmos = p / ( (R_GAS/mw_atmos) * t )
	drho = rhop - rho_atmos

	#Cunningham correction (slip factor for gas kinetic effects)
	slip = 1. + 1.26*knudsen

	#   Stokes terminal velocity (low Reynolds number)
	vfall_r = slip*(2.0/9.0)*drho*grav*r**2 / visc
	reynolds = 2.0*r*rho_atmos*vfall_r / visc

	if (reynolds > 1.) and (reynolds<=1e3):

		#correct drag coefficient for turbulence (Re = Cd Re^2 / 24)

		x = np.log( reynolds )
		y = b1*x + b2*x**2
		reynolds = np.exp(y)
		vfall_r = vf_visc*reynolds / (2.*r*rho_atmos)

	elif reynolds > 1e3 :
		#   drag coefficient independent of Reynolds number
		vfall_r = slip*np.sqrt( 8.*drho*r*grav / (3.*cdrag*rho_atmos) )

	return vfall_r


