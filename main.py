import gas_properties
import astropy.constants as c
import astropy.units as u
import pandas as pd
import numpy as np
import pvaps
import os
from scipy import optimize 
from root_functions import advdiff, vfall
def justdoit(atmo, directory = None, do_optics=False):
	#loop through gases to get properties and optics/Mie params


	mmw = atmo.mmw
	mh = atmo.mh
	condensibles = atmo.condensibles

	ngas = len(condensibles)

	gas_mw = np.zeros(ngas)
	gas_mmr = np.zeros(ngas)
	rho_p = np.zeros(ngas)
	wave=np.zeros(shape=(nwave))
	qext=np.zeros(shape=(nwave,nrad,ngas))
	qscat = np.zeros(shape=(nwave,nrad,ngas))
	cos_qscat=np.zeros(shape=(nwave,nrad,ngas))
	radius=np.zeros(shape=(nrad))
	rup=np.zeros(shape=(nrad))
	dr=np.zeros(shape=(nrad))
	vrat = 2.2
	rmin = 1e-5
	pw = 1. / 3.
	f1 = ( 2.0*vrat / ( 1.0 + vrat) )**pw
	f2 = (( 2.0 / ( 1.0 + vrat ) )**pw) * (vrat**pw-1.0)
	nrad=40
	
	for i, igas in zip(range(ngas),condensibles) : 

		#GET GAS PROPETIES
		run_gas = getattr(gas_properties, igas)
		gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh)
		
		

		#GET OR READ IN OPTICS
		if do_optics:
			## Calculates optics by reading refrind files
			thetd=0.0   # incident wave angle
			n_thetd=1
			
			## obtaining refrind data for ith gas
			wave_in,nn,kk = get_refrind(igas,directory)
			wave=wave_in*1e-4  ## converting to cm 
			nwave=len(wave)  #number of wavalength bin centres for calculation
			
			#Setup up a particle size grid on first run and calculate single-particle scattering
			if i==0:
				for irad in range(nrad):
    				radius[irad],rup[irad],dr[irad] = rmin * vrat**(float(irad)/3.),f1*radius[irad],f2*radius[irad]
 
				
			#compute individual parameters for each gas
			
			for iwave in range(nwave):
				for irad in range(nrad):
					if irad== 0 :
                				dr5= (( rup[0] - radius[0] ) / 5.)
                				rr= radius[0]
            				else:
                				dr5 = ( rup[irad] - rup[irad-1] ) / 5.
                				rr  = rup[irad-1]
					corerad = 0.
            				corereal = 1.
            				coreimag = 0.
					## averaging over 6 radial bins to avoid fluctuations
					bad_mie= False
					for isub in range(6):
                				wvno=2*pi/wave[iwave]
                				qe_pass, qs_pass, c_qs_pass,istatus= calc_mie(rr, nn[iwave], kk[iwave], thetd, n_thetd, corerad, corereal, coreimag, wvno)
                				if istatus == 0:
                    					qe=qe_pass
                    					qs=qs_pass
                    					c_qs=c_qs_pass
                				else:
                    					if bad_mie == False :
                        				bad_mie = True
                    					raise Exception('do_optics(): no Mie solution for irad, r(um), iwave, wave(um), n, k, gas = ')
                    					
             
                
                				qext[iwave,irad,i]+= qe
                				qscat[iwave,irad,i]+= qs
                				cos_qscat[iwave,irad,i] += c_qs
                				rr+=dr5
					## adding to master arrays
					qext[iwave,irad,igas] = qext[iwave,irad,i] / 6.     
            				qscat[iwave,irad,igas] = qscat[iwave,irad,i] / 6.
            				cos_qscat[iwave,irad,igas] = cos_qscat[iwave,irad,i] / 6.
        
            

			
			
		
		else: 
			#get optics from database
			qext_gas, qscat_gas, cos_qscat_gas, nwave, radius = get_mie(igas,directory)
			
			if i==0: 
				nradii = len(radius)
				qext = np.zeros((nwave,nradii,ngas))
				qscat = np.zeros((nwave,nradii,ngas))
				cos_qscat = np.zeros((nwave,nradii,ngas))

			#add to master matrix 
			qext[:,:,i], qscat[:,:,i], cos_qscat[:,:,i] = qext_gas, qscat_gas, cos_qscat_gas


	kz, qt, qc, ndz, rg, reff = eddysed(atmo.t_top, atmo.p_top, atmo.t, atmo.p, 
		condensibles, gas_mw, gas_mmr, rho_p[i] , mmw, atmo.g, atmo.kz)

def eddysed( t_top, p_top,t_mid, p_mid, condensibles, gas_mw, gas_mmr,rho_p,
	 mw_atmos,gravity, kz, do_virtual=True, supsat=0):
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
		Array of gas mmw from `gas_properties`
	gas_mmr : ndarray 
		Array of gas mmr from `gas_properties`
	gravity : float 
		Gravity of planet cgs
	kz : float or ndarray
		Kzz in cgs, either float or ndarray depending of whether or not 
		it is set as input
	do_virtual : bool 
		(Optional) include decrease in condensate mixing ratio below model domain
	supsat : float
		(Optional) Default = 0 , Saturation factor (after condensation)

	Returns
	-------

	"""
	t_bot = t_top[-1]
	p_bot = p_top[-1]
	nz = len(t_mid)


	for i, igas in zip(range(len(condensibles)), condensibles):

		q_below = gas_mmr[i]


		#include decrease in condensate mixing ratio below model domain
		if do_virtual: 
			qvs_factor = (supsat+1)*gas_mw[i]/mw_atmos
			get_pvap = getattr(pvaps, igas)
			if igas == 'Mg2SiO4':
				pvap = get_pvap(t_bot, p_bot)
			else:
				pvap = get_pvap(t_bot)

			qvs = qvs_factor*pvap/p_bot	

			if qvs <= q_below :	
				#find the pressure at cloud base 
				return 'DO VIRTUAL NEEDS TO BE COMPLETED'
				#parameters for finding root 
			
		for iz in range(nz-1,-1,-1): #goes from BOA to TOA

			if not isinstance(kz,(float,int)): kz_in = kz[iz]
			else: kz_in = kz

			layer( igas, rho_p, t_mid[iz], p_mid[iz], 
				t_top[iz],t_top[iz+1], p_top[iz], p_top[iz+1],
				 kz_in, gravity, mw_atmos, gas_mw[i], q_below, supsat
			 )

	return kz, qt, qc, ndz, rg, reff 

def layer(gas_name,rho_p, t_layer, p_layer, t_top, t_bot, p_top, p_bot,
	kz, gravity, mw_atmos, gas_mw, q_below, supsat, kz_min = 1e5):
	"""
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

	#   parameter in Lennard-Jones potential for viscosity (K) (Rosner, 2000)
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
	mixl = np.max( [0.10, lapse_ratio ]) * scale_h


	#   scale factor for eddy diffusion: 1/3 is baseline
	scalef_kz = 1./3.

	#   vertical eddy diffusion coefficient (cm^2/s)
	#   from Gierasch and Conrath (1985)
	if kz == np.nan : 
		kz = (scalef_kz * scale_h * (mixl/scale_h)**(4./3.) * #when dont know kz
			  ( ( r_atmos*chf ) / ( rho_atmos*c_p ) )**(1./3.)) #when dont know kz

	#   no less than minimum value (for radiative regions)
	kz = np.max( [kz, kz_min])

	#   cloud fractional coverage
	#cloudf = (cloudf_min +
	#		  max( 0., min( 1., 1.-lapse_ratio )) *
	#	  ( 1. - cloudf_min ))

	#   atmospheric number density (molecules/cm^3)
	n_atmos = p_layer / ( K_BOLTZ*t_layer )

	#   atmospheric mean free path (cm)
	mfp = 1. / ( np.sqrt(2.)*n_atmos*PI*d_molecule**2 )

	#   atmospheric viscosity (dyne s/cm^2)
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
			p_sub = 0.5*( p_bot_sub + p_top_sub )
			t_sub = t_bot + np.log( p_bot/p_sub )*dtdlnp

			qt_top, qc_sub, qt_sub, rg_sub, reff_sub,ndz_sub= calc_qc(
					gas_name, supsat, t_sub, p_sub,r_atmos, r_cloud,
						qt_below, mixl, dz_sub, gravity,mw_atmos,mfp,visc,rho_p)
			
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
		elif (opd_layer == 0.) or (nsub > nsub_max): 
			converge = True
		elif ( abs( 1. - opd_test/opd_layer ) < 1e-2 ) : 
			converge = True
		else: 
			opd_test = opd_layer
		
		nsub = nsub * 2

	#   Update properties at bottom of next layer

	q_below = qt_top

	#Get layer averages

	if opd_layer > 0. : 
		reff_layer = 1.5*qc_layer / (rho_p*opd_layer)
		lnsig2 = 0.5*np.log( sig_layer )**2
		rg_layer = reff_layer*np.exp( -5*lnsig2 )
	else : 
		reff_layer = 0.
		rg_layer = 0.

	qc_layer = qc_layer*gravity / dp_layer
	qt_layer = qt_layer*gravity / dp_layer


def calc_qc(gas_name, supsat, t_layer, p_layer
	,r_atmos, r_cloud, q_below, mixl, dz_layer, gravity,mw_atmos,mfp,visc,rho_p):

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
		ad_rainf = rainf_all

		#   Find total condensate mixing ratio at top of layer
		qt_top = optimize.root_scalar(advdiff, bracket=[qlo, qhi], method='brentq', 
				args=(ad_qbelow,ad_qvs, ad_mixl,ad_dz ,ad_rainf))

		qt_top = qt_top.root

		#   Use trapezoid rule (for now) to calculate layer averages
		#   -- should integrate exponential
		qt_layer = 0.5*( q_below + qt_top )

		#   Diagnose condensate mixing ratio
		qc_layer = np.max( [0., qt_layer - qvs] )

		#   --------------------------------------------------------------------
		#   Find <rw> corresponding to <w_convect> using function vfall()

		#   range of particle radii to search (cm)
		rlo = 1.e-10
		rhi = 10.

		#   precision of vfall solution (cm/s)
		#delta_v = w_convect / 1000.

		rw_layer = optimize.root_scalar(vfall, bracket=[rlo, rhi], method='brentq', 
				args=(gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p))

		print(rw_layer.root)
		print('TRYING TO GET THE SAME ROOM HERE')
		


	return qt_top, qc_layer,qt_layer, rg_layer,reff_layer,ndz_layer 

class Atmosphere():
	def __init__(self,condensibles,mh=1,mmw=2.2) :
		"""
		Parameters
		----------
		mmw : float 
			MMW of the atmosphere 

		"""
		self.mh = 1
		self.mmw = 2.2
		self.condensibles = condensibles

	def get_pt(self, df = None, filename=None,**pd_kwargs):
		"""
		Read in file or define dataframe
	
		Parameters
		----------
		df : dataframe or dict
			Dataframe with "pressure"(bars),"temperature"(K),"z"(cm)
		filename : str 
			Filename read in. Will be read in with pd.read_csv and should 
			result in three named headers "pressure"(bars),"temperature"(K),"z"(cm). 
		pd_kwargs : kwargs
			Pandas key words for file read in. 
			If reading old style eddysed files, you would need: 
			skiprows=3, delim_whitespace=True, header=None, names=["ind","pressure","temperature","kzz"]
		"""
		if not isinstance(df, type(None)):
			if isinstance(df, dict): df = pd.DataFrame(df)
			df = df.sort_values('pressure')
			self.pressure = np.array(df['pressure'])
			self.temperature = np.array(df['temperature'])
		elif not isinstance(filename, type(None)):
			df = pd.read_csv(filename, **pd_kwargs)
			df = df.sort_values('pressure')
			self.pressure = np.array(df['pressure'])
			self.temperature = np.array(df['temperature'])

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


	def get_kz(self,df = None, filename = None, constant=None): 
		"""
		Define Kzz in CGS. Should be on same grid as pressure. 
			1) Defining DataFrame with keys 'pressure' (in bars), and 'kzz'
			2) Define constant kzz 
		"""
		if not isinstance(df, type(None)):
			print(df,type(df))
			self.kz = np.array(df['kzz'])
		elif not isinstance(df, type(None)):
			self.kz = constant
		else: 
			self.kz=0


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

	return qscat, qext, cos_qscat, nwave, radii


def get_r_grid(r_min=1e-5, n_radii=40):
	"""
	Get spacing of radii to run Mie code

	r_min : float 
		Minimum radius to compute 

	n_radii : int
		Number of radii to compute 
	"""
	vrat = 2.2 
	pw = 1. / 3.
	f1 = ( 2.0*vrat / ( 1.0 + vrat) )**pw
	f2 = (( 2.0 / ( 1.0 + vrat ) )**pw) * (vrat**pw-1.0)

	radius = r_min * vrat**(np.linspace(0,n_radii-1,n_radii)/3.)
	rup = f1*radius
	dr = f2*radius

	return radius, rup, dr



#INPUTS REQUIRED 

#directory of the refrind and mieff files
dirr = "/Users/natashabatalha/Documents/LP_Project/input/optics/"

#mean molecular weight of atmo
mmw = 2.2 
condensibles = ['H2O','Fe', 'Al2O3' ,'Na2S','NH3', 'KCl',
 				'MnS','ZnS','Cr' 'MgSiO3' ,'Mg2SiO4'  ]

#minimum eddy diffusion coefficient in cm^2/s
kzz_min = 1e5 

#geometric standard deviation of lognormal size distribution 
sig_all = 2.0 

#minimum cloud coverage (a diagnostic not applied to the clouds)
cloudf_min = 0.75 #keep tabs on this.... 

#minimum cloud coverage 
nsub_max = 64 #why two of these... 

#ramp up optical depth below cloud base in calc_optics()
do_subcloud = True 

#   saturation factor (after condensation)
supsat = 0

#rain factor 
rainf_all = 0.5

