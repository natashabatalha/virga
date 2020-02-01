import numpy as np

def init_optics(condensibles, nrad=40, rmin=1e-10, read_mie=True): 
	"""
	Setup up a particle size grid and calculate single-particle scattering
	and absorption efficiencies and other parameters to be used by
	`calc_optics()`

	Parameters
	----------
	do_optics : bool 
		(True/False) Calculate optics (T) or use pre computed files (F)
	read_mie : bool 
		(True/False) Read in Mie coefficients from pre compute files `gas_name.mieff`
	condensibles : list of str
		Name in str of all condensible gases e.g. ['H2O','CH4']
	nrad : int 
		Number of radius grid points 
	rmin : float 
		Minimum number of radius grid (cm) 

	Returns
	-------
	wave : array 
		Wavelength bin centers (cm)
	radius : array
		Radius bin centers (cm)
	dr : array 
		Widths of radius bins (cm)
	qscat : array 
		Scattering efficiency 
	qext : array 
		Extinction efficiency
	cos_qscat : array
		qscat * acerage <cos (scattering angle)> 
	"""
	#equations to compute radius bins for particles 
	#these used to be a matrix with different min radii 

	vrat = 2.0
	pw = 1. / 3.
	f1 = ( 2*vrat / ( 1 + vrat) )**pw
	f2 = ( 2 / ( 1 + vrat ) )**pw *  (vrat**pw - 1) 
	radius = rmin * vrat**(irad/3.)
	rup = f1*radius
	dr = f2*radius

	if read_mie: 
		#Read extinction and scattering coefficients 
		#for each condensing vapor
		wave, qscat, qext, cos_qscat = get_meiff()

	else: 
		#Calculate single-scattering efficiencies etc from refractive indices
		#for each condensing vapor

		#Mie parameters:
		#thetd is angle between incident and scattered radiation
		#n_thetd is number of thetd values to consider

		thetd = 0.0
		n_thetd = 1

		#read in refractive indices
		for gas in condensibles: 
			micron_wave, nn, kk = get_refrind(gas)

			cm_wave = micron_wave*1e-4

			wvno = np.pi*2/cm_wave

			for irad in range(nrad):

				#subdivide radius grid into 6 bin to average
				#out oschilations and call Mie code
				if i == 0 : 
					dr5 = (rup[0] - radius[0])/5
					rr = radius[0]
				else:
					dr5 = (rup[irad] - rup[irad-1])/5
					rr = rup[irad-1]

				corerad = 0.0 
				corereal = 1.0
				coreimag = 0.0

				for isub in range(6):
					#should return something that is wave x radius
					qext, qscat, cos_qscat = mie_calc(rr, nn, kk, thetd,
											n_thetd, corerad, corereal, coreimag, wvno)




def get_refrind(condensible, directory='~/Documents/eddysed/input/optics'):
	"""
	Read old style refrind files 

	Parameters
	----------
	condensible : str 
		Condensible name (e.g. Al2O3)

	Returns
	-------
	micron_wave, nn, kk as ndarrays
	"""
	df = pd.read_csv(os.path.join(directory,condensible+'.refrind'),
		skiprows=2, header=None,delim_whitespace=True,
                names=['i', 'wavelength', 'nn', 'kk'])
	micron_wave=df['wavelength'].values
	nn = df['nn'].values
	kk = df['kk'].values
	return micron_wave, nn, kk 









