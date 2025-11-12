
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from virga import justdoit as jdi
from picaso import justdoit as jdip

from wasp107_data import get_wasp107_data

# ==== inputs
mieff_directory = 'opacities/'
poffset = 1
tcase = get_wasp107_data()
species = ['TiO2', 'Fe', 'Mg2SiO4']
densities = [4.25, 7.875, 3.214]
fsed = 0.1
species = ['Mg2SiO4']
densities = [3.214]

# ==== Cloud model run
# set up the atmosphere
sum_planet = jdi.Atmosphere(species, fsed=fsed, mh=tcase['mh_planet'],
                            mmw=tcase['mmw'], mixed=False)
sum_planet.gravity(gravity=tcase['gravity'], gravity_unit=u.Unit('cm/(s**2)'))
pd.DataFrame({'pressure':tcase['pressure_bar'],
              'temperature':tcase['temperature'],
              'kz':tcase['kzz']}).to_csv('struct.csv')
sum_planet.ptk(filename='struct.csv', usecols = [1,2,3])
# run calculation
all_out = jdi.compute(sum_planet, directory=mieff_directory, as_dict=True, quick_mix=False)
# opacity calc for picaso
df_cloud = jdi.picaso_format(
    all_out['opd_per_layer'], all_out['single_scattering'], all_out['asymmetry'],
    pressure=all_out['pressure'], wavenumber=1/all_out['wave']/1e-4,
)

# ==== Set up Picaso class
opa = jdip.opannection(wave_range=[0, 25])
case1 = jdip.inputs()
case1.phase_angle(0)
case1.gravity(mass=tcase['M_planet'], mass_unit=u.Unit('M_jup'),
              radius=tcase['R_planet'], radius_unit=u.Unit('R_jup'))
case1.star(opa, tcase['T_star'], tcase['mh_star'], tcase['logg_star'],
           radius=tcase['R_star'], radius_unit=jdip.u.Unit('R_sun'))

# ==== Get atmosphere from tcase
d = {'pressure': tcase['pressure_bar'], 'temperature': tcase['temperature']}
for key in tcase['chemistry']:
    d[key] = tcase['chemistry'][key]
df = pd.DataFrame(data=d)
case1.atmosphere(df=df)

# ==== Add clouds if given
case1.clouds(df=df_cloud)

# ==== Calculate transmission and emission spectra
t_df = case1.spectrum(opa, full_output=True, calculation='transmission')

# ==== Regrid output
trans_wavn, trans_rprs2 = t_df['wavenumber'], t_df['transit_depth']
trans_wavn_bins, trans_wavn_rprs2 = jdip.mean_regrid(trans_wavn, trans_rprs2, R=150)

plt.figure()
plt.plot(1e4/trans_wavn_bins, trans_wavn_rprs2)
plt.xscale('log')
plt.show()


# ==== Cloud model run
# set up the atmosphere
sum_planet = jdi.Atmosphere(species, fsed=fsed, mh=tcase['mh_planet'],
                            mmw=tcase['mmw'], mixed=True)
sum_planet.gravity(gravity=tcase['gravity'], gravity_unit=u.Unit('cm/(s**2)'))
pd.DataFrame({'pressure':tcase['pressure_bar'],
              'temperature':tcase['temperature'],
              'kz':tcase['kzz']}).to_csv('struct.csv')
sum_planet.ptk(filename='struct.csv', usecols = [1,2,3])
# run calculation
all_out = jdi.compute(sum_planet, directory=mieff_directory, as_dict=True, quick_mix=False)
# opacity calc for picaso
df_cloud = jdi.picaso_format(
    all_out['opd_per_layer'], all_out['single_scattering'], all_out['asymmetry'],
    pressure=all_out['pressure'], wavenumber=1/all_out['wave']/1e-4,
)

# ==== Set up Picaso class
opa = jdip.opannection(wave_range=[0, 25])
case1 = jdip.inputs()
case1.phase_angle(0)
case1.gravity(mass=tcase['M_planet'], mass_unit=u.Unit('M_jup'),
              radius=tcase['R_planet'], radius_unit=u.Unit('R_jup'))
case1.star(opa, tcase['T_star'], tcase['mh_star'], tcase['logg_star'],
           radius=tcase['R_star'], radius_unit=jdip.u.Unit('R_sun'))

# ==== Get atmosphere from tcase
d = {'pressure': tcase['pressure_bar'], 'temperature': tcase['temperature']}
for key in tcase['chemistry']:
    d[key] = tcase['chemistry'][key]
df = pd.DataFrame(data=d)
case1.atmosphere(df=df)

# ==== Add clouds if given
case1.clouds(df=df_cloud)

# ==== Calculate transmission and emission spectra
t_df = case1.spectrum(opa, full_output=True, calculation='transmission')

# ==== Regrid output
trans_wavn, trans_rprs2 = t_df['wavenumber'], t_df['transit_depth']
trans_wavn_binsxx, trans_wavn_rprs2xx = jdip.mean_regrid(trans_wavn, trans_rprs2, R=150)

plt.figure()
plt.plot(1e4/trans_wavn_binsxx, trans_wavn_rprs2xx)
plt.xscale('log')
plt.show()

plt.figure()
plt.plot(1e4/trans_wavn_binsxx, np.abs(trans_wavn_rprs2 - trans_wavn_rprs2xx)/trans_wavn_rprs2)
plt.xscale('log')
plt.show()

# ==== Get the data
# mass mixing ratio
qc = all_out['condensate_mmr']
# pressures
pres = all_out['pressure']
# # calculate volume mixing ratios
# a = qc / np.asarray(densities)[np.newaxis, :]
# b = np.sum(a, axis=1)
# vmr = a/b[:, np.newaxis]

# # ==== plot the profiles
# plt.figure()
# plt.plot(qc, all_out['pressure'], label=species)
# plt.ylim(bottom=tcase['pressure_bar'][-1], top=tcase['pressure_bar'][0])
# plt.yscale('log')
# plt.legend()
# plt.show()