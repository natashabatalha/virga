import numpy as np
import pandas as pd
import astropy.units as u
import pvaps
from picaso import justdoit as pj
import virga.justdoit as jdi

mieff_directory = "~/Documents/codes/all-data/mieff_files"
TP_directory = "~/Documents/codes/all-data/Mark_data/"
fsed = 0.05
gas_name = "Cr"
filenames = ["t1000g1000nc_m0.0.dat", "t1500g1000nc_m0.0.dat", "t1700g1000f3_m0.0k.dat", \
                "t200g3160nc_m0.0.dat", "t2400g3160nc_m0.0.dat"]
filename = TP_directory + filenames[0]
df = pd.read_csv(filename, delim_whitespace=True, usecols=[1,2], header=None)
df.columns = ["pressure", "temperature"]

pressure = np.array(df["pressure"])[1:]
temperature = np.array(df["temperature"])[1:]
grav = df["pressure"][0] * 100 
kz = 1e9

metallicity = 1 #atmospheric metallicity relative to Solar
mean_molecular_weight = 2.2 # atmospheric mean molecular weight

a = jdi.Atmosphere([gas_name],fsed=fsed)
a.gravity(gravity=grav, gravity_unit=u.Unit('cm/(s**2)'))
a.ptk(df = pd.DataFrame({'pressure':pressure, 'temperature':temperature,
                               'kz':kz}))

#   run virga with original mmr solver
all_out = jdi.compute(a, as_dict=True, directory=mieff_directory, layers=False, refine_TP=False)
pres = all_out['pressure']
qt = all_out['cond_plus_gas_mmr'][:,0]
qc = all_out['condensate_mmr'][:,0]
z = all_out['altitude']
temp = all_out["temperature"]
reff = all_out["droplet_eff_r"]
rg = all_out["mean_particle_r"]
ndz = all_out["column_density"]
opd = all_out["opd_per_layer"]
w0 = all_out["single_scattering"]
g0 = all_out["asymmetry"]
    
#all_out = jdi.compute(a, as_dict=True, directory=mieff_directory)#, layers=True)
#pres_layer = all_out['pressure']
#qt_layer = all_out['cond_plus_gas_mmr'][:,0]
#qc_layer = all_out['condensate_mmr'][:,0]
#z_layer = all_out['altitude']
#temp_layer = all_out["temperature"]

import matplotlib.pyplot as plt
plt.figure()
plt.ylim(pres[len(pres)-1], pres[0])
plt.loglog(qt, pres, '-', label="qt virga")
#plt.loglog(CR_qt, CR_pres, '--', label="qt caoimhe")
plt.loglog(qc, pres, '-', label="qc virga")
#plt.loglog(CR_qc, CR_pres, '--', label="qc caoimhe")
plt.loglog(qt-qc, pres, '-', label="qv virga")
#plt.loglog(CR_qt-CR_qc, CR_pres, '--', label="qv caoimhe")
plt.legend(loc="best")
plt.ylabel("pressure")
plt.show()
import sys; sys.exit()

plt.figure()
plt.loglog(reff, z, '-', label="effective radius")
plt.legend(loc="best")
plt.ylabel("altitude")
plt.show()

plt.figure()
plt.loglog(rg, z, '-', label="mean radius")
plt.legend(loc="best")
plt.ylabel("altitude")
plt.show()

plt.figure()
plt.loglog(ndz, z, '-', label="column density")
plt.legend(loc="best")
plt.ylabel("altitude")
plt.show()

plt.figure()
plt.loglog(opd, z, '-', label="optical density")
plt.legend(loc="best")
plt.ylabel("altitude")
plt.show()

plt.figure()
plt.loglog(w0, z, '-', label="single scattering")
plt.legend(loc="best")
plt.ylabel("altitude")
plt.show()

plt.figure()
plt.loglog(g0, z, '-', label="asymmetry")
plt.legend(loc="best")
plt.ylabel("altitude")
plt.show()
