import numpy as np
import pandas as pd
import astropy.units as u
import virga.justdoit as jdi
import virga.justplotit as jpi
import matplotlib.pyplot as plt
import time
from bokeh.plotting import show, figure

#   locate data
mieff_directory = "~/Documents/codes/all-data/mieff_files"
fsed = 1.
Mark_data = False
refine_TP = True
quick_stop = True

if Mark_data:
    TP_directory = "~/Documents/codes/all-data/Mark_data/"
    filenames = ["t1000g1000nc_m0.0.dat", "t1500g1000nc_m0.0.dat", "t1700g1000f3_m0.0k.dat", \
                    "t200g3160nc_m0.0.dat", "t2400g3160nc_m0.0.dat"]
    filename = TP_directory + filenames[1]
    
    #   define atmosphere properties
    df = pd.read_csv(filename, delim_whitespace=True, usecols=[1,2], header=None)
    df.columns = ["pressure", "temperature"]
    pressure = np.array(df["pressure"])[1:]
    temperature = np.array(df["temperature"])[1:]
    grav = df["pressure"][0] * 100 
    kz = 1e9
    
    metallicity = 1 #atmospheric metallicity relative to Solar
    mean_molecular_weight = 2.2 # atmospheric mean molecular weight
    #get pyeddy recommendation for which gases to run
    recommended_gases = jdi.recommend_gas(pressure, temperature,
                                                 metallicity, mean_molecular_weight)
    
    a = jdi.Atmosphere([recommended_gases[0]], fsed=fsed, mh=metallicity, mmw=mean_molecular_weight)
    a.gravity(gravity=grav, gravity_unit=u.Unit('cm/(s**2)'))
    a.ptk(df = pd.DataFrame({'pressure':pressure, 'temperature':temperature,
                                   'kz':kz}))

else:
    metallicity = 1 #atmospheric metallicity relative to Solar
    mean_molecular_weight = 2.2 # atmospheric mean molecular weight
    
    #set the run 
    #a = jdi.Atmosphere(['MnS','Cr','MgSiO3','Fe'],
    a = jdi.Atmosphere(['MnS'],#, 'Cr'],
                      fsed=fsed,mh=metallicity,
                     mmw = mean_molecular_weight)
    
    #set the planet gravity
    a.gravity(gravity=7.460, gravity_unit=u.Unit('m/(s**2)'))
    
    #Get preset pt profile for testing
    a.ptk(df = jdi.hot_jupiter())

#   verify original and new solvers give same mixing ratios
labels = ["original", "new"]
lines = ["-", "--"]
solver = [True, False]
output = []
fig1, ax1 = plt.subplots()
for i in range(2):
    all_out = jdi.compute(a, as_dict=True, directory=mieff_directory, layers=solver[i], refine_TP=refine_TP, 
            quick_stop=quick_stop)
    output.append(all_out)
    pres = all_out['pressure']
    qt = all_out['cond_plus_gas_mmr'][:,0]
    qc = all_out['condensate_mmr'][:,0]

    ax1.loglog(qt, pres, lines[i], label="qt " + labels[i] )
    ax1.loglog(qc, pres, lines[i], label="qc " + labels[i] )
    ax1.loglog(qt-qc, pres, lines[i], label="qv " + labels[i] )

pres = output[0]["pressure"]
qc = output[0]["condensate_mmr"][:,0]
ax1.set_ylim(pres[len(pres)-1], pres[0])
#ax1.set_xlim([np.max([1e-9, np.min(qc*0.9)]), np.max(qc*1.1)])
ax1.set_ylabel("pressure")
ax1.legend(loc="best")
plt.show()

plt.ylim(pres[len(pres)-1], pres[0])
for i in range(2):
    pres_ = output[i]["pressure"]
    reff = output[i]["mean_particle_r"][:,0]
    plt.loglog(reff, pres_, lines[i], label="reff " + labels[i] )
plt.ylabel("pressure")
plt.xlabel("effective radius")
plt.legend(loc="best")
plt.show()



import sys; sys.exit()

show(jpi.opd_by_gas(output[0]))
show(jpi.opd_by_gas(output[1]))
import sys; sys.exit()

z = all_out['altitude']
reff = all_out["droplet_eff_r"][:,0]
rg = all_out["mean_particle_r"][:,0]
ndz = all_out["column_density"][:,0]
opd = all_out["opd_per_layer"]
w0 = all_out["single_scattering"]
g0 = all_out["asymmetry"]
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