import numpy as np
import csv
import os
import astropy.units as u
import pandas as pd

# ==== Important papers:
# P1) https://www.aanda.org/articles/aa/full_html/2017/08/aa30439-17/aa30439-17.html (discovery paper)
# P2) https://iopscience.iop.org/article/10.3847/1538-3881/abcd3c/pdf (Stellar parameters)
# Related papers
# Pe1) https://arxiv.org/pdf/2505.13602 (Large wavelength analysis)

# ==== Planet data
tcase = {}
tcase['gravity'] = 10**2.49  # cgs  (Anderson et al. 2017)
tcase['M_planet'] = 0.12  # in jupiter mass (Anderson et al. 2017)
tcase['R_planet'] = 0.94  # in jupiter radii (Anderson et al. 2017)
tcase['mh_planet'] = 1 # not used
tcase['mmw'] = 2.34  # (assumed)
tcase['sig'] = 2  # assumed

# ==== Stellar data
tcase['R_star'] = 0.67  # in solar radii (Piaulet et al. 2022)
tcase['T_star'] = 4425  # in K (Piaulet et al. 2022)
tcase['mh_star'] = 0.02  # in solar metalicity (Piaulet et al. 2022)
tcase['logg_star'] = 4.633  # in cgs (Piaulet et al. 2022)
tcase['distance'] = 64.7  # not used

# ==== Default values
tcase['am_fitting'] = False

# chemistry from ARCiS fit
tcase['add_gas_phase_species_constant'] = {
    'H2O': 10**-2.19,
    'SO2': 10**-5.03,  # not available
    'H2S': 10**-2.65,
    'NH3': 10**-5.47,
    'CO': 10**-2.41,
    'PH3': 10**-6.29,  # insignificant
    'HCN': 10**-9.26,  # insignificant, not available
    'C2H2': 10**-9.08,  # insignificant, not available
    'SiO': 10**-6.08,  # insignificant, not available
    'CH4': 10**-8.52,  # insignificant
    'CO2': 10**-8.05,  # insignificant
    'SO': 10**-7.38,  # insignificant, not available
}

# ==== temperature, pressure, kzz, chem vmr
reader = csv.reader(open(os.path.dirname(os.path.abspath(__file__)) + "/data/wasp107.dat"), delimiter="\t")
vals = []
for r, row in enumerate(reader):
    content = row[0].split()
    vals.append(content)
# read in temperature, pressure, kzz
data = np.asarray(vals[73:133]).T
pres_mid = np.asarray([float(i) for i in data[0]])
tcase['pressure_bar'] = pres_mid
tcase['temperature'] = np.asarray([float(i) for i in data[2]])
tcase['kzz'] = np.asarray([float(i) for i in data[3]])
data = np.asarray(vals[2:63]).T
chem = {}
pres_edge = np.asarray([float(i) for i in data[1]])
for n, name in enumerate(vals[1][4:]):
    fac = 1
    if name == 'HE':
        continue
    edge_vals = np.asarray([float(i) for i in data[4 + n]]) * fac
    mid_vals = (edge_vals[1:] - edge_vals[:1]) / (pres_edge[1:] - pres_edge[:1]) * (
                pres_mid - pres_edge[:1]) + edge_vals[:1]
    chem[name] = mid_vals
oa = np.ones_like(pres_mid)
if 'add_gas_phase_species_constant' in tcase:
    for chem_spec in tcase['add_gas_phase_species_constant']:
        chem[chem_spec] = oa * tcase['add_gas_phase_species_constant'][chem_spec]
tcase['chemistry'] = chem
tcase['chem_original'] = tcase['chemistry'].copy()

# ==== JWST data of WASP-107b

jwst_data_miri = csv.reader(
    open(os.path.dirname(os.path.abspath(__file__)) + "/data/MIRI_LRS_Fiducial_Eureka_Spectrum.txt"), delimiter="\t"
)
miri_vals = []
for r, row in enumerate(jwst_data_miri):
    if r == 0:
        continue
    content = row[0].split()
    data = [float(i) for i in content]
    data.pop(1)
    miri_vals.append(data)
all_data = np.asarray(miri_vals)

# jwst_data_nircamf3 = csv.reader(
#     open(os.path.dirname(os.path.abspath(__file__)) + "/data/NIRCam_F322W2_Fiducial_Eureka_Spectrum.txt"), delimiter="\t"
# )
# nircamf3 = []
# for r, row in enumerate(jwst_data_nircamf3):
#     if r == 0:
#         continue
#     content = row[0].split()
#     data = [float(i) for i in content]
#     data.pop(1)
#     nircamf3.append(data)
# all_data = np.concatenate((nircamf3 ,all_data))
#
# jwst_data_nircamf4 = csv.reader(
#     open(os.path.dirname(os.path.abspath(__file__)) + "/data/NIRCam_F444W_Fiducial_Eureka_Spectrum.txt"), delimiter="\t"
# )
# nircamf4 = []
# for r, row in enumerate(jwst_data_nircamf4):
#     if r == 0:
#         continue
#     content = row[0].split()
#     data = [float(i) for i in content]
#     data.pop(1)
#     nircamf4.append(data)
# all_data = np.concatenate((nircamf4, all_data))

jwst_data_wfc3g102 = csv.reader(
    open(os.path.dirname(os.path.abspath(__file__)) + "/data/WFC3_G102_Fiducial_Pegasus_Spectrum.txt"), delimiter="\t"
)
wfc3g102 = []
for r, row in enumerate(jwst_data_wfc3g102):
    if r == 0:
        continue
    content = row[0].split()
    data = [float(i) for i in content]
    data.pop(1)
    wfc3g102.append(data)
additional_data = np.asarray(wfc3g102)

jwst_data_wfc3g141 = csv.reader(
    open(os.path.dirname(os.path.abspath(__file__)) + "/data/WFC3_G141_Fiducial_Pegasus_Spectrum.txt"), delimiter="\t"
)
wfc3g141 = []
for r, row in enumerate(jwst_data_wfc3g141):
    if r == 0:
        continue
    content = row[0].split()
    data = [float(i) for i in content]
    data.pop(1)
    wfc3g141.append(data)
additional_data = np.concatenate((wfc3g141, additional_data))

ms = all_data[:, 0].argsort()
all_data = all_data[ms, :]
tcase['jwst_data'] = np.asarray(all_data)
tcase['jwst_data_error'] = all_data[:, 2] / all_data[:, 1]

ms = additional_data[:, 0].argsort()
additional_data = additional_data[ms, :]
tcase['jwst_data_additional'] = np.asarray(additional_data)
tcase['jwst_data_additional_error'] = additional_data[:, 2] / additional_data[:, 1]

# ==== Function to get the data
def get_wasp107_data():
    return tcase
