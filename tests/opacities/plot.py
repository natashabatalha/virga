import matplotlib.pyplot as plt
import numpy as np

def get_file_data(file):
    data_file = open(file)
    data = []
    for line in data_file.readlines():
        info = line.split()
        data.append([float(info[1]), float(info[2]), float(info[3])])
    return np.asarray(data)

fe_data = get_file_data('Fe.refrind').T

fig, ax = plt.subplots(1, 2, figsize=(6,3))
ax[0].plot(fe_data[0], fe_data[1], label='Real part n')
ax[0].plot(fe_data[0], fe_data[2], label='Imaginary part k')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].legend()
ax[0].set_ylabel("Refractive index")
ax[0].set_xlabel(r"Wavelength [$\mu$m]")
fig.subplots_adjust(wspace=0, top=0.98, right=0.98, bottom=0.19, left=0.1)
plt.show()


