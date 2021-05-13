# Visualize raw time series with 4 classes
# 'PVHA', 'PVLA', 'NVHA', 'NVLA'
from __future__ import print_function

import matplotlib

matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
import numpy as np
from loadData import loadData

# Importing the data files
numClasses = 4
if numClasses == 12:
    classNames = ['PVHA_i1', 'PVHA_i2', 'PVHA_i3', 'PVLA_i1', 'PVLA_i2', 'PVLA_i3', 'NVHA_i1', 'NVHA_i2', 'NVHA_i3',
                  'NVLA_i1', 'NVLA_i2', 'NVLA_i3']
else:
    classNames = ['PVHA', 'PVLA', 'NVHA', 'NVLA']

path = 'DATA/DATA_' + str(numClasses) + '/'

(X, y) = loadData(path, numClasses)
X = np.transpose(X, [3, 0, 1, 2])
X.shape
# average over time
X1 = np.mean(X, axis=2)

# Extracting beta, lower gamma and higher gamma bands and averaging each separately
temp1 = np.mean(X[:, 13:30, :, :], axis=1)
temp2 = np.mean(X[:, 31:50, :, :], axis=1)
temp3 = np.mean(X[:, 51:79, :, :], axis=1)

########################################################################################################################
# visualize the average activations for every frequency bands
########################################################################################################################
a0 = np.mean(temp1[y == 0, :, :], axis=0)
a1 = np.mean(temp1[y == 1, :, :], axis=0)
a2 = np.mean(temp1[y == 2, :, :], axis=0)
a3 = np.mean(temp1[y == 3, :, :], axis=0)

a00 = np.mean(a0, axis=1)
a01 = np.mean(a1, axis=1)
a02 = np.mean(a2, axis=1)
a03 = np.mean(a3, axis=1)
ax = plt.figure(figsize=(10, 6))
plt.plot(a00, linewidth=2, color='b', label='PVHA')
plt.plot(a01, linewidth=2, color='r', label='PVLA')
plt.plot(a02, linewidth=2, color='g', label='NVHA')
plt.plot(a03, linewidth=2, color='k', label='NVLA')
plt.legend()
plt.title("Power spectral density averaged over sensors and trials for beta band (13-30 Hz)", fontsize=15)
plt.xlabel("timebins, 100 msec")
plt.ylabel("power spectral density $\mu V^2$/Hz")
plt.tight_layout()
plt.savefig("FIGURES/SPECTRAL_DENSITIES/psdBeta_" + str(numClasses) + ".pdf")

b0 = np.mean(temp2[y == 0, :, :], axis=0)
b1 = np.mean(temp2[y == 1, :, :], axis=0)
b2 = np.mean(temp2[y == 2, :, :], axis=0)
b3 = np.mean(temp2[y == 3, :, :], axis=0)
b00 = np.mean(b0, axis=1)
b01 = np.mean(b1, axis=1)
b02 = np.mean(b2, axis=1)
b03 = np.mean(b3, axis=1)
ax = plt.figure(figsize=(10, 6))
plt.plot(b00, linewidth=2, color='b', label='PVHA')
plt.plot(b01, linewidth=2, color='r', label='PVLA')
plt.plot(b02, linewidth=2, color='g', label='NVHA')
plt.plot(b03, linewidth=2, color='k', label='NVLA')
plt.legend()
plt.title("Power spectral density averaged over sensors and trials for lower gamma band (31-50 Hz)", fontsize=15)
plt.xlabel("timebins, 100 msec")
plt.ylabel("power spectral density $\mu V^2$/Hz")
plt.tight_layout()
plt.savefig("FIGURES/SPECTRAL_DENSITIES/psdLowGamma_" + str(numClasses) + ".pdf")

c0 = np.mean(temp3[y == 0, :, :], axis=0)
c1 = np.mean(temp3[y == 1, :, :], axis=0)
c2 = np.mean(temp3[y == 2, :, :], axis=0)
c3 = np.mean(temp3[y == 3, :, :], axis=0)
c00 = np.mean(c0, axis=1)
c01 = np.mean(c1, axis=1)
c02 = np.mean(c2, axis=1)
c03 = np.mean(c3, axis=1)
ax = plt.figure(figsize=(10, 6))
plt.plot(c00, linewidth=2, color='b', label='PVHA')
plt.plot(c01, linewidth=2, color='r', label='PVLA')
plt.plot(c02, linewidth=2, color='g', label='NVHA')
plt.plot(c03, linewidth=2, color='k', label='NVLA')
plt.legend()
plt.title("Power spectral density averaged over sensors and trials for higher gamma band (51-79 Hz)", fontsize=15)
plt.xlabel("timebins, 100 msec")
plt.ylabel("power spectral density $\mu V^2$/Hz")
plt.tight_layout()
plt.savefig("FIGURES/SPECTRAL_DENSITIES/psdHighGamma_" + str(numClasses) + ".pdf")

# plot all together:
ax = plt.figure(figsize=(10, 6))
plt.plot(a00, linewidth=2, color='b', label='PVHA')
plt.plot(a01, linewidth=2, color='r', label='PVLA')
plt.plot(a02, linewidth=2, color='g', label='NVHA')
plt.plot(a03, linewidth=2, color='k', label='NVLA')
plt.plot(b00, linewidth=2, color='b', label='PVHA')
plt.plot(b01, linewidth=2, color='r', label='PVLA')
plt.plot(b02, linewidth=2, color='g', label='NVHA')
plt.plot(b03, linewidth=2, color='k', label='NVLA')
plt.plot(c00, linewidth=2, color='b', label='PVHA')
plt.plot(c01, linewidth=2, color='r', label='PVLA')
plt.plot(c02, linewidth=2, color='g', label='NVHA')
plt.plot(c03, linewidth=2, color='k', label='NVLA')
plt.legend()
plt.title("Power spectral density averaged over sensors and trials", fontsize=15)
plt.xlabel("timebins, 100 msec")
plt.ylabel("power spectral density $\mu V^2$/Hz")
plt.tight_layout()
plt.savefig("FIGURES/SPECTRAL_DENSITIES/psdAll_" + str(numClasses) + ".pdf")

########################################################################################################################
# activation depending on frequency
########################################################################################################################
# average over the sensors
X2 = np.mean(X1, axis=2)
X2.shape

########################################################################################################################
tmpX_0 = np.mean(X2[y == 0, :], axis=0)
tmpX_1 = np.mean(X2[y == 1, :], axis=0)
tmpX_2 = np.mean(X2[y == 2, :], axis=0)
tmpX_3 = np.mean(X2[y == 3, :], axis=0)

ax = plt.figure(figsize=(10, 6))
plt.plot(tmpX_0, linewidth=2, color='b', label='PVHA')
plt.plot(tmpX_1, linewidth=2, color='r', label='PVLA')
plt.plot(tmpX_2, linewidth=2, color='g', label='NVHA')
plt.plot(tmpX_3, linewidth=2, color='k', label='NVLA')
plt.legend()
plt.title("Power spectral density vs frequency averaged over sensors and trials", fontsize=15)
plt.xlabel("frequency, Hz")
plt.ylabel("power spectral density $\mu V^2$/Hz")
plt.tight_layout()
plt.savefig("FIGURES/SPECTRAL_DENSITIES/psdFreq" + str(numClasses) + ".pdf")
