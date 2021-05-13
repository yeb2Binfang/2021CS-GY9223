# visualizing SVM boundaries for 4 emotional categories
# PVHA, PVLA, NVHA, NVLA,
from __future__ import print_function

import matplotlib

matplotlib.rcParams.update({'font.size': 16})
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# processing the data
X = np.transpose(X, [3, 0, 1, 2])
X.shape
# average over time
X1 = np.mean(X, axis=2)

# Extracting beta, lower gamma and higher gamma bands and averaging each separately
temp1 = np.mean(X1[:, 13:30, :], axis=1)
temp2 = np.mean(X1[:, 31:50, :], axis=1)
temp3 = np.mean(X1[:, 51:79, :], axis=1)

X = np.concatenate((temp1, temp2, temp2), axis=1)
print(X.shape)

# parameters
n_classes = 4
C = 100

# split into train/test
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.20, random_state=2)

# Create the scaler objects
xscal = StandardScaler()

# Fit and transform the data
Xtr1 = xscal.fit_transform(Xtr)
Xts1 = xscal.transform(Xts)

# Fit PCA on the training data
pca = PCA(n_components=25, svd_solver='randomized', whiten=True)
pca.fit(Xtr1)

# Transform the training and test
Ztr = pca.transform(Xtr1)
Zts = pca.transform(Xts1)

k = 1
h = .2  # step size in the mesh
# create a mesh to plot in
x_min, x_max = Ztr[:, k].min() - 1, Ztr[:, k].max() + 1
y_min, y_max = Ztr[:, k + 1].min() - 1, Ztr[:, k + 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
svc = SVC(C=C, kernel='rbf', gamma=0.1)
svc.fit(Ztr[:, k:k + 2], ytr.astype(int))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

n = 5  # number of components to visualize
svc = SVC(C=C, kernel='rbf', gamma=0.1)
fig, axes = plt.subplots(figsize=(8, 8), sharex=True, sharey=True, ncols=n - 1, nrows=n - 1)
row = 0
levels = [0, 1, 2, 3, 4]
for i in range(n):
    col = 0
    for j in range(n):
        x_min, x_max = Ztr[:, i].min() - 1, Ztr[:, i].max() + 1
        y_min, y_max = Ztr[:, j].min() - 1, Ztr[:, j].max() + 1
        svc.fit(Ztr[:, [i, j]], ytr.astype(int))
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if i > j:
            im = axes[row, col].contourf(xx, yy, Z.astype(int) + 1, levels=levels, cmap=plt.cm.coolwarm, alpha=0.8)
            # Plot also the training points
            axes[row, col].scatter(Ztr[:, i], Ztr[:, j], s=2, c=ytr.astype(int), cmap=plt.cm.coolwarm)
            axes[row, col].set_xlabel('component ' + str(i + 1), fontsize=10)
            axes[row, col].set_ylabel('component ' + str(j + 1), fontsize=10)
            axes[row, col].set_xlim(xx.min(), xx.max())
            axes[row, col].set_ylim(yy.min(), yy.max())
            axes[row, col].set_xticks(())
            axes[row, col].set_yticks(())
        if row < col and row < n - 1 and col < n - 1:
            axes[row, col].axis('off')
        if j == n - 2:
            if i != 0 and j != 0:
                row = row + 1
        col = col + 1
fig.subplots_adjust(right=0.8)
# put colorbar at desire position
cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.78])
fig.colorbar(im, cax=cbar_ax, ticks=[0.5, 1.5, 2.5, 3.5])  #
cbar_ax.set_yticklabels(classNames)  # vertically oriented colorbar
plt.savefig('FIGURES/svm_5_comp_4_classes.pdf')
plt.savefig('FIGURES/svm_5_comp_4_classes.png')

n = 25  # number of components to visualize
svc = SVC(C=C, kernel='rbf', gamma=0.1)
fig, axes = plt.subplots(figsize=(18, 8), sharex=True, sharey=True, ncols=n - 1, nrows=n - 1)
row = 0
levels = [0, 1, 2, 3, 4]
for i in range(n):
    col = 0
    for j in range(n):
        x_min, x_max = Ztr[:, i].min() - 1, Ztr[:, i].max() + 1
        y_min, y_max = Ztr[:, j].min() - 1, Ztr[:, j].max() + 1
        svc.fit(Ztr[:, [i, j]], ytr.astype(int))
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if i > j:
            im = axes[row, col].contourf(xx, yy, Z.astype(int) + 1, levels=levels, cmap=plt.cm.coolwarm, alpha=0.8)
            # Plot also the training points
            axes[row, col].scatter(Ztr[:, i], Ztr[:, j], s=0.05, marker=".", c=ytr.astype(int), cmap=plt.cm.coolwarm)
            axes[row, col].set_xlabel(str(i + 1), fontsize=5)
            axes[row, col].set_ylabel(str(j + 1), fontsize=5, labelpad=-0.2)
            axes[row, col].set_xlim(xx.min(), xx.max())
            axes[row, col].set_ylim(yy.min(), yy.max())
            axes[row, col].set_xticks(())
            axes[row, col].set_yticks(())
        if row < col and row < n - 1 and col < n - 1:
            axes[row, col].axis('off')
        if j == n - 2:
            if i != 0 and j != 0:
                row = row + 1
        col = col + 1
fig.subplots_adjust(right=0.8)
# put colorbar at desire position
cbar_ax = fig.add_axes([0.85, 0.11, 0.02, 0.78])
fig.colorbar(im, cax=cbar_ax, ticks=[0.5, 1.5, 2.5, 3.5])
cbar_ax.set_yticklabels(classNames)  # vertically oriented colorbar
plt.savefig('FIGURES/svm_25_comp_4_classes.pdf')
plt.savefig('FIGURES/svm_25_comp_4_classes.png', dpi=400)
