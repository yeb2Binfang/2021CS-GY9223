# dimensionality reduction and visualization with  PCA and t-SNE
#https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
#https://distill.pub/2016/misread-tsne/
from __future__ import print_function

import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.rcParams.update({'font.size': 16})
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
temp1 = np.mean(X1[:, 13:30, :], axis=1)
temp2 = np.mean(X1[:, 31:50, :], axis=1)
temp3 = np.mean(X1[:, 51:79, :], axis=1)

X = np.concatenate((temp1, temp2, temp2), axis=1)
print(X.shape)


########################################################################################################################
#
# VISUALIZING WITH PCA
#
########################################################################################################################
numComponents = 3
pca = PCA(n_components=numComponents)
pca_result = pca.fit_transform(X)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

rndperm = np.random.permutation(X.shape[0])
feat_cols = [str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns = feat_cols)
y_cols_names = [classNames[int(i)] for i in y]

pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
df['y_numeric'] = y
df['y'] = y_cols_names
sns.set(font_scale=1,style="white")
#plot one component vs another:
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data = df,
    legend="full",
    alpha=0.3
)

# plot all components vs each other
plt.figure(figsize=(15,6))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df,
    legend="full",
    alpha=0.5,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="pca-one", y="pca-three",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df,
    legend="full",
    alpha=0.5,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="pca-two", y="pca-three",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df,
    legend="full",
    alpha=0.5,
    ax=ax3
)
plt.tight_layout()
plt.savefig("FIGURES/DIMENSIONALITY_REDUCTION/pca2D_comp_"+str(numComponents)+"_classes_"+str(numClasses)+".pdf")


#plot 3D PCA
ax = plt.figure(figsize=(6,6)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"],
    ys=df.loc[rndperm,:]["pca-two"],
    zs=df.loc[rndperm,:]["pca-three"],
    c=df.loc[rndperm,:]["y_numeric"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.tight_layout()
plt.savefig("FIGURES/DIMENSIONALITY_REDUCTION/pca3D_comp_"+str(numComponents)+"_classes_"+str(numClasses)+".pdf")

########################################################################################################################
#
# VISUALIZING WITH tSNE
#
########################################################################################################################

numComponents=3
df_subset = df.copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=numComponents)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1]
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=numComponents, verbose=1, perplexity=40, n_iter=5000)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['tsne-2d-three'] = tsne_results[:,2]

#plot one component vs another:
plt.figure(figsize=(6,6))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.5
)

#plot all components vs each other:
plt.figure(figsize=(15,6))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.5,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-three",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.5,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-2d-two", y="tsne-2d-three",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.5,
    ax=ax3
)
plt.tight_layout()
plt.savefig("FIGURES/DIMENSIONALITY_REDUCTION/tSNE2D_comp_"+str(numComponents)+"_classes_"+str(numClasses)+".pdf")

#plot 3D t-SNE
ax = plt.figure(figsize=(6,6)).gca(projection='3d')
ax.scatter(
    xs=df_subset.loc[rndperm,:]["tsne-2d-one"],
    ys=df_subset.loc[rndperm,:]["tsne-2d-two"],
    zs=df_subset.loc[rndperm,:]["tsne-2d-three"],
    c=df_subset.loc[rndperm,:]["y_numeric"],
    cmap='tab10'
)
ax.set_xlabel('tsne-2d-one')
ax.set_ylabel('tsne-2d-two')
ax.set_zlabel('tsne-2d-three')
plt.tight_layout()
plt.savefig("FIGURES/DIMENSIONALITY_REDUCTION/tSNE3D_comp_"+str(numComponents)+"_classes_"+str(numClasses)+".pdf")


########################################################################################################################
#
# PLOT tSNE vs PCA
#
########################################################################################################################
plt.figure(figsize=(10,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
plt.savefig("FIGURES/DIMENSIONALITY_REDUCTION/tSNE_PCA2D_comp_"+str(numComponents)+"_classes_"+str(numClasses)+".pdf")

########################################################################################################################
#
# PERFORM tSNE on PCA
#
########################################################################################################################
pca_12 = PCA(n_components=12)
pca_result_12 = pca_12.fit_transform(data_subset)
print('Cumulative explained variation for 12 principal components: {}'.format(np.sum(pca_12.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=5000)
tsne_pca_results = tsne.fit_transform(pca_result_12)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset['tsne-pca12-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca12-two'] = tsne_pca_results[:,1]

plt.figure(figsize=(15,6))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca12-one", y="tsne-pca12-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)
plt.tight_layout()
plt.savefig("FIGURES/DIMENSIONALITY_REDUCTION/tSNE_PCA2D_comp_12_3_classes_"+str(numClasses)+".pdf")
