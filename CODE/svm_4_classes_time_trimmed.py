# SVM model with 4 classes first 10 time bins trimmed dataset
# 10-fold CV of CNN on the dataset with 4 classes
# PVHA, PVLA, NVHA, NVLA,
from __future__ import print_function

import matplotlib

matplotlib.rcParams.update({'font.size': 16})
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
from loadData import loadData
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

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

# average over trimmed time
X1 = np.mean(X[:, :, 10:, :], axis=2)

# Extracting beta, lower gamma and higher gamma bands and averaging each separately
temp1 = np.mean(X1[:, 13:30, :], axis=1)
temp2 = np.mean(X1[:, 31:50, :], axis=1)
temp3 = np.mean(X1[:, 51:79, :], axis=1)

X = np.concatenate((temp1, temp2, temp2), axis=1)
print(X.shape)

# parameters
n_classes = 4
nfold = 10
npc = 25
gam = 0.1
C = 100

## 10 fold cross validation with optimal parameters npcs = 25 gamma = 0.1
# Create cross-validation object
kf = KFold(nfold, shuffle=True)

# Create the scaler objects
xscal = StandardScaler()
yscal = StandardScaler()

# Run the cross-validation
rsq = np.zeros(nfold)
acc = np.zeros(nfold)
yhatPerFold = []
ytsPerFold = []
accuracyTrain = []
accuracyVal = []
confusionMatrices = []
reports = []
for ifold, ind in enumerate(kf.split(X)):
    print('Fold = %d' % ifold)
    # Get the training data in the split
    Itr, Its = ind
    Xtr = X[Itr, :]
    ytr = y[Itr]
    Xts = X[Its, :]
    yts = y[Its]

    # Fit and transform the data
    Xtr1 = xscal.fit_transform(Xtr)
    Xts1 = xscal.transform(Xts)

    # Fit PCA on the training data
    pca = PCA(n_components=npc, svd_solver='randomized', whiten=True)
    pca.fit(Xtr1)

    # Transform the training and test
    Ztr = pca.transform(Xtr1)
    Zts = pca.transform(Xts1)

    # Fiting on the transformed training data
    svc = SVC(C=C, kernel='rbf', gamma=gam)
    svc.fit(Ztr, ytr)

    # Predict on the test data
    yhat = svc.predict(Zts)
    yhatPerFold.append(yhat)
    rsq[ifold] = r2_score(yts, yhat)
    acc[ifold] = accuracy_score(yts, yhat)

    print('R^2     = %12.4e' % rsq[ifold])
    print('acc     = %12.4e' % acc[ifold])

    a = classification_report(yts, yhat, target_names=classNames, output_dict=True)
    reports.append(a)
    print("Confusion matrix on the test data")
    cm = confusion_matrix(yts, yhat, labels=range(n_classes))
    print(cm)
    confusionMatrices.append(cm)

acc_mean = []
# Compute mean accuracy
for i in range(nfold):
    acc_mean.append(reports[i]['accuracy'])

# Save history
accSVM_4_fn = ('REPORTS/accuracySVM_4_time_trimmed.p')
with open(accSVM_4_fn, 'wb') as fp:
    pickle.dump(acc_mean, fp)

reports_4_fn = ('REPORTS/reportsSVM_4_time_trimmed.p')
with open(reports_4_fn, 'wb') as fp:
    pickle.dump(reports, fp)

confusionMatrices_4_fn = ('REPORTS/confusionMatricesSVM_4_time_trimmed.p')
with open(confusionMatrices_4_fn, 'wb') as fp:
    pickle.dump(confusionMatrices, fp)

with plt.style.context("default"):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(1, 10, 10), acc_mean, label='accuracy')
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.title("Accuracy for each fold", fontsize=18)
    plt.legend()
    plt.xlim([0.5, 10.5])
    plt.ylim([0.4, 1.0])
    plt.xticks(np.arange(1, 10.5, step=1))
    plt.grid()
    plt.savefig('FIGURES/accuracy_svm_10fold_4_classes_time_trimmed.pdf')

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatrices)):
    cm = confusionMatrices[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / nfold

# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(9.6, 4.1))  # 5.7
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 18}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNames, rotation=0, fontsize="16", va="center")
ax.set_xticklabels(classNames, rotation=0, fontsize="16", ha="center")
plt.tight_layout()
plt.savefig('FIGURES/normCM_svm_10fold_4_classes_time_trimmed.pdf')

# print performance metrics after 10 fold CV
avgPrecision = np.zeros(numClasses)
avgRecall = np.zeros(numClasses)
avgF1 = np.zeros(numClasses)
classes = list(reports[0].keys())[0:numClasses]
for clIdx in range(len(classes)):
    tmpPrecision = []
    tmpRecall = []
    tmpF1 = []

    for f in range(nfold):
        tmpPrecision.append(reports[f][classes[clIdx]]['precision'])
        tmpRecall.append(reports[f][classes[clIdx]]['recall'])
        tmpF1.append(reports[f][classes[clIdx]]['f1-score'])

    avgPrecision[clIdx] = np.mean(tmpPrecision)
    avgRecall[clIdx] = np.mean(tmpRecall)
    avgF1[clIdx] = np.mean(tmpF1)
print(np.mean(acc_mean))
print(np.mean(avgPrecision))
print(np.mean(avgRecall))
print(np.mean(avgF1))
