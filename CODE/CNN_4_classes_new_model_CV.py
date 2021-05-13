# 10-fold cross-validation with the new model on the dataset with 4 classes
# PVHA, PVLA, NVHA, NVLA,
from __future__ import print_function

import matplotlib

matplotlib.rcParams.update({'font.size': 16})

import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

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
X = np.mean(X, axis=2)

# we need to add additional dimension to X to make it resembling pictures dataset
X = np.expand_dims(X, 3)
print(X.shape)
# split on testing and training datasets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.20, random_state=2)  # 42

print('Xtr shape:  ' + str(Xtr.shape))
print('Xts shape:  ' + str(Xts.shape))

Xtr = Xtr.astype('float32')
Xts = Xts.astype('float32')


# three layer configuration avg accuracy 0.62
def create_mod(use_dropout=False, use_bn=False):
    num_classes = numClasses
    model = Sequential()
    model.add(Conv2D(64, (3, 3),
                     padding='valid', activation='relu',
                     input_shape=Xtr.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    if use_bn:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    if use_bn:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='preds'))
    return model


def create_datagen():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        horizontal_flip=False,  # randomly flip images # True
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    return datagen


# Parameters
nepochs = 100
batch_size = 32
lr = 1e-3
decay = 1e-4
nfold = 10

## 10 fold cross validation with optimal parameters npcs = 25 gamma = 0.1
# Create cross-validation object

# plot train/test distribution
with plt.style.context("default"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=False, sharey=False)
    n, bins, patches = axs[0].hist(ytr, numClasses, rwidth=0.5, alpha=0.8)
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    axs[0].set_xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w))
    axs[0].set_xlim(bins[0], bins[-1])
    axs[0].set_ylim(0, 601)
    axs[0].set_xticklabels(classNames, fontsize=8, rotation=0)
    axs[0].set_ylabel('Number of train instances')

    n, bins, patches = axs[1].hist(yts, numClasses, rwidth=0.5, alpha=0.8)
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    axs[1].set_xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w))
    axs[1].set_xlim(bins[0], bins[-1])
    axs[1].set_ylim(0, 201)
    axs[1].set_xticklabels(classNames, fontsize=8, rotation=0)
    axs[1].set_ylabel('Number of test instances')
    plt.tight_layout()
    plt.savefig('FIGURES/trainTestDist_cnn_10fold_4_classes_new.pdf')

kf = KFold(nfold, shuffle=True)

# Run the cross-validation
rsq = np.zeros(nfold)
acc = np.zeros(nfold)
yhatPerFold = []
ytsPerFold = []
accuracyTrain = []
accuracyVal = []
confusionMatrices = []
reports = []

xscal = StandardScaler()

for ifold, ind in enumerate(kf.split(X)):
    print('Fold = %d' % ifold)
    # Get the training data in the split
    Itr, Its = ind
    Xtr = X[Itr, :]
    ytr = y[Itr]
    Xts = X[Its, :]
    yts = y[Its]

    # clear session for each fold, other wise it will be utilizing model trained from previous fold
    # and it will result in the validation accuracy values close to 1
    K.clear_session()
    model = create_mod()
    # Create the optimizer
    opt = optimizers.RMSprop(lr=lr, decay=decay)

    # Compile
    hist = model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=opt,
                         metrics=['accuracy'])
    print(model.summary())

    # Fit and transform the data
    Xtr = xscal.fit_transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xts = xscal.fit_transform(Xts.reshape(-1, Xts.shape[-1])).reshape(Xts.shape)

    # Fit the model with no data augmentation
    hist = model.fit(Xtr, ytr, batch_size=batch_size,
                     epochs=nepochs, validation_data=(Xts, yts),
                     shuffle=True)
    hist_dict = hist.history
    testAcc = hist_dict['accuracy']
    valAcc = hist_dict['val_accuracy']
    accuracyTrain.append(testAcc)
    accuracyVal.append(valAcc)

    yhat = model.predict(Xts)
    labBin = LabelBinarizer()
    labBin.fit(yts)
    yhat1 = labBin.inverse_transform(np.round(yhat))

    yhat = yhat1

    yhatPerFold.append(yhat)
    ytsPerFold.append(yts)
    rsq[ifold] = r2_score(yts, yhat)
    acc[ifold] = accuracy_score(yts, yhat)

    print('R^2     = %12.4e' % rsq[ifold])
    print('acc     = %12.4e' % acc[ifold])

    a = classification_report(yts, yhat, target_names=classNames, output_dict=True)
    reports.append(a)
    print("Confusion matrix on the test data")
    cm = confusion_matrix(yts, yhat, labels=range(numClasses))
    print(cm)
    confusionMatrices.append(cm)

# Save accuracy per fold per epoch
accTrain_fn = ('REPORTS/accuracyTrain_4_new.p')
with open(accTrain_fn, 'wb') as fp:
    pickle.dump(accuracyTrain, fp)
accVal_fn = ('REPORTS/accuracyVal_4_new.p')
with open(accVal_fn, 'wb') as fp:
    pickle.dump(accuracyVal, fp)

acc_mean = []
# Compute mean accuracy
for i in range(nfold):
    acc_mean.append(reports[i]['accuracy'])

# Save history
accTotal_4_fn = ('REPORTS/accuracyTotal_4_new.p')
with open(accTotal_4_fn, 'wb') as fp:
    pickle.dump(acc_mean, fp)

reports_4_fn = ('REPORTS/reportsCNN_4_new.p')
with open(reports_4_fn, 'wb') as fp:
    pickle.dump(reports, fp)

confusionMatrices_4_fn = ('REPORTS/confusionMatricesCNN_4_new.p')
with open(confusionMatrices_4_fn, 'wb') as fp:
    pickle.dump(confusionMatrices, fp)

# Compute mean accuracy
acc_mean_train = np.mean(accuracyTrain, axis=1)
acc_mean_val = np.mean(accuracyVal, axis=1)

with plt.style.context("default"):
    plt.figure(figsize=(10, 8))
    plt.plot(np.linspace(1, 10, 10), acc_mean_train, label='training accuracy')
    plt.plot(np.linspace(1, 10, 10), acc_mean_val, label='validation accuracy')
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.title("Accuracy for each fold")
    plt.legend()
    plt.xlim([0.5, 10.5])
    plt.ylim([0.4, 1.0])
    plt.xticks(np.arange(1, 10.5, step=1))
    plt.grid
    plt.savefig('FIGURES/accTrainVal_10foldCV_4_classes_new.pdf')

# normalize confusion matrices
normalizedAvgCM = np.zeros((numClasses, numClasses))
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
plt.savefig('FIGURES/normCM_cnn_10fold_4_classes_new.pdf')

with plt.style.context("default"):
    plt.figure(figsize=(10, 5))
    for iplt in range(2):

        plt.subplot(1, 2, iplt + 1)

        if iplt == 0:
            acc = np.mean(accuracyTrain, axis=0)
        else:
            acc = np.mean(accuracyVal, axis=0)
        plt.plot(acc, '-', linewidth=3)

        n = len(acc)
        nepochs = len(acc)
        plt.grid()
        plt.xlim([0, nepochs])

        plt.xlabel('Epoch')
        if iplt == 0:
            plt.ylabel('Train accuracy')
        else:
            plt.ylabel('Test accuracy')

    plt.tight_layout()
    plt.savefig('FIGURES/accuracyTrainVal_cnn_10fold_4_classes_new.pdf')

# save model
h5_fn = ('MODELS/CNN_4_classes_new_model.h5')
model.save(h5_fn)
print('Model saved as %s' % h5_fn)

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
