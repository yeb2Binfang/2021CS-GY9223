# utility to load the preprocessed EEG data (power spectral densities)
import pickle

import numpy as np


def loadData(path, numClasses):
    # load the data
    X = []
    y = []
    for i in range(numClasses):
        with open(path + 'class' + str(numClasses) + '_' + str(i + 1) + '.pickle', 'rb') as handle:
            tmpX = pickle.load(handle)
            X.append(tmpX)
            y.append(i * np.ones((tmpX.shape[3])))

    X = np.concatenate(X, axis=3)
    y = np.concatenate(y, axis=0)

    return (X, y)
