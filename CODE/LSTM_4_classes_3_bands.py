# 10-fold cross-validation LSTM 2 layers on the dataset with 4 classes
# PVHA, PVLA, NVHA, NVLA,from __future__ import print_function
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from sklearn.utils import shuffle
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

# load data
(X, y) = loadData(path, numClasses)

# the idea is for every time point extract beta, lower gamma and higher gamma bands and averaging each separately,
# concatenating them along the channels into vector of length 177 (59X3)

# thus from a vector of size (2720, 80, 160, 59) we switch to a vector of size (2720, 177, 160)
X1 = np.transpose(X, [3, 0, 1, 2])
X1.shape

X4 = np.zeros((X1.shape[0], 177, X1.shape[2]))
for i in range(160):
    X2 = X1[:, :, i, :]

    # Extracting beta, lower gamma and higher gamma bands and averaging each separately
    temp1 = np.mean(X2[:, 13:30, :], axis=1)
    temp2 = np.mean(X2[:, 31:50, :], axis=1)
    temp3 = np.mean(X2[:, 51:79, :], axis=1)

    X3 = np.concatenate((temp1, temp2, temp2), axis=1)

    X4[:, :, i] = X3

X4.shape

# we need to split the original data into training validation and testing sets
# for more details see: https://stackoverflow.com/questions/64004193/how-to-split-dataset-to-train-test-and-valid-in-python

train_size = 0.8
validate_size = 0.1
idxArray = np.linspace(0, X4.shape[0] - 1, X4.shape[0]).astype('int')
idxArrayNew = shuffle(idxArray, random_state=2)
idx_train, idx_val, idx_test = np.split(idxArrayNew, [int(train_size * idxArrayNew.shape[0]),
                                                      int((validate_size + train_size) * idxArrayNew.shape[0])])

X_train = X4[idx_train, :, :]
X_val = X4[idx_val, :, :]
X_test = X4[idx_test, :, :]
y_train = y[idx_train]
y_val = y[idx_val]
y_test = y[idx_test]

# check the sizes
print('Train set shape: ' + str(X_train.shape))
print('Validation set shape: ' + str(X_val.shape))
print('Test set shape: ' + str(X_test.shape))

# it is important to scale the data
xscal = StandardScaler()
# Fit and transform the data
Xtr_scaled = xscal.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
Xval_scaled = xscal.fit_transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
Xts_scaled = xscal.fit_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
print(X_train.shape)
print(Xtr_scaled.shape)
print(X_val.shape)
print(Xval_scaled.shape)
print(X_test.shape)
print(Xts_scaled.shape)

# Initializing the classifier Network
classifier = Sequential()

# Adding the input LSTM network layer
classifier.add(LSTM(256, input_shape=(Xtr_scaled.shape[1:]), return_sequences=True))
classifier.add(Dropout(0.2))

# Adding a second LSTM network layer
classifier.add(LSTM(256))

# Adding a dense hidden layer
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(numClasses, activation='softmax'))

# Compiling the network
classifier.compile(loss='sparse_categorical_crossentropy',
                   optimizer=Adam(lr=0.001, decay=1e-6),
                   metrics=['accuracy'])

# Fitting the data to the model
classifier.fit(Xtr_scaled,
               y_train,
               epochs=100,
               validation_data=(Xval_scaled, y_val))

# check accuracy on the test
test_loss, test_acc = classifier.evaluate(Xts_scaled, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# with 2 LSTM layers:
# after 100 epochs:
# 2176/2176 [==============================] - 123s 57ms/step - loss: 0.1751 - acc: 0.9426 - val_loss: 3.4544 - val_acc: 0.4338
# Test Loss: 3.4543614387512207
# Test Accuracy: 0.4338235294117647
