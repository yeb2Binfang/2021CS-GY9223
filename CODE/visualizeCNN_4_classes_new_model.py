# Visualization of hidden layers for new CNN model on the dataset with 4 classes
# PVHA, PVLA, NVHA, NVLA,
from __future__ import print_function

import matplotlib
from tensorflow.keras.models import Model

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

import numpy as np
from loadData import loadData

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
# Importing the data files

import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

with open('Xtr.pickle', 'wb') as handle:
    pickle.dump(Xtr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Xts.pickle', 'wb') as handle:
    pickle.dump(Xts, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('ytr.pickle', 'wb') as handle:
    pickle.dump(ytr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('yts.pickle', 'wb') as handle:
    pickle.dump(yts, handle, protocol=pickle.HIGHEST_PROTOCOL)


# three layer configuration avg accuracy 0.62
def create_mod(use_dropout=False, use_bn=False):
    num_classes = 4
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

# clear session 
K.clear_session()
model = create_mod()
# Create the optimizer
opt = optimizers.RMSprop(lr=lr, decay=decay)

# Compile
hist = model.compile(loss='sparse_categorical_crossentropy',
                     optimizer=opt,
                     metrics=['accuracy'])
print(model.summary())
xscal = StandardScaler()
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

yhat = model.predict(Xts)
labBin = LabelBinarizer()
labBin.fit(yts)
yhat1 = labBin.inverse_transform(np.round(yhat))

yhat = yhat1

rsq = r2_score(yts, yhat)
acc = accuracy_score(yts, yhat)

print('R^2     = %4.4e' % rsq)
print('acc     = %4.4e' % acc)

reports = classification_report(yts, yhat, target_names=classNames, output_dict=True)

print("Confusion matrix on the test data")
cm = confusion_matrix(yts, yhat, labels=range(numClasses))
print(cm)

# Save model
h5_fn = ('MODELS/CNN_4_classes_new.h5')
model.save(h5_fn)
print('Model saved as %s' % h5_fn)

#########################################################################################################################
#load model
from tensorflow import keras
h5_fn = ('MODELS/CNN_4_classes_new.h5')
model = keras.models.load_model(h5_fn)
print(model.summary())

########################################################################################################################
# visualize the dense layer with activation maximization
########################################################################################################################
# #https://github.com/raghakot/keras-vis/blob/master/examples/mnist/activation_maximization.ipynb
from tf_keras_vis.activation_maximization import ActivationMaximization
import tensorflow as tf


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear


activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)

from tf_keras_vis.utils.callbacks import Print

for filter_number in range(numClasses):
    def loss(output):
        return output[..., filter_number]

    activation = activation_maximization(loss, callbacks=[Print(interval=50)])
    image = np.squeeze(activation[0].astype(np.uint8))

    subplot_args = {'nrows': 1, 'ncols': 1, 'figsize': (5, 5)}
    f, ax = plt.subplots(**subplot_args)
    im = ax.imshow(image)
    ax.invert_yaxis()
    ax.set_xlabel('channels', fontsize=12)
    ax.set_ylabel('frequencies, Hz', fontsize=12)
    ax.set_xlim([0, 58])
    ax.set_ylim([0, 79])
    xticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_title('class ' + classNames[filter_number], fontsize=14)
    plt.tight_layout()
    f.colorbar(im)
    plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/dense_layer_" + classNames[filter_number] + "_new.pdf")

########################################################################################################################
# visualize the conv layer with activation maximization
########################################################################################################################
model.summary()
# first convolutional layer
layer_name = 'conv2d'  # The first layer


def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)

subplot_args = {'nrows': 4, 'ncols': 16, 'figsize': (12, 4)}
f, ax = plt.subplots(**subplot_args)
for filter_number in range(64):
    def loss(output):
        return output[..., filter_number]


    activation = activation_maximization(loss, callbacks=[Print(interval=50)])
    image = np.squeeze(activation[0].astype(np.uint8))
    # specify subplot and turn of axis
    ax = plt.subplot(4, 16, filter_number + 1)
    im = ax.imshow(image, cmap='plasma')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig = plt.gcf()
# put colorbar at desire position
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cbar_ax)
im.set_clim(vmin=0, vmax=3.5)
plt.suptitle('64 feature maps of the first Conv2D layer', fontsize=12)
plt.show()
# plt.tight_layout()
fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/conv_layer_1_" + str(numClasses) + "new.pdf")

########################################################################################################################
# first maxpooling layer
layer_name = 'max_pooling2d'  # The first maxpooling layer


def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)

subplot_args = {'nrows': 4, 'ncols': 16, 'figsize': (12, 4)}
f, ax = plt.subplots(**subplot_args)
for filter_number in range(64):
    def loss(output):
        return output[..., filter_number]


    activation = activation_maximization(loss, callbacks=[Print(interval=50)])
    image = np.squeeze(activation[0].astype(np.uint8))
    # specify subplot and turn of axis
    ax = plt.subplot(4, 16, filter_number + 1)
    im = ax.imshow(image, cmap='plasma')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig = plt.gcf()
# put colorbar at desire position
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cbar_ax)
im.set_clim(vmin=0, vmax=3.5)
plt.suptitle('64 feature maps of the first max pooling layer', fontsize=12)
plt.show()
fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
# plt.tight_layout()
plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/maxPool_layer_1_" + str(numClasses) + "_new.pdf")

########################################################################################################################

# first convolutional layer
layer_name = 'conv2d_1'  # The first convolutional layer


def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)

subplot_args = {'nrows': 4, 'ncols': 16, 'figsize': (12, 4)}
f, ax = plt.subplots(**subplot_args)
for filter_number in range(64):
    def loss(output):
        return output[..., filter_number]


    activation = activation_maximization(loss, callbacks=[Print(interval=50)])
    image = np.squeeze(activation[0].astype(np.uint8))
    # specify subplot and turn of axis
    ax = plt.subplot(4, 16, filter_number + 1)
    im = ax.imshow(image, cmap='plasma')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig = plt.gcf()
# put colorbar at desire position
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cbar_ax)
im.set_clim(vmin=0, vmax=3.5)
plt.suptitle('64 feature maps of the second convolutional layer', fontsize=12)
fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
plt.show()
# plt.tight_layout()
plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/conv_layer_2_" + str(numClasses) + "_new.pdf")
########################################################################################################################

########################################################################################################################
# explore features
########################################################################################################################
# https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
extractor = Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers],
)
features = extractor(Xts)
list(map(lambda weights: weights.shape, features))

########################################################################################################################
# visualize spectrograms for all classes:
########################################################################################################################
for currentClass_id in range(4):
    X_class1 = Xts[yts == currentClass_id, :, :, :]
    y_class1 = yts[yts == currentClass_id]

    # let's average over trials: (over the first dimension):
    X_class1_trial_avg = np.mean(X_class1, axis=0)

    # now our data for the class 1 has shape (80,160,1)

    # let's plot spectrograms:

    fig, axs = plt.subplots(figsize=(10, 7.5))
    im = axs.imshow(np.squeeze(X_class1_trial_avg), cmap='plasma', aspect=0.5)
    axs.invert_yaxis()
    axs.set_ylim([0, 80])
    axs.set_xlim([0, 59])
    axs.set_xticks(np.arange(0, 59, 5))
    axs.set_title(classNames[currentClass_id])
    axs.set_xlabel('channels')
    axs.set_ylabel('frequencies, Hz')
    cbar_ax = fig.add_axes([0.91, 0.12, 0.025, 0.75])
    fig.colorbar(im, cax=cbar_ax)
    # im.set_clim(vmin=5, vmax=40)
    plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/spectrograms_" + classNames[currentClass_id] + ".pdf")

########################################################################################################################
#
# plot feature maps of the first convolutional layer for all classes
#
########################################################################################################################
model.layers
model1 = Model(inputs=model.inputs, outputs=model.layers[0].output)
model1.summary()

# get feature map for first hidden layer
feature_maps = model1.predict(Xts)
feature_maps.shape  # 32 feature maps
########################################################################################################################
# plot feature maps of the first convolutional layer for each class
########################################################################################################################
for currentClass in [0, 1, 4, 11]:
    if yhat[currentClass] == yts[currentClass]:
        print("currentClass = " + classNames[int(yts[currentClass])])
        feature_maps_current_class = feature_maps[currentClass]
        feature_maps_trial_avg = feature_maps_current_class
        ix = 1
        fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(12, 4))
        for _ in range(4):
            for _ in range(16):
                # specify subplot and turn of axis
                ax = plt.subplot(4, 16, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                im = plt.imshow(feature_maps_trial_avg[:, :, ix - 1], cmap='plasma')
                ax.invert_yaxis()
                ix += 1
        # show the figure
        fig = plt.gcf()
        # put colorbar at desire position
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
        fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
        im.set_clim(vmin=0, vmax=3.5)
        plt.show()
    plt.suptitle("64 feature maps of the first Conv2D layer " + classNames[int(yts[currentClass])], fontsize=15)
    plt.savefig(
        "FIGURES/VISUALIZING_FEATURE_MAPS_4/featureMapsConv2D_1_new" + classNames[int(yts[currentClass])] + ".pdf")

########################################################################################################################
# plot feature maps of the first convolutional layer for all classes
########################################################################################################################
feature_maps_trial_avg = np.mean(feature_maps, axis=0)
ix = 1
fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(12, 4))
for _ in range(4):
    for _ in range(16):
        # specify subplot and turn of axis
        ax = plt.subplot(4, 16, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        im = plt.imshow(feature_maps_trial_avg[:, :, ix - 1], cmap='plasma')
        ax.invert_yaxis()
        ix += 1
# show the figure
fig = plt.gcf()
# put colorbar at desire position
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cbar_ax)
im.set_clim(vmin=0, vmax=3.5)
plt.show()
fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
plt.suptitle('64 feature maps of the first Conv2D layer', fontsize=15)
plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/featureMapsConv2D_1_4_new.pdf")
########################################################################################################################
# redefine model to output right after the second hidden layer
########################################################################################################################
model.layers
model1 = Model(inputs=model.inputs, outputs=model.layers[1].output)
model1.summary()

# get feature map for first hidden layer
feature_maps = model1.predict(Xts)
feature_maps.shape  # 32 feature maps
########################################################################################################################
# plot feature maps of the first maxpooling  layer for each class
########################################################################################################################
for currentClass in [0, 1, 4, 11]:
    if yhat[currentClass] == yts[currentClass]:
        print("currentClass = " + classNames[int(yts[currentClass])])
        feature_maps_current_class = feature_maps[currentClass]
        feature_maps_trial_avg = feature_maps_current_class
        ix = 1
        fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(12, 4))
        for _ in range(4):
            for _ in range(16):
                # specify subplot and turn of axis
                ax = plt.subplot(4, 16, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                im = plt.imshow(feature_maps_trial_avg[:, :, ix - 1], cmap='plasma')
                ax.invert_yaxis()
                ix += 1
        # show the figure
        fig = plt.gcf()
        # put colorbar at desire position
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
        fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
        im.set_clim(vmin=0, vmax=3.5)
        plt.show()
        plt.suptitle("64 feature maps of the first MaxPooling2D layer for " + classNames[int(yts[currentClass])],
                     fontsize=15)
        plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/featureMapsMaxPooling2D_1_new_" + classNames[
            int(yts[currentClass])] + ".pdf")
    else:
        print("misclassified")
########################################################################################################################
# plot feature maps of the first maxpooling layer for all classes
########################################################################################################################

feature_maps_current_class = feature_maps
feature_maps_trial_avg = np.mean(feature_maps_current_class, axis=0)
ix = 1
fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(12, 4))
for _ in range(4):
    for _ in range(16):
        # specify subplot and turn of axis
        ax = plt.subplot(4, 16, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        im = plt.imshow(feature_maps_trial_avg[:, :, ix - 1], cmap='plasma')
        ax.invert_yaxis()
        ix += 1
# show the figure
fig = plt.gcf()
# put colorbar at desire position
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
im.set_clim(vmin=0, vmax=3.5)
plt.show()
plt.suptitle('64 feature maps of the first MaxPooling2D layer', fontsize=15)
plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/featureMapsMaxPooling2D_1_new.pdf")

########################################################################################################################
# plot feature maps of the second convolutional layer for all classes
########################################################################################################################
model.layers
model1 = Model(inputs=model.inputs, outputs=model.layers[2].output)
model1.summary()

# get feature map for first hidden layer
feature_maps = model1.predict(Xtr)
feature_maps.shape  # 64 feature maps
########################################################################################################################
# plot feature maps of the second convolutional layer for each class
########################################################################################################################
for currentClass in [0, 1, 4, 11]:
    if yhat[currentClass] == yts[currentClass]:
        print("currentClass = " + classNames[int(yts[currentClass])])
        feature_maps_current_class = feature_maps[currentClass]
        feature_maps_trial_avg = feature_maps_current_class
        ix = 1
        fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(12, 4))
        for _ in range(4):
            for _ in range(16):
                # specify subplot and turn of axis
                ax = plt.subplot(4, 16, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                im = plt.imshow(feature_maps_trial_avg[:, :, ix - 1], cmap='plasma')
                ax.invert_yaxis()
                ix += 1
        # show the figure
        fig = plt.gcf()
        # put colorbar at desire position
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
        fig.colorbar(im, cax=cbar_ax)
        im.set_clim(vmin=0, vmax=3.5)
        fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
        plt.show()
        plt.suptitle("64 feature maps of the second Conv2D layer for " + classNames[int(yts[currentClass])],
                     fontsize=12)
        plt.savefig(
            "FIGURES/VISUALIZING_FEATURE_MAPS_4/featureMapsConv2D_2_new_" + classNames[int(yts[currentClass])] + ".pdf")
    else:
        print("misclassified")

########################################################################################################################
# plot feature maps of the second convolutional layer for all classes
########################################################################################################################

feature_maps_trial_avg = np.mean(feature_maps, axis=0)
ix = 1
fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(12, 4))
for _ in range(4):
    for _ in range(16):
        # specify subplot and turn of axis
        ax = plt.subplot(4, 16, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        im = plt.imshow(feature_maps_trial_avg[:, :, ix - 1], cmap='plasma')
        ax.invert_yaxis()
        ix += 1
# show the figure
fig = plt.gcf()
# put colorbar at desire position
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cbar_ax)
im.set_clim(vmin=0, vmax=3.5)
fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  #
plt.show()
plt.suptitle('64 feature maps of the second Conv2D layer', fontsize=12)
plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/featureMapsConv2D_2_4_new.pdf")

########################################################################################################################
# visualize saliency maps
########################################################################################################################
# The `output` variable refer to the output of the model,
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

for filter_number in range(4):
    def loss(output):
        return output[..., filter_number]


    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m


    # Create Saliency object.
    # If `clone` is True(default), the `model` will be cloned,
    # so the `model` instance will be NOT modified, but it takes a machine resources.
    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)

    # Generate saliency map
    saliency_map = saliency(loss, Xts)
    saliency_map = normalize(saliency_map)

    idx0 = [i for i in range(yts.shape[0]) if yts[i] == filter_number]
    i = filter_number
    # Render
    subplot_args = {'nrows': 1, 'ncols': 1, 'figsize': (5, 5)}
    f, ax = plt.subplots(**subplot_args)
    im = ax.imshow(np.mean(saliency_map[idx0, :, :], axis=0), cmap='jet')
    ax.invert_yaxis()
    ax.set_xlabel('channels', fontsize=12)
    ax.set_ylabel('frequencies, Hz', fontsize=12)
    ax.set_xlim([0, 58])
    ax.set_ylim([0, 79])
    xticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_title('class ' + classNames[i], fontsize=14)
    plt.tight_layout()
    f.colorbar(im)
    plt.tight_layout()
    im.set_clim(vmin=0, vmax=0.52)
    plt.show()
    plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/saliencyMap" + classNames[i] + "_new_1.pdf")

########################################################################################################################
# visualize saliency maps SmoothGrad
########################################################################################################################
# The `output` variable refer to the output of the model,
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

for filter_number in range(4):
    def loss(output):
        return output[..., filter_number]


    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.softmax
        return m


    # Create Saliency object.
    # If `clone` is True(default), the `model` will be cloned,
    # so the `model` instance will be NOT modified, but it takes a machine resources.
    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(loss,
                            Xts,
                            smooth_samples=50,  # The number of calculating gradients iterations.
                            smooth_noise=0.20)  # noise spread level.
    saliency_map = normalize(saliency_map)

    idx0 = [i for i in range(yts.shape[0]) if yts[i] == filter_number]
    i = filter_number
    # Render
    subplot_args = {'nrows': 1, 'ncols': 1, 'figsize': (5, 5)}
    f, ax = plt.subplots(**subplot_args)
    im = ax.imshow(np.mean(saliency_map[idx0, :, :], axis=0), cmap='jet')
    ax.invert_yaxis()
    ax.set_xlabel('channels', fontsize=12)
    ax.set_ylabel('frequencies, Hz', fontsize=12)
    ax.set_xlim([0, 58])
    ax.set_ylim([0, 79])
    xticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_title('class ' + classNames[i], fontsize=14)
    plt.tight_layout()
    f.colorbar(im)
    plt.tight_layout()
    plt.show()
    plt.savefig("FIGURES/VISUALIZING_FEATURE_MAPS_4/saliencyMapSmoothGrad" + classNames[i] + "_new_.pdf")
