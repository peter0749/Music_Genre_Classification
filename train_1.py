# -*- coding: utf-8 -*-
### 參考 keras/example 裡的 neural_style_transfer.py
### 詳細可到這裡觀看他們的原始碼：
### https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
import sys
import os
import os.path
import numpy as np
import math
import time
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from scipy.optimize import fmin_l_bfgs_b
import argparse
import scipy
import scipy.io.wavfile
import conv_net_sound
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
import pandas as pd
import utils


AUDIO_DIR = str(sys.argv[1])

tracks = utils.load('tracks.csv')
features = utils.load('features.csv')
echonest = utils.load('echonest.csv')

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

subset = tracks.index[tracks['set', 'subset'] <= 'medium']

assert subset.isin(tracks.index).all()
assert subset.isin(features.index).all()

features_all = features.join(echonest, how='inner').sort_index(axis=1)
print('Not enough Echonest features: {}'.format(features_all.shape))

tracks = tracks.loc[subset]
features_all = features.loc[subset]

train = np.array(tracks.index[tracks['set', 'split'] == 'training'])
val = np.array(tracks.index[tracks['set', 'split'] == 'validation'])
test = np.array(tracks.index[tracks['set', 'split'] == 'test'])

print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
#genres = list(tracks['track', 'genre_top'].unique())
print('Top genres ({}): {}'.format(len(genres), genres))
genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
print('All genres ({}): {}'.format(len(genres), genres))

labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)

lr = 0.0001
batch_size = 2
rate = 11025
iteration = int(sys.argv[2])

loader = utils.LibrosaLoader(sampling_rate=rate)
SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader)

keras.backend.clear_session()

model = conv_net_sound.conv_net(input_shape = loader.shape,
                          class_n = int(labels_onehot.shape[1])
                         )
if (os.path.isfile('./top_weight.h5')):
    model.load_weights('./top_weight.h5')
model.summary()
optimizer = SGD(lr=lr, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
checkPoint = ModelCheckpoint(filepath="./top_weight.h5", verbose=1, save_best_only=True, monitor='loss', mode='min', save_weights_only=True, period=50)

model.fit_generator(SampleLoader(train, batch_size=batch_size), steps_per_epoch=int(math.ceil(train.size/batch_size)), epochs=iteration, callbacks=[checkPoint])
model.evaluate_generator(SampleLoader(val, batch_size=batch_size), steps_per_epoch=int(math.ceil(val.size/batch_size)))
model.evaluate_generator(SampleLoader(test, batch_size=batch_size), steps_per_epoch=int(math.ceil(test.size/batch_size)))

model.save_weights('./conv_net.h5')
