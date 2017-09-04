# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers

def custom_STFT_layer(x, FFT_n=2048, FFT_t=256, img_nrows=505, img_ncols=768):
    ## input_shape: (batch_size, timestep)
    ## output_shape:(batch_size, sample, freq_range, channel)
    scale = 1.0/32768.0 ## pcm_s16le -> pcm_f32le
    y = tf.cast(x, tf.float32)
    y = tf.scalar_mul(scale, y)
    stft = tf.contrib.signal.stft(y, FFT_n, FFT_t)
    stft = tf.expand_dims(stft, -1)
    dense = tf.abs(stft[:, :img_nrows, :img_ncols, :])
    spec  = tf.log1p(dense)
    return spec
    ##  end of spectrogram

def conv_net(input_tensor = None,
               input_shape = None,
               class_n = None,
               weight_path = None
               ):
    if input_tensor is None:
        input_layer = Input(input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            input_layer = Input(tensor=input_tensor, shape=input_tensor.shape)
        else:
            input_layer = input_tensor

    stfted = Lambda(custom_STFT_layer, name='STFT')(input_layer)
    ## VGG19 Net:
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='random_normal')(stfted)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='random_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='random_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='random_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='random_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='random_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='random_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_initializer='random_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool2')(x)

    if class_n is not None:
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='fc1')(x) ## reduced net for memory
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='fc2')(x) ## reduced net for memory
        x = Dropout(0.5)(x)
        x = Dense(class_n, activation='softmax')(x)
    model = Model(input_layer, x) ## model: wav -> features
    if weight_path is not None:
        model.load_weights(str(weight_path))
    return model


