#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import cv2
import imutils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_imgs(path,dim, path_gt='',struc='TE'):
    files=[path/f for f in os.listdir(path)]
    x = np.array([cv2.resize(cv2.imread(str(f),cv2.IMREAD_GRAYSCALE), dim) for f in files])
    if path_gt == '':
        y=None
    else:
        y = np.array([cv2.threshold(cv2.resize(cv2.imread(str(path_gt/(f.stem+' '+struc+'_Mask.bmp')),cv2.IMREAD_GRAYSCALE), dim),127,255,cv2.THRESH_BINARY)[1] for f in files])
    return x,y

def std_norm(x):
    mean, std= np.mean(x, axis=0), np.std(x, axis=0) 
    return ((x.astype('float32') - mean) / std , mean, std)

def std_norm_test(x,mean,std):
    return (x.astype('float32') - mean) / std

def data_augmentation(x_train, y_train):
    x_train_augmented, y_train_augmented=[], []
    for k in range(len(x_train)):
        img=x_train[k]
        mask=y_train[k]
        for i in range(37):
            x_train_augmented.append(imutils.rotate(img, angle=10*i))
            y_train_augmented.append(imutils.rotate(mask, angle=10*i))
    return np.array(x_train_augmented), np.array(y_train_augmented)


#MODEL UNET HARUN
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(input)
    x = Dropout(0.4) (x)
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(x)
    x = Dropout(0.4) (x)
    return x

#Encoder block: Conv block followed by maxpooling and residual block

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters) # convolutional block
#     x = BatchNormalization()(x)
    input = Conv2D(num_filters, 1, padding="same")(input)
    transfer = tf.math.add(x,input)
    p = MaxPool2D(2)(transfer) #subsampling block  
    p = Activation("relu")(p)
    return x, p  


#Bottleneck block: 4 dilaten convolution layers

def bottleneck_block(input, num_filters):
    x1 = Conv2D(num_filters, 3, dilation_rate=1,  padding="same" )(input)
    x1 =  Dropout(0.4) (x1)
    x2 = Conv2D(num_filters, 3, dilation_rate=2, padding="same" )(x1)
    x2 =  Dropout(0.4) (x2)
    x3 = Conv2D(num_filters, 3, dilation_rate=4, padding="same" )(x2)
    x3 =  Dropout(0.4) (x3)
    x4 = Conv2D(num_filters, 3, dilation_rate=8, padding="same" )(x3)
    x4 =  Dropout(0.4) (x4)
    c = Concatenate(axis=3)([x1, x2, x3, x4])
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(c)
    x = Dropout(0.05) (x)
    return  x


#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = UpSampling2D(size=2)(input)
    x = BatchNormalization()(x) 
    x1 = Concatenate(axis=3)([x, skip_features])
      
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(x1)
    x = Dropout(0.4) (x)
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(x)
    x = Dropout(0.4) (x)
    x1 = Conv2D(num_filters, 1, padding="same")(x1)
    return tf.math.add(x,x1)

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 16)
    s2, p2 = encoder_block(p1, 32)
    s3, p3 = encoder_block(p2, 64)
    s4, p4 = encoder_block(p3, 128)

    b1   = bottleneck_block(p4, 128) #Bridge

    d1 = decoder_block(b1, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 16)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  
    model = Model(inputs, outputs, name="U-Net")
    return model


#METRICS

def jaccard_index(y_true, y_pred):
    """ Calculates mean of Jaccard index as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection) / (sum_ - intersection)
    return tf.reduce_mean(jac)

def my_loss_fn(y_true, y_pred):
    return BinaryCrossentropy(from_logits=True)(y_true, y_pred)-tf.math.log(jaccard_index(y_true, y_pred))

from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y=shuffle(batch_x, batch_y, random_state=1)
        return batch_x, batch_y

