import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                          MaxPooling2D, Concatenate, UpSampling2D,
                          Activation, BatchNormalization)
from keras import optimizers as opt
from itertools import product
from aux_metrics import *
from keras.utils import multi_gpu_model

# Function to get model learning rate
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# Function which creates 2D unet inspired network model
def create_unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5, 5),
                        deconvolution_kernel_size=(5, 5),
                        pool_size=(2, 2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001, class_weights = {}, gpu_num = 2):


    # Turns number of layers into iterable list
    layers = np.arange(layers)

    # Sets number of classification labels (number of distinct tissues to be considered)
    number_of_classification_labels = n_labels

    # Defines input layer to model
    inputs = Input(shape=input_image_size)


    # defines enconding section of the network, assigns variables amount of convolutional layers depending
    encoding_convolution_layers = []
    pool = None

    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2 ** (layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                           padding='same')(inputs)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)



        elif i == 1:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          padding='same')(pool)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          padding='same')(pool)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)


        else:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          padding='same')(pool)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          padding='same')(conv)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)



        conv_buff = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                           padding='same')(conv)
        conv_buff = BatchNormalization()(conv_buff)
        conv_buff = Activation('relu')(conv_buff)


        encoding_convolution_layers.append(conv_buff)


        if i < len(layers) - 1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers) - 1]
    for i in range(1, len(layers)):
        number_of_filters = lowest_resolution * 2 ** (len(layers) - layers[i] - 1)
        
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)

        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers) - i - 1]])


        if i == 1 or i == 2:
            outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                          padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

            outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

            outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

        else:
            outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                          padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

            outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                             padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)


    # network is used for classification which requires sigmoid activation for last layer if only tissue is classified against background
    # last layer is set to softmax if there is more than one tissue to be classified
    if mode == 'classification':


        if number_of_classification_labels == 1:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                             activation='sigmoid')(outputs)
        else:
            
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                             activation='softmax')(outputs)

        # Joins previously defined layers of model with newly defined output layer
        unet_model = Model(inputs=inputs, outputs=outputs)


    return unet_model








# Alternative Loss Functions to Play With

def jaccard_distance(y_true, y_pred, smooth=100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer

class WeightedCategoricalCrossEntropy(object):
    def __init__(self, weights):
        weights_len = len(weights)
        self.weights = np.ones((weights_len, weights_len))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'w_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.w_categorical_crossentropy(y_true, y_pred)

    def w_categorical_crossentropy(self, y_true, y_pred):
        weights_len = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for cost_pred, cost_targ in product(range(weights_len),range( weights_len)):
            w = K.cast(self.weights[cost_pred, cost_targ], K.floatx())
            y_predict = K.cast(y_pred_max_mat[...,cost_pred], K.floatx())
            y_target = K.cast(y_pred_max_mat[...,cost_targ], K.floatx())
            final_mask += w * y_predict * y_target
        return K.categorical_crossentropy(y_true,y_pred) * final_mask

def focal_loss(y_true, y_pred, gamma=2):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    return -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred),
                  axis=-1)

def focal_loss_alt(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed