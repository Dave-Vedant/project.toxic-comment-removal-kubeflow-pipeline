"""Module to create model.
Helper functions to create a multi-layer perceptron model and a separable CNN
model. These functions take the model hyper-parameters as input. This will
allow us to create model instances with slightly varying architectures.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D


# helper function...
def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.
    # Arguments
        num_classes: int, number of classes.
    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

# simple net
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """crates an instance of a multi-layer perceptron model..

    # Arguments:
        layers: int, number of 'Dense layer in the model
        iits: int, output dimension of the layers...
        dropout_rate: float, percentage of input sample drops during the process (to avoid the overfit)
        input_shape: tuple, shape of the input size
        num_classes: int, number of outout classes...

    # return
        An MLP model instance...
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate,input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, acivation='relu')) # middle is relus
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation)) # only on last level
    return model

# 
def universal_cnn_model(blocks, filters, kernel_size, embedding_dim, dropout_rate, pool_size, input_shape, num_classes,
                num_features, use_pretrained_embedding=False, is_embedding_trainable=False, embedding_matrix=None):
    """Creates an instance of the CNN model

    # Arguments
        blocks = int, num of pair of cnn and pooling blocks
        filters= int, output dimension of layers
        kernel_size = int, lenght of convolution windows
        emmbedding_dim = int, dimension of the embedding vectors
        dropout_rate = float, percentage of input to drop at dropout layers
        pool_size = int, actor by which to downscale input at max pooling layers
        input_shape = tuple, shape of the iput to the model
        num_callses = input, number of the output classes
        num_features = int, number of the words ( embedding input dimensions)
        use_pretraind_embedding = bool, for the decision of pretirnaed embedding on/off for model training
        is_embeddign_trainable = boole, true if embedding layer is trainable
        embedding_matrix = dictionary, dictionary whti embeding coefficient

    # Returns
        A CNN model instance with above configuration....

    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = model.Sequencial()

    # add the embedding layer (if pretrained embedding is used, add weight to embedding layers AND set input is_embedding_trainable=true)
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                    output_dim= embedding_dim,
                    input_length=input_shape[0],
                    weights= [embedding_matrix],
                    trainable= is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
        output_dim=embedding_dim,
        input_length=input_shape[0]))

    for  _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    activation= 'relu',
                                    bias_initializer='random_uniform',
                                    depthwise_initializer='random_uniform',
                                    padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    activation='relu',
                                    bias_initializer= 'random_uniform',
                                    depthwise_initializer='random_uniform',
                                    padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model


