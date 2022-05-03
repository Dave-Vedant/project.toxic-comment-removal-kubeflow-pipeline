# shallow NN
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.keras.metrics import categorical_accuracy

# relative import
import sys
sys.path.append("./src")
import data

# from build_model import _get_last_layer_units_and_activation, mlp_model # ::: 



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

# let's do the simple NN
simple_net = mlp_model(layers=2, units= 64 , dropout_rate=0.2, input_shape=x_train.shape[1:], num_classes= num_classes)

simple_net.compile(loss='categorical_crossentropy', optimizer= optimizers.Adam(), metrics=['accuracy'])
simple_net.summary()

simple_net.fit(x_train_ngram, y_train, epochs=10, verbose=1)
print("training loss, accuracy: " + str(simple_net.evaluate(x_train_ngram, y_train, verbose=0)))
print("test loss, accuracy: " + str(simple_net.evaluate(x_test_ngram, y_test, verbose=0)))