import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

# relative import
import sys
sys.path.append("./src")
import data

# load the pre-trained word-embedding vectors
embedding_index ={}

for i, line in enumerate(open('../data/')):
    values = line.split()
    embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(x_train)
word_index=token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectos
x_train_seq = sequence.pad_sequences(token.texts_to_sequence(x_train), maxlen=70)
x_test_seq = sequence.pad_sequences(token.texts_to_sequence(x_test), maxlen=70)
print('x_train_seq.shape', x_train_seq.shape)
print('x_test_seq.shape', x_test_seq.shape)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def create_bidirectional_rnn():
    input_layer = layers.Input((70,))

    # add the word embeddings...
    embedding_layer = layers.Embedding(len(word_index) +1, 300, weights= [embedding_matrix], trainable=False) (input_layer)
    embedding_layer = layers.SpatialDropout(0.3)(embedding_layer)

    # add the bidirectional layer
    bidir_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # add the output layer
    output_layer1  = layers.Dense(50, acitavation='relu')(bidir_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(2, activation='sigmoid')(output_layer1)

    # compile the model
    model = models.Model(inputs= input_layer, outputs= output_layer2)
    model.compile(optimizer= optimizers.Adam(), loss='binary_crossentropy')
    return model

Bidirectional_classifier = create_bidirectional_rnn()
Bidirectional_classifier.fit(x_train_seq, y_train, batch_size=2000, epochs=10, shuffle=True)
Bidirectional_classifier.summary()

predictions = Bidirectional_classifier.predict(x_test_seq)
predictions = np.argmax(predictions, axis=1)

predictions2 = Bidirectional_classifier.predict(x_train_seq)
predictions2 = np.argmax(predictions2, axis=1)


y_test_int64 = reddit_test_numpy[:,1].astype(int)
y_train_int64 = reddit_train_numpy[:,1].astype(int)

print("RNN-Bidirectional, word embedding accurancy (Ein)= ", accuracy_score(predictions2, y_train_int64))
print("RNN-Bidirectional, word embedding accurancy (Eout)= ", accuracy_score(predictions, y_train_int64))