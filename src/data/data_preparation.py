import numpy as np
import os
os.system('pip3 install category_encoder')
from sklearn.preprocessing import OneHotEncoder

from explore_data import reddit_test_numpy, reddit_train_numpy


x_train = reddit_train_numpy[:,0]
x_test = reddit_test_numpy[:,0]
y_train = reddit_train_numpy[:,1]
y_test = reddit_test_numpy[:,1]

x = np.concatenate(reddit_train_numpy[:,0], reddit_test_numpy[:,0])
print(x_train.shape, "+", x_test.shape, "=", x.shape)


def get_num_classes(labels):
    """
    get total number of classes (logic)

    # arguments
     labels: list, label values
     there shoud be at least one  example for the values in the range (0, num_classes -1)

    # return
        int, toal number of classes
    
    # Raises
        ValueError: if any label value in the range (0,num_classes-1) is missing or if number of classes is <=1

    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing sample with label value(s)' '{missing_classes}. Please make sure you hvae'
        'at least one sample for every lable value ' 'in the ramge(0,{max_class})'.format(missing_classes=missing_classes, max_class=num_classes-1))

    if num_classes <=1:
        raise ValueError('Invalid number of lables: {num_classes}.' 'Please make sure there are at least two classes '
        'of smaples'. format(num_classes = num_classes))
        
    return num_classes

num_classes = get_num_classes(y_train)  # I add..


# helper functions...
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def ngram_vectorize(train_texts, train_labels, val_texts, MIN_DOCUMENT_FREQUENCY=2, TOKEN_MODE = 'word', NGRAM_RANGE=(1,2), TOP_K = 20000):
    """Vectorizes tesxts as n-gram vectors...

    text = tf-idf vector the lenght of vocabulary of unigrams + bigrams

    # Arguments
        train_texts = list, training text strings,
        train_labels = np.ndarray,. training labels
        val_texts = list, validation text strings
        NGRAM_RANGE = Range (inclusive) of n-gram sizes for tokenizing texts
        TOP_K = limit on the number of features
        one of 'word', 'char', whether text should be split into word or characters n-gram
        MIN_DOCUMENT_FREQUNCY = Min doc/corpus frequency below which a token will be discarded..

    # Return
        x_train, x_val = vectorized training and valication texts

    """
    # create teh key argumnet rules for the tf-idf vectorizer
    kwargs = {
        'ngram_range': NGRAM_RANGE, # USE 1-gram + 2-grams here in the alog...
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error' :'replace',
        'analyzer': TOKEN_MODE, # split text into word token
        'min_df' : MIN_DOCUMENT_FREQUENCY
    }
    vectorize = TfidfVectorizer(**kwargs) # load the rules...

    # learn vocabular from training text and vectorize training texts
    x_train = vectorize.fit_transform(train_texts)

    # vectorize validation texts
    x_val = vectorize.fit_transform(val_texts)

    # select top 'k' on the vectorized features...
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val


x_train_ngram, x_test_ngram = ngram_vectorize(x_train, x_test)
print('x_train_ngram.shape= ', x_train_ngram.shape)
print('x_test_ngram.shape= ', x_test_ngram.shape)

ohe = OneHotEncoder(handle_unknown='ignore')
y_train = ohe.fit_transform(reddit_train_numpy[:,1])
y_test  = ohe.fit_transform(reddit_test_numpy[:,1])
print(y_train)
print("transformed into")
print(reddit_train_numpy[0:4,1])
print(reddit_train_numpy[13709, 13717,1])


# rename input and labels...
print(type(x_train_ngram))

x_train_ngram = x_train_ngram.todense()
print(type(x_train_ngram))
print(x_train_ngram.shape)

input_size = x_train_ngram.shape[1]

