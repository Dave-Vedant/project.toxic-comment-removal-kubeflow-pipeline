# import libraries

import numpy as np
import pandas as pd

import os
for dir, _, filenames in os.walk('data'):
    for file in filenames:
        print(os.path.join(dir, file))

# define directories for the project
reddit_200k_train = pd.read_csv("data/reddit_200k_train.csv", delimiter=',', encoding='ISO-8859-1')
reddit_200k_test = pd.read_csv("data/reddit_200k_test.csv", delimiter=',', encoding='ISO-8859-1')
reddit_train = pd.read_csv("data/reddit_test.csv", delimiter=',', encoding='ISO-8859-1')
reddit_test = pd.read_csv("data/reddit_test.csv", delimiter=',', encoding='ISO-8859-1')

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)  
pd.set_option('display.max_colwidth', 199)  

print(reddit_200k_test.head(2))
print(reddit_200k_train.head(2))
print(reddit_train.head(2))
print(reddit_test.head(2))

# clean up unnecessary columns
reddit_train = reddit_train.drop(columns="Unnamed: 0")
reddit_train = reddit_train.drop(columns="X")
reddit_test = reddit_test.drop(columns="Unnamed: 0")
reddit_test = reddit_test.drop(columns="X")


gap_reddit_train= len(reddit_train[reddit_train['REMOVED'] == 0])- len(reddit_train[reddit_train['REMOVED'] == 1])
print("reddit_train : 0 vs 1")
print(len(reddit_train[reddit_train['REMOVED'] == 0])," vs ", len(reddit_train[reddit_train['REMOVED'] == 1])," = ", gap_reddit_train)
print("")

gap_reddit_test = len(reddit_test[reddit_test['REMOVED']== 0]) - len(reddit_test[reddit_test['REMOVED']== 1])
print("reddit_test: 0 vs 1")
print(len(reddit_test[reddit_test['REMOVED']== 0]), " vs ", len(reddit_test[reddit_test['REMOVED']== 1]))

reddit_train_numpy = reddit_train.to_numpy()
print('reddit_test_numpy_shape= ', reddit_train_numpy.shape)

reddit_test_numpy = reddit_test.to_numpy()
print('reddit_test_numpy= ', reddit_test_numpy.shape)

#collect key matrix in trainng data...
import numpy as np
import matplotlib.pyplot as plt

sample_texts = reddit_train_numpy[:,0]

num_words = [len(s.split()) for s in sample_texts]

print("max = ",np.max(num_words))
print("min = ",np.min(num_words))
print("median= ",np.median(num_words))

