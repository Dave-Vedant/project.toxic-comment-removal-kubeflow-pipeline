# import libraries

import numpy as np
import pandas as pd
import os 
import json

secret_path = '.secret/kaggle.json'

def get_keys(path):
    with open(path) as f:
        return json.load(f)

# import authentication informations (tokens)
auth_keys = get_keys(secret_path)
KAGGLE_USERNAME = auth_keys['KAGGLE_USERNAME']
KAGGLE_KEY = auth_keys["KAGGLE_KEY"]

# import authentication informations (tokens)
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME # username from the json file
os.environ['KAGGLE_KEY'] = KAGGLE_KEY # key from the json file (generate new key everytime from account setting)

try:
    os.system("kaggle competitions download science-popular-comment-removal -p Data")
except:
    print('Please download data manually, Hosting data not found! Please check competition running status and/or correct name...')

if os.path.exists("Data/reddit_200k_train.csv"):
    exit
else:
    os.system("unzip Data/reddit.zip -d Data")

print('unzip completed')

for dir, _, filenames in os.walk('Data'):
    for file in filenames:
        print(os.path.join(dir, file))


# define directories for the project
reddit_200k_train = pd.read_csv("Data/reddit_200k_train.csv", delimiter=',', encoding='ISO-8859-1')
reddit_200k_test = pd.read_csv("Data/reddit_200k_test.csv", delimiter=',', encoding='ISO-8859-1')
reddit_train = pd.read_csv("Data/reddit_test.csv", delimiter=',', encoding='ISO-8859-1')
reddit_test = pd.read_csv("Data/reddit_test.csv", delimiter=',', encoding='ISO-8859-1')

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)  
pd.set_option('display.max_colwidth', 199)  

print(reddit_200k_test.head(2))
print(reddit_200k_train.head(2))
print(reddit_train.head(2))
print(reddit_test.head(2))
