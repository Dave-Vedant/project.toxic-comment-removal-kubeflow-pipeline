import sys
sys.path.append('./src/data') # set package directory to subdirectory (one level above the .py file)
import data_preparation


# simplifying the importing ..
from .data_preparation import x_train, x_test, y_test, y_train, x_test_ngram, x_train_ngram, input_size

from .explore_data import reddit_test_numpy, reddit_train_numpy 

