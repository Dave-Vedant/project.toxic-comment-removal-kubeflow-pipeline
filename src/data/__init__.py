import sys
sys.path.append('./src/data') # set package directory to subdirectory (one level above the .py file)

# implicite importing

# importing prepared data for models...
from .data_preparation import x_train, x_test, y_test, y_train, x_test_ngram, x_train_ngram, input_size


# numpy conversion of dataset...
from .explore_data import reddit_test_numpy, reddit_train_numpy 

# load main dataset...
from .load_data import reddit_train, reddit_test, reddit_200k_test, reddit_200k_train

