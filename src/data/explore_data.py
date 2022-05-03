import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from numpy import vectorize
from phonenumbers import country_code_for_valid_region
from sklearn.feature_extraction.text import CountVectorizer

from load_data import reddit_train, reddit_test

# clean up unnecessary columns
reddit_train = reddit_train.drop(columns="Unnamed: 0")
reddit_train = reddit_train.drop(columns="X")
reddit_test = reddit_test.drop(columns="Unnamed: 0")
reddit_test = reddit_test.drop(columns="X")


# find gap between the data with different classes/labels
gap_reddit_train= len(reddit_train[reddit_train['REMOVED'] == 0])- len(reddit_train[reddit_train['REMOVED'] == 1])
print("reddit_train : 0 vs 1")
print(len(reddit_train[reddit_train['REMOVED'] == 0])," vs ", len(reddit_train[reddit_train['REMOVED'] == 1])," = ", gap_reddit_train)
print("")

gap_reddit_test = len(reddit_test[reddit_test['REMOVED']== 0]) - len(reddit_test[reddit_test['REMOVED']== 1])
print("reddit_test: 0 vs 1")
print(len(reddit_test[reddit_test['REMOVED']== 0]), " vs ", len(reddit_test[reddit_test['REMOVED']== 1]))


# convert to numpy array
reddit_train_numpy = reddit_train.to_numpy()
print('reddit_test_numpy_shape= ', reddit_train_numpy.shape)

reddit_test_numpy = reddit_test.to_numpy()
print('reddit_test_numpy= ', reddit_test_numpy.shape)


#collect key matrix in trainng data...
sample_texts = reddit_train_numpy[:,0]

num_words = [len(s.split()) for s in sample_texts]

print("max = ",np.max(num_words))
print("min = ",np.min(num_words))
print("median= ",np.median(num_words))


# helper functions....
def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plots the frequency distribution of n-grams.

    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': (1, 1),
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    fig = plt.figure()
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    fig.savefig('output/freq_dist_n_grams.png')



def plot_sample_length_distribution(sample_texts):
    """
    plots the sample length distribution...

    #Arguments:
        sample_texts: list, sample texts

    # return
    plot the sample distribution
    """
    fig = plt.figure()
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel("Length of a sample")
    plt.ylabel("Number of sample")
    plt.title("Sample length distribution")
    fig.savefig('output/sample_length_distribution.png')



frequency_distribution = plot_frequency_distribution_of_ngrams(sample_texts, ngram_range=(1,2), num_ngrams=50)

sample_length_distribution = plot_sample_length_distribution(sample_texts)


# calculate ratio for the model selection
ratio = reddit_train_numpy.shape[0]/ np.median(num_words)
print(ratio)