from tqdm import tqdm_notebook as tqdm
import unicodedata
import io
import requests
import zipfile as zp
import numpy as np
import pandas as pd
import itertools
import re
from bs4 import BeautifulSoup
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# nltk.download('rslp')
# nltk.download('punkt')

# TODO: Join some adjacent texts belonging to same book to obtain 1000 word excerpts
# TODO: exclude the text corresponding to the beginning of the book
# TODO: Perform POS filtering (Viterbi algorithm)
# TODO: Balance training set according to author distribution in "population"

# Building Corpus
# ----------------------------------------------------------------------------------------------------------------------
def get_files_zip():
    with zp.ZipFile("./nlp_data.zip") as myzip:
        list = myzip.namelist()
    return list


def read_txt_zip(files):
    df_list = []
    for file in files:
        with zp.ZipFile("./nlp_data.zip") as myzip:
            with myzip.open(file) as myfile:
                df_list.append(myfile.read().decode("utf-8"))
    return df_list


# Reading files from zip without extracting them
files = list(filter(lambda x: ".txt" in x, get_files_zip()))
texts = read_txt_zip(files)
df = pd.DataFrame({"file": files, "text": texts})

# Creating train_df and test_df
df["type"] = df["file"].apply(lambda x: re.findall(r"(^[a-z]+)/", x)[0])
train_df = df.loc[df["type"] == "train", ["file", "text"]].reset_index(drop=True)
train_df["author"] = train_df["file"].apply(lambda x: re.findall(r"/([a-zA-Z]+)/", x)[0])
train_df["book_id"] = train_df["file"].str.extract(r"/.+/(.+).txt")
train_df = train_df[["file", "book_id", "author", "text"]]
test_df = df.loc[df["type"] == "test", ["file", "text"]].reset_index(drop=True)

# Preprocessing - Reference: http://www.nltk.org/howto/portuguese_en.html
# ----------------------------------------------------------------------------------------------------------------------

url = "https://simplemaps.com/static/data/country-cities/pt/pt.csv"
s = requests.get(url).content
cities = pd.read_csv(io.StringIO(s.decode('utf-8')))  # Cities for tokenization

def clean(text_list, punctuation=None, lemmatize=None, stemmer=None):
    """
    Function that a receives a list of strings and preprocesses it.

    :param text_list: List of strings.
    :param punctuation: list of punctuation to exclude.
    :param lemmatize: lemmatizer to apply if provided.
    :param stemmer: stemmer to apply if provided.
    """
    updates = []
    for j in tqdm(range(len(text_list))):

        text = text_list[j]

        # LOWERCASING
        text = text.lower()

        # TOKENIZATION
        text = re.sub('\S+\$\d+( réis)*', '#PRICE', text)  # price token
        text = re.sub('(\d{1,2}\D\d{1,2}\D\d{2,4})|(\d{4}\D\d{1,2}\D\d{1,2})|(\d{1,2} de [a-zA-Z]+ de \d{2,4})',
                      "#DATE", text)  # date token
        text = re.sub('[0-9]+', '#NUMBER', text)  # number token

        # REMOVE PUNCTUATION - Even if we end up not removing punctuation we need to watch out with some symbols
        # (e.g. *, %, >)
        if punctuation:
            for i in punctuation:
                text = text.replace(i, " ")

        # REMOVE STOPWORDS
        stop = set(nltk.corpus.stopwords.words('portuguese'))
        text = " ".join(" " if word in stop else word for word in text.split())

        # REMOVE HTML TAGS
        text = BeautifulSoup(text).get_text()

        if lemmatize:
            lemma = None  # Not defined
            text = " ".join(lemma.lemmatize(word) for word in text.split())

        if stemmer:
            # stemmer = nltk.stem.RSLPStemmer()
            # stemmer = nltk.stem.SnowballStemmer('portuguese')
            text = " ".join(stemmer.stem(word) for word in text.split())

        # REMOVING ACCENTUATION
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn')

        updates.append(text)

    return updates


def update_df(dataframe, list_updated):
    dataframe.update(pd.Series(list_updated, name="text"))  # Modify in place using non-NA values from another DataFrame


updates = clean(
    train_df["text"],
    punctuation=['#', '$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[',  '\\', ']', '^', '_',
                 '`', '{', '|', '}', '~'],
    stemmer=nltk.stem.SnowballStemmer('portuguese')
)
update_df(train_df, updates)

# # Checking words frequency
# def word_freq(list_strings, top):
#     words = []
#     for string in list_strings:
#         words += string.split()
#     fd = nltk.FreqDist(words)
#     word_freqs = []
#     for word in list(fd.keys()):
#         word_freqs.append(word, fd[word])
#
# word_freq(train_df["text"].to_list(), 30)

# Sampling Excerpts
# ----------------------------------------------------------------------------------------------------------------------
def sample_excerpts(data):

    dataframe = []
    for _, row in data.iterrows():
        words = nltk.word_tokenize(row["text"])
        groups = []
        group = []
        stoppers = [".", "...", "!", "?"]
        for word in words:
            group.append(word)
            if (len(group) >= 480) & (group[-1] in stoppers):
                groups.append((row["book_id"], row["author"], " ".join(group)))
                group = []
            elif len(group) >= 600:  # Dealing with lack of punctuation
                groups.append((row["book_id"], row["author"], " ".join(group)))
                group = []
        dataframe += groups
    return pd.DataFrame(dataframe, columns=["book_id", "author", "text"])


train_df = sample_excerpts(train_df)
train_df["word_count"] = train_df["text"].apply(lambda x: len(nltk.word_tokenize(x)))  # creating number of words column
# train_df["word_count"].describe()

# Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# Bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

cv = CountVectorizer(max_df=0.9, binary=True)

X = cv.fit_transform(train_df["text"])
y = np.array(train_df["author"])

# Model
# ----------------------------------------------------------------------------------------------------------------------
# K-nearest neighbors
from sklearn.model_selection import train_test_split


X_train, X_dev, y_train, y_dev = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=15,
                                                    shuffle=True,
                                                    stratify=y)

# K-nearest neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier

modelknn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                         metric='cosine', metric_params=None, n_jobs=1)
modelknn.fit(X_train, y_train)

predict_dev = modelknn.predict(X_dev)

# Model evaluation
from sklearn.metrics import classification_report
print(classification_report(predict_dev, y_dev, target_names=np.unique(predict_dev)))


from sklearn.metrics import confusion_matrix
confusion_matrix(predict_dev, y_dev)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_cm(confusion_matrix: np.array,
            classnames: list):
    """
    Function that creates a confusion matrix plot using the Wikipedia convention for the axis.
    :param confusion_matrix: confusion matrix that will be plotted
    :param classnames: labels of the classes

    Returns:
        - Plot of the Confusion Matrix
    """

    confusionmatrix = confusion_matrix
    class_names = classnames

    fig, ax = plt.subplots()
    im = plt.imshow(confusionmatrix, cmap=plt.cm.cividis)
    plt.colorbar()

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusionmatrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.ylim(top=len(class_names) - 0.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-0.5)  # adjust the bottom leaving top unchanged
    return plt.show()

plot_cm(confusion_matrix(predict_dev, y_dev), np.unique(predict_dev))

# Prediction
# ----------------------------------------------------------------------------------------------------------------------
# Predict the author in the test set
pred_test = cv.transform(test_df["text"])
predict_test = modelknn.predict(pred_test)
predict_test

# Extras
# ----------------------------------------------------------------------------------------------------------------------
def word_counter(text_list):
    """
    Function that receives a list of strings and returns the frequency of each word
    in the set of all strings.
    """
    split = [i.split() for i in text_list]
    words_in_df = list(itertools.chain.from_iterable(split)) # merges lists in list
    # Count all words
    freq = pd.Series(words_in_df).value_counts()
    return freq

word_count = word_counter(df["text"].to_list())
word_count.head(20)







