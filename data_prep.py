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
# TODO: tokenize week days (e.g. sexta-feira, quinta-feira, sabado, ...), cidades portuguesas, datas, ...
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
        text = re.sub('[0-9]+', '#NUMBER', text)

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
cv = CountVectorizer(max_df=0.9, binary=True)

X = cv.fit_transform(train_df["text"])
y = np.array(train_df["author"])

# Model
# ----------------------------------------------------------------------------------------------------------------------
# K-nearest neighbors
modelknn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                         metric='cosine', metric_params=None, n_jobs=1)
modelknn.fit(X, y)

test4 = cv.transform(test_df.text)

predict = modelknn.predict(test4)
predict


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

# a = train.loc[:, "text"].apply(lambda x: re.findall("\n{3,5}([\s\S]*)", x))


# Stemmer example
# list_of_words = ["consultoria", "consultores", "consulta", "consultava"]
# for word in list_of_words:
#     print("Stem of {} is {}".format(word, stemmer.stem(word)))
#     print("----")
#
# def rm_nonalphanumeric(string):
#     re.su
#
# df["text"].apply(rm_nonalphanumeric)


# Clean data

# df['texts'].iloc[18] = token(df['texts'].iloc[18])

# a = df['texts'].apply(lambda x: re.findall('\$',x))
