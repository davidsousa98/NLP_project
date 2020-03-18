from utility import get_files_zip, read_txt_zip, clean, update_df, sample_excerpts, plot_cm, word_counter, save_excel, get_top_n_grams
import numpy as np
import pandas as pd
import re
import nltk
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('rslp')
# nltk.download('punkt')

# TODO: Join some adjacent texts belonging to same book to obtain 1000 word excerpts
# TODO: Perform POS filtering (Viterbi algorithm)
# TODO: Balance training set according to author distribution in "population" (population distribution reflected in
#  training set?)

# Building Corpus
# ----------------------------------------------------------------------------------------------------------------------
# Reading files from zip without extracting them
files = list(filter(lambda x: ".txt" in x, get_files_zip()))
df = pd.DataFrame({"file": files,
                   "text": read_txt_zip(files)})

# Creating train_df
df["type"] = df["file"].apply(lambda x: re.findall(r"(^[a-z]+)/", x)[0])
train_df = df.loc[df["type"] == "train", ["file", "text"]].reset_index(drop=True)
train_df["author"] = train_df["file"].apply(lambda x: re.findall(r"/([a-zA-Z]+)/", x)[0])
train_df["book_id"] = train_df["file"].str.extract(r"/.+/(.+).txt")
train_df = train_df[["file", "book_id", "author", "text"]]

# Creating test_df
test_df = df.loc[df["type"] == "test", ["file", "text"]].reset_index(drop=True)

# Preprocessing - Reference: http://www.nltk.org/howto/portuguese_en.html
# ----------------------------------------------------------------------------------------------------------------------
updates = clean(
    train_df["text"],
    punctuation=['$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[',  '\\', ']', '^', '_',
                 '`', '{', '|', '}', '~'] + [',', '.', '``', '?', '#', '!', "'", '"'],
    stoppers=[".", "...", "!", "?"],
    stemmer=nltk.stem.SnowballStemmer('portuguese')
)
update_df(train_df, updates)

# Data Exploration
# ----------------------------------------------------------------------------------------------------------------------
# See Keyword Extraction Notebook (words frequency, top unigrams, bigrams, ...)
# Obtaining word frequency, token frequency and finding non alphanumeric characters
word_freq = word_counter(train_df["text"].to_list())  # possibly we should remove some more punctuation
token_freq = word_freq.loc[word_freq.index.str.contains("#")]
non_alphanum = word_freq.index.str.extract("([^a-zA-Z0-9])")[0]
non_alphanum = non_alphanum.loc[~non_alphanum.isna()]

# Sampling Excerpts
# ----------------------------------------------------------------------------------------------------------------------
train_excerpt_df = sample_excerpts(dataframe=train_df,
                                   stoppers=[".", "...", "!", "?"])
# creating number of words column
train_excerpt_df["word_count"] = train_excerpt_df["text"].apply(lambda x: len(nltk.word_tokenize(x)))
train_excerpt_df["word_count"].describe()  # word count mean is around 500

# Balancing the excerpts according to the target population distribution
# original_props = {"JoseRodriguesSantos": 52,
#                   "JoseSaramago": 79,
#                   "CamiloCasteloBranco": 131,
#                   "EcaDeQueiros": 33,
#                   "AlmadaNegreiros": 59,
#                   "LuisaMarquesSilva": 59}
# rus = RandomUnderSampler(random_state=15, sampling_strategy=original_props)
# X_res, y_res = rus.fit_resample(train_excerpt_df.drop("author", axis=1), train_excerpt_df["author"])
# train_excerpt_df = pd.concat([X_res, y_res], axis=1)
# train_excerpt_df["author"].value_counts()/train_excerpt_df.shape[0]
# train_df["author"].value_counts()/train_df.shape[0]

# Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# Bag-of-words
cv = CountVectorizer(max_df=0.9, binary=True, stop_words=[".", "...", "!", "?"])  # ignores terms with document
# frequency above 0.9
X = cv.fit_transform(train_excerpt_df["text"])
y = np.array(train_excerpt_df["author"])

# N-Gram
cv = CountVectorizer(
    max_df=0.9,
    binary=False,
    stop_words=[".", "...", "!", "?"],
    ngram_range=(1,3)
)
X = cv.fit_transform(train_excerpt_df["text"])
y = np.array(train_excerpt_df["author"])

top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=1)
# top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=2)
# top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=3)

# TF-IDF

cv = TfidfVectorizer(max_df=0.9, stop_words=[".", "...", "!", "?"], ngram_range = (1,3))

X = cv.fit_transform(train_excerpt_df["text"])

feature_names = cv.get_feature_names()

# Model
# ----------------------------------------------------------------------------------------------------------------------
# Train / Test split
X_train, X_dev, y_train, y_dev = train_test_split(X,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=15,
                                                  shuffle=True,
                                                  stratify=y)

#Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB
modelnaive = GaussianNB()

X_train = X_train.toarray()
modelnaive.fit(X_train, y_train)
predict_dev = modelnaive.predict(X_dev)

# K-nearest neighbors
modelknn = KNeighborsClassifier(n_neighbors=5,
                                weights='distance',
                                algorithm='brute',
                                leaf_size=30,
                                p=2,
                                metric='cosine',
                                metric_params=None,
                                n_jobs=1)
modelknn.fit(X_train, y_train)
predict_dev = modelknn.predict(X_dev)

# Model evaluation
print(classification_report(predict_dev, y_dev, target_names=list(np.unique(predict_dev))))
evaluation_metrics = pd.DataFrame(classification_report(predict_dev, y_dev, target_names=list(np.unique(predict_dev)), output_dict=True))
save_excel(evaluation_metrics, 'NGRAM13_KNN5') # save the results in a excel file

plot_cm(confusion_matrix(predict_dev, y_dev), list(np.unique(predict_dev)))  # plot confusion matrix

# Prediction
# ----------------------------------------------------------------------------------------------------------------------
# Predict the author in the test set
test_texts = clean(
    test_df["text"],
    punctuation=['$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[',  '\\', ']', '^', '_',
                 '`', '{', '|', '}', '~'] + [',', '.', '``', '?', '#', '!', "'", '"'],
    stoppers=[".", "...", "!", "?"],
    stemmer=nltk.stem.SnowballStemmer('portuguese')
)
X_test = cv.transform(test_texts)
predict_test = modelknn.predict(X_test)

# Creating csv file with predictions
submission = test_df
submission["prediction"] = predict_test
submission.to_csv("./submission.csv", index=False)

# Extras
# ----------------------------------------------------------------------------------------------------------------------
