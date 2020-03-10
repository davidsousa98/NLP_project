from utility import get_files_zip, read_txt_zip, clean, update_df, sample_excerpts, plot_cm, word_counter
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# nltk.download('rslp')
# nltk.download('punkt')

# TODO: Join some adjacent texts belonging to same book to obtain 1000 word excerpts
# TODO: Perform POS filtering (Viterbi algorithm)
# TODO: Balance training set according to author distribution in "population"

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
    stemmer=nltk.stem.SnowballStemmer('portuguese')
)
update_df(train_df, updates)

# Obtaining word frequency, token frequency and finding non alphanumeric characters
word_freq = word_counter(train_df["text"].to_list())  # possibly we should remove some more punctuation
token_freq = word_freq.loc[word_freq.index.str.contains("#")]
non_alphanum = word_freq.index.str.extract("([^a-zA-Z0-9])")[0]
non_alphanum = non_alphanum.loc[~non_alphanum.isna()]

# Sampling Excerpts
# ----------------------------------------------------------------------------------------------------------------------
train_df = sample_excerpts(train_df)
train_df["word_count"] = train_df["text"].apply(lambda x: len(nltk.word_tokenize(x)))  # creating number of words column
# train_df["word_count"].describe() # word count mean is around 500

# Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# Bag-of-words
cv = CountVectorizer(max_df=0.9, binary=True)  # ignores terms with document frequency above 0.9
X = cv.fit_transform(train_df["text"])
y = np.array(train_df["author"])

# Model
# ----------------------------------------------------------------------------------------------------------------------
# Train / Test split
X_train, X_dev, y_train, y_dev = train_test_split(X,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=15,
                                                  shuffle=True,
                                                  stratify=y)

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
plot_cm(confusion_matrix(predict_dev, y_dev), list(np.unique(predict_dev)))  # plot confusion matrix

# Prediction
# ----------------------------------------------------------------------------------------------------------------------
# Predict the author in the test set
test_texts = clean(
    test_df["text"],
    punctuation=['$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[',  '\\', ']', '^', '_',
                 '`', '{', '|', '}', '~'] + [',', '.', '``', '?', '#', '!', "'", '"'],
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
