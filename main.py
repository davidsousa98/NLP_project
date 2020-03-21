from utility import get_files_zip, read_txt_zip, clean, update_df, sample_excerpts, plot_cm, word_counter, save_excel, get_top_n_grams
import numpy as np
import pandas as pd
import re
import nltk
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

# Train / Test split
X_train, X_dev, y_train, y_dev = train_test_split(train_excerpt_df['text'],
                                                  train_excerpt_df['author'],
                                                  test_size=0.3,
                                                  random_state=15,
                                                  shuffle=True,
                                                  stratify=train_excerpt_df['author'])

# Oversampling
X_temp = X_train.loc[(y_train == 'AlmadaNegreiros') | (y_train == 'LuisaMarquesSilva')].values.reshape(-1, 1)
y_temp = y_train.loc[(y_train == 'AlmadaNegreiros') | (y_train == 'LuisaMarquesSilva')].values.reshape(-1, 1)

author_weights = {"AlmadaNegreiros": 200, "LuisaMarquesSilva": 200}
ros = RandomOverSampler(sampling_strategy=author_weights, random_state=15)
X_res, y_res = ros.fit_resample(X_temp, y_temp)

max_index = train_excerpt_df.index.max() + 1
X_train = X_train.append(pd.Series(X_res.flatten(), index=pd.RangeIndex(start=max_index,
                                                                        stop=max_index + sum(author_weights.values()))))
y_train = y_train.append(pd.Series(y_res.flatten(), index=pd.RangeIndex(start=max_index,
                                                                        stop=max_index + sum(author_weights.values()))))

# Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# Bag-of-words
cv = CountVectorizer(max_df=0.9, binary=True, stop_words=[".", "...", "!", "?"])  # ignores terms with document
# frequency above 0.9
X = cv.fit_transform(X_train)
y = y_train
X_dev = cv.transform(X_dev)

# N-Gram
cv = CountVectorizer(
    max_df=0.9,
    binary=False,
    stop_words=[".", "...", "!", "?"],
    ngram_range=(1,3)
)
X = cv.fit_transform(X_train)
y = y_train
X_dev = cv.transform(X_dev)

top_df = get_top_n_grams(X_train, top_k=20, n=1)
# top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=2)
# top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=3)

# TF-IDF
cv = TfidfVectorizer(max_df=0.8, stop_words=[".", "...", "!", "?"], ngram_range = (1,3))

X = cv.fit_transform(X_train)
y = y_train
X_dev = cv.transform(X_dev)

feature_names = cv.get_feature_names()

# Model
# ----------------------------------------------------------------------------------------------------------------------
# Naive Bayes Classifier
modelnaive = GaussianNB()

X_train = X_train.toarray()
modelnaive.fit(X_train, y_train)
y_pred = modelnaive.predict(X_dev)

# K-nearest neighbors
modelknn = KNeighborsClassifier(n_neighbors=5,
                                weights='distance',
                                algorithm='brute',
                                leaf_size=30,
                                p=2,
                                metric='cosine',
                                metric_params=None,
                                n_jobs=1).fit(X, y)
y_pred = modelknn.predict(X_dev)
    
# Logistic Regression
log_reg = LogisticRegression(multi_class='multinomial', random_state=15).fit(X, y)
y_pred = log_reg.predict(X_dev)

# Random Forest Classifier
rfc = RandomForestClassifier(class_weight='balanced', random_state=15).fit(X, y)
y_pred = rfc.predict(X_dev)

unique, counts = np.unique(y_pred, return_counts=True)
print(np.asarray((unique, counts)).T)

# Model evaluation
print(classification_report(y_dev, y_pred, target_names=list(np.unique(y_pred))))
evaluation_metrics = pd.DataFrame(classification_report(y_dev, y_pred, target_names=list(np.unique(y_pred)), output_dict=True))
save_excel(evaluation_metrics, 'tfidf80_ngram13_KNN5') # save the results in a excel file

plot_cm(confusion_matrix(y_dev, y_pred), list(np.unique(y_dev)))  # plot confusion matrix

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
