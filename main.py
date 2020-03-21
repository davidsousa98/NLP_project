from utility import get_files_zip, read_txt_zip, clean, update_df, sample_excerpts, plot_cm, word_counter, save_excel, \
    get_top_n_grams, model_selection
import numpy as np
import pandas as pd
import re
import nltk
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer
from sklearn.pipeline import Pipeline


# nltk.download('rslp')
# nltk.download('punkt')

# TODO: Join some adjacent texts belonging to same book to obtain 1000 word excerpts
# TODO: Perform POS tagging (Viterbi algorithm). 1) filter out unnecessary tags; 2) count the number of each tag
# TODO: Perform ensemble on best models

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
submission_df = df.loc[df["type"] == "test", ["file", "text"]].reset_index(drop=True)

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
X_train, X_test, y_train, y_test = train_test_split(train_excerpt_df['text'],
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
X_test = cv.transform(X_test)

# N-Gram
cv = CountVectorizer(
    max_df=0.9,
    binary=False,
    stop_words=[".", "...", "!", "?"],
    ngram_range=(1, 3)
)
X = cv.fit_transform(X_train)
y = y_train
X_test = cv.transform(X_test)

top_df = get_top_n_grams(X_train, top_k=20, n=1)
# top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=2)
# top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=3)

# TF-IDF
cv = TfidfVectorizer(max_df=0.8, stop_words=[".", "...", "!", "?"], ngram_range = (1,3))
X = cv.fit_transform(X_train)
y = y_train
X_test = cv.transform(X_test)

feature_names = cv.get_feature_names()

# Model
# ----------------------------------------------------------------------------------------------------------------------
#Naive Bayes Classifier
modelnaive = ComplementNB()  # ComplementNB appropriate for text data and imbalanced classes
modelnaive.fit(X_train, y_train)
y_pred = modelnaive.predict(X_test)

# K-nearest neighbors
modelknn = KNeighborsClassifier(n_neighbors=5,
                                weights='distance',
                                metric='cosine')
modelknn.fit(X_train, y_train)
y_pred = modelknn.predict(X_test)

# Logistic Regression
log_reg = LogisticRegression(multi_class='multinomial', random_state=15).fit(X, y)
y_pred = log_reg.predict(X_test)

# Random Forest Classifier
rfc = RandomForestClassifier(class_weight='balanced', random_state=15).fit(X, y)
y_pred = rfc.predict(X_test)

unique, counts = np.unique(y_pred, return_counts=True)
print(np.asarray((unique, counts)).T)

# Model Selection
# ----------------------------------------------------------------------------------------------------------------------
# Reference: https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-3.html
X = np.array(train_excerpt_df["text"])
y = np.array(train_excerpt_df["author"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=15,
                                                    shuffle=True,
                                                    stratify=y)

# Construct some pipelines
pipe_cv_cnb = Pipeline([('cv', CountVectorizer()),
                        ('cnb', ComplementNB())])

pipe_tfidf_cnb = Pipeline([('tfidf', TfidfVectorizer()),
                           ('cnb', ComplementNB())])

pipe_cv_knn = Pipeline([('cv', CountVectorizer()),
                        ('knn', KNeighborsClassifier(metric='cosine'))])

pipe_tfidf_knn = Pipeline([('tfidf', TfidfVectorizer()),
                           ('knn', KNeighborsClassifier(metric='cosine'))])

# Set grid search params
grid_params_cv_cnb = [{"cv__max_df": np.arange(0.8, 1, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "cnb__norm": [True, False]}]

grid_params_tfidf_cnb = [{"tfidf__max_df": np.arange(0.8, 1, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "cnb__norm": [True, False]}]

grid_params_cv_knn = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "knn__n_neighbors": np.arange(5, 31, 5),
                       "knn__weights": ["uniform", "distance"]}]

grid_params_tfidf_knn = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "knn__n_neighbors": np.arange(5, 31, 5),
                          "knn__weights": ["uniform", "distance"]}]

# Construct grid searches
jobs = -1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=52)
scoring = make_scorer(recall_score, average="macro")

gs_cv_cnb = GridSearchCV(estimator=pipe_cv_cnb,
                         param_grid=grid_params_cv_cnb,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_cnb = GridSearchCV(estimator=pipe_tfidf_cnb,
                            param_grid=grid_params_tfidf_cnb,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

gs_cv_knn = GridSearchCV(estimator=pipe_cv_knn,
                         param_grid=grid_params_cv_knn,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_knn = GridSearchCV(estimator=pipe_tfidf_knn,
                            param_grid=grid_params_tfidf_knn,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

# List of pipelines for ease of iteration
grids = [gs_cv_cnb, gs_tfidf_cnb, gs_cv_knn, gs_tfidf_knn]

# Dictionary of pipelines and classifier types for ease of reference
grid_labels = ['CountVectorizer, ComplementNB', 'TfidfVectorizer, ComplementNB',
               'CountVectorizer, KNeighborsClassifier', 'TfidfVectorizer, KNeighborsClassifier']

# Fit the grid search objects
model_selection(grids, X_train, y_train, X_test, y_test, grid_labels)

# Model Assessment
# ----------------------------------------------------------------------------------------------------------------------
print(classification_report(y_test, y_pred, target_names=list(np.unique(y_test))))
evaluation_metrics = pd.DataFrame(classification_report(y_test, y_pred, target_names=list(np.unique(y_test)),
                                                        output_dict=True))
save_excel(evaluation_metrics, 'NGRAM13_KNN5')  # save the results in a excel file

plot_cm(confusion_matrix(y_test, y_pred), list(np.unique(y_pred)))  # plot confusion matrix

# Prediction
# ----------------------------------------------------------------------------------------------------------------------
# Predict the author in the submission set
test_texts = clean(
    submission_df["text"],
    punctuation=['$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[',  '\\', ']', '^', '_',
                 '`', '{', '|', '}', '~'] + [',', '.', '``', '?', '#', '!', "'", '"'],
    stoppers=[".", "...", "!", "?"],
    stemmer=nltk.stem.SnowballStemmer('portuguese')
)
X_test = cv.transform(test_texts)
predict_test = modelknn.predict(X_test)

# Creating csv file with predictions
submission = submission_df
submission["prediction"] = predict_test
submission.to_csv("./submission.csv", index=False)

# Extras
# ----------------------------------------------------------------------------------------------------------------------
