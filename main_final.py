from utility import get_files_zip, read_txt_zip, TextCleaner, update_df, sample_excerpts, visualize_groups, plot_cm, \
    model_selection, model_assessment_vis, save_excel, model_assessment_author_vis
from joblib import load
import itertools
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.classifier import StackingCVClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# nltk.download('rslp')
# nltk.download('punkt')

# Changes from previous Pipeline
# ----------------------------------------------------------------------------------------------------------------------
# 1. train the model with 500 word and test both with 500 and 1000 words
# 2. split the corpora into train and test set (70/30) and deny the possibility of finding excerpts of the same book
# both in train and test set
# 3. using F1 score in Grid Search for tuning hyper-parameters

# Limitations
# ----------------------------------------------------------------------------------------------------------------------
# Weren't able to implement the TextCleaner in the Pipeline

# Building Corpus
# ----------------------------------------------------------------------------------------------------------------------
# Reading files from zip without extracting them
files = list(filter(lambda x: ".txt" in x, get_files_zip()))
df = pd.DataFrame({"file": files, "text": read_txt_zip(files)})

# Creating train_df
df["type"] = df["file"].apply(lambda x: re.findall(r"(^[a-z]+)/", x)[0])
train_df = df.loc[df["type"] == "train", ["file", "text"]].reset_index(drop=True)
train_df["author"] = train_df["file"].apply(lambda x: re.findall(r"/([a-zA-Z]+)/", x)[0])
train_df["book_id"] = train_df["file"].str.extract(r"/.+/(.+).txt")
train_df = train_df[["file", "book_id", "author", "text"]]

# Creating submission_df
submission_df = df.loc[df["type"] == "test", ["file", "text"]].reset_index(drop=True)

# Preprocessing - Reference: http://www.nltk.org/howto/portuguese_en.html
# ----------------------------------------------------------------------------------------------------------------------
cleaner = TextCleaner(
    punctuation=['$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
                 '}', '~'] + [',', '.', '``', '?', '#', '!', "'", '"'],
    stoppers=[".", "...", "!", "?"],
    stopwords=stopwords.words('portuguese'),
    stemmer="Snowball"
)

updates = cleaner.transform(train_df["text"])
update_df(train_df, updates)

# Sampling Excerpts
# ----------------------------------------------------------------------------------------------------------------------
train_excerpt_df = sample_excerpts(dataframe=train_df, stoppers=[".", "...", "!", "?"],
                                   size='large')  # short = 500 or large = 1000
# creating number of words column
train_excerpt_df["word_count"] = train_excerpt_df["text"].apply(lambda x: len(nltk.word_tokenize(x)))
train_excerpt_df["word_count"].describe()  # word count mean is around 500

# visualize excerpt partition according to Books and Authors
# visualize_groups(train_excerpt_df["author"], train_excerpt_df["book_id"])  # by visualizing the plot we can understand
# that we can't just do a stratified train_test_split as it will end up with excerpts from the same book in both the
# training and test set leading to the problem of data leakage.


# Model Selection
# ----------------------------------------------------------------------------------------------------------------------
# Reference: https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-3.html
# Train test split with stratification and same book restriction
# Grouping by book_id - like each observation were a book instead of an excerpt
groupby = train_excerpt_df[["book_id", "author"]].groupby(by="book_id").first()
X_groupby, y_groupby = groupby.index, groupby["author"].values
# Splitting books in stratified way
X_groupby_train, X_groupby_test, _, _ = train_test_split(X_groupby, y_groupby,
                                                         test_size=0.3,
                                                         random_state=0,
                                                         stratify=y_groupby)

# Obtaining the shuffled excerpt partitions and the group labels for the train data for future cross validation
train_1000df = train_excerpt_df.loc[train_excerpt_df["book_id"].isin(X_groupby_train), ["text", "author", "book_id"]].\
    sample(frac=1, random_state=15)
test_1000df = train_excerpt_df.loc[train_excerpt_df["book_id"].isin(X_groupby_test), ["text", "author", "book_id"]].\
    sample(frac=1, random_state=15)
X_1000test, y_1000test = test_1000df["text"], test_1000df["author"]

# Verifying 70/30 split
# test_1000df.shape[0]/train_excerpt_df.shape[0]
# = 0.284 -> Proportion of 1000 words excerpts belonging to test set

# Splitting excerpts in training set into 500 words and creating 500 words test set
train_500df = sample_excerpts(dataframe=train_1000df, stoppers=[".", "...", "!", "?"], size='short').\
    sample(frac=1, random_state=15)
# train_500df["text"].apply(lambda x: len(nltk.word_tokenize(x))).describe()  # mean=496, std=18
X_500train, y_500train = train_500df["text"], train_500df["author"]

test_500df = sample_excerpts(dataframe=test_1000df, stoppers=[".", "...", "!", "?"], size='short').\
    sample(frac=1, random_state=15)
# test_500df["text"].apply(lambda x: len(nltk.word_tokenize(x))).describe()  # mean=499, std=22
X_500test, y_500test = test_500df["text"], test_500df["author"]

# Verifying 70/30 split
# train_500df.shape[0]/(train_500df.shape[0] + test_500df.shape[0])
# = 0.72 -> Proportion of 500 words excerpts belonging to training set

# Over Sampling 500 words train set
author_over_weights = {"LuisaMarquesSilva": int((y_500train == "LuisaMarquesSilva").sum()*1.5)}
ros = RandomOverSampler(sampling_strategy=author_over_weights, random_state=15)
X_temp, y_temp = X_500train.values.reshape(-1, 1), y_500train.values.reshape(-1, 1)
X_500train, y_500train = ros.fit_resample(X_temp, y_temp)

# Under Sampling 500 words train set
author_under_weights = {"JoseRodriguesSantos": int((y_500train == "JoseRodriguesSantos").sum()*0.6)}
rus = RandomUnderSampler(sampling_strategy=author_under_weights, random_state=15)
# X_temp, y_temp = X_500train.reshape(-1, 1), y_500train.reshape(-1, 1)
X_500train, y_500train = rus.fit_resample(X_500train, y_500train)

# Shuffle training samples
shuffle_temp = pd.DataFrame({"X": X_500train.flatten(), "y": y_500train}).sample(frac=1, random_state=15)
X_500train, y_500train = shuffle_temp["X"].values, shuffle_temp["y"].values


# Construct some pipelines
# pipe_cv_cnb = Pipeline([('cv', CountVectorizer()),
#                         ('cnb', ComplementNB())])
#
# pipe_tfidf_cnb = Pipeline([('tfidf', TfidfVectorizer()),
#                            ('cnb', ComplementNB())])
#
# pipe_cv_knn = Pipeline([('cv', CountVectorizer()),
#                         ('knn', KNeighborsClassifier(metric='cosine'))])

pipe_tfidf_knn = Pipeline([('tfidf', TfidfVectorizer(binary=True, max_df=0.8, ngram_range=(1, 3), stop_words=[".", "...", "!", "?"])),
                             ('knn', KNeighborsClassifier(metric='cosine', n_neighbors=5, weights='distance'))])

pipe_cv_knc = Pipeline([('cv', CountVectorizer(binary=True, max_df=0.85, ngram_range=(1, 1), stop_words=[".", "...", "!", "?"])),
                        ('knc', NearestCentroid(metric='euclidean'))])


# pipe_tfidf_knc = Pipeline([('tfidf', TfidfVectorizer()),
#                            ('knc', NearestCentroid())])

pipe_cv_log = Pipeline([('cv', CountVectorizer(binary=True, max_df=0.8, ngram_range=(1, 3), stop_words=[".", "...", "!", "?"])),
                        ('log', SGDClassifier(random_state=15, alpha=0.001, penalty='l2'))])

# pipe_tfidf_log = Pipeline([('tfidf', TfidfVectorizer()),
#                            ('log', SGDClassifier(random_state=15))])
#
# pipe_cv_sgd = Pipeline([('cv', CountVectorizer()),
#                         ('sgd', SGDClassifier(random_state=15))])
#
# pipe_tfidf_sgd = Pipeline([('tfidf', TfidfVectorizer()),
#                            ('sgd', SGDClassifier(random_state=15))])
#
# pipe_cv_rfc = Pipeline([('cv', CountVectorizer(ngram_range=(1, 2), stop_words=[".", "...", "!", "?"], max_df=0.8)),
#                         ('rfc', RandomForestClassifier(class_weight='balanced', random_state=15))])
#
# pipe_tfidf_rfc = Pipeline([('tfidf', TfidfVectorizer()),
#                            ('rfc', RandomForestClassifier(class_weight='balanced', random_state=15))])
#
# pipe_cv_mlpc = Pipeline([('cv', CountVectorizer(ngram_range=(1, 2), stop_words=[".", "...", "!", "?"], max_df=0.8)),
#                          ('mlpc', MLPClassifier(hidden_layer_sizes=(50, 50), learning_rate="adaptive",
#                                                 random_state=15))])
#
# pipe_tfidf_mlpc = Pipeline([('cv', TfidfVectorizer(ngram_range=(1, 2), stop_words=[".", "...", "!", "?"])),
#                             ('mlpc', MLPClassifier(hidden_layer_sizes=(50, 50), learning_rate="adaptive",
#                                                    random_state=15))])
#
# pipe_cv_pac = Pipeline([('cv', CountVectorizer()),
#                         ('pac', PassiveAggressiveClassifier(class_weight='balanced', random_state=15))])
#
# pipe_tfidf_pac = Pipeline([('tfidf', TfidfVectorizer()),
#                            ('pac', PassiveAggressiveClassifier(class_weight='balanced', random_state=15))])
#
# # Set grid search params
# # NOTE: Keep in mind that the grid size is given by multiplication of the number of values to explore in each parameter!
# grid_params_cv_cnb = [{"cv__max_df": np.arange(0.8, 1.01, 0.05),
#                        "cv__binary": [True, False],
#                        "cv__stop_words": [[".", "...", "!", "?"], None],
#                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                        "cnb__norm": [True, False]}]
#
# grid_params_tfidf_cnb = [{"tfidf__max_df": np.arange(0.8, 1.01, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "cnb__norm": [True, False]}]
#
# grid_params_cv_knn = [{"cv__max_df": np.arange(0.8, 1.01, 0.05),
#                        "cv__binary": [True, False],
#                        "cv__stop_words": [[".", "...", "!", "?"], None],
#                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                        "knn__n_neighbors": np.arange(5, 31, 5),
#                        "knn__weights": ["uniform", "distance"]}]
#
# grid_params_tfidf_knn = [{"tfidf__max_df": np.arange(0.8, 1.01, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "knn__n_neighbors": np.arange(5, 31, 5),
#                           "knn__weights": ["uniform", "distance"]}]
#
# grid_params_cv_knc = [{"cv__max_df": np.arange(0.8, 1.01, 0.05),
#                        "cv__binary": [True, False],
#                        "cv__stop_words": [[".", "...", "!", "?"], None],
#                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                        "knc__metric": ["euclidean", "manhattan"]}]
#
# grid_params_tfidf_knc = [{"tfidf__max_df": np.arange(0.8, 1.01, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "knc__metric": ["euclidean", "manhattan"]}]
#
# grid_params_cv_log = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
#                        "cv__binary": [True, False],
#                        "cv__stop_words": [[".", "...", "!", "?"], None],
#                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                        "log__penalty": ['l1', 'l2'],
#                        "log__alpha": np.logspace(-3, 3, 7)}]
#
# grid_params_tfidf_log = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "log__penalty": ['l1', 'l2'],
#                           "log__alpha": np.logspace(-3, 3, 7)}]
#
# grid_params_cv_sgd = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
#                        "cv__binary": [True, False],
#                        "cv__stop_words": [[".", "...", "!", "?"], None],
#                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                        "sgd__penalty": ['l2', 'elasticnet'],
#                        "sgd__loss": ["modified_huber", "squared_hinge", "perceptron"]}]
#
# grid_params_tfidf_sgd = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "sgd__penalty": ['l2', 'elasticnet'],
#                           "sgd__loss": ["modified_huber", "squared_hinge", "perceptron"]}]
#
# grid_params_cv_rfc = [{"cv__binary": [True, False],
#                        "rfc__n_estimators": np.arange(100, 400, 100),
#                        "rfc__max_features": ['sqrt', 'log2'],
#                        "rfc__criterion": ["gini", "entropy"]}]
#
# grid_params_tfidf_rfc = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "rfc__n_estimators": np.arange(100, 600, 100)}]
#
# # ~12 minutes to run each
# grid_params_cv_mlpc = [{"cv__binary": [True, False],
#                         "mlpc__activation": ['tanh', 'relu'],
#                         'mlpc__alpha': [0.0001, 0.05]}]
#
# grid_params_tfidf_mlpc = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
#                            "tfidf__binary": [True, False],
#                            "mlpc__activation": ['tanh', 'relu'],
#                            'mlpc__alpha': [0.0001, 0.05]}]
#
# grid_params_cv_pac = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
#                        "cv__binary": [True, False],
#                        "cv__stop_words": [[".", "...", "!", "?"], None],
#                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                        "pac__early_stopping": [False, True],
#                        "pac__warm_start": [False, True]}]
#
# grid_params_tfidf_pac = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
#                           "tfidf__binary": [True, False],
#                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
#                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
#                           "pac__early_stopping": [False, True],
#                           "pac__warm_start": [False, True]}]
#
# # Construct grid searches
# jobs = -1
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=52)
# scoring = make_scorer(f1_score, average="macro")
#
# gs_cv_cnb = GridSearchCV(estimator=pipe_cv_cnb,
#                          param_grid=grid_params_cv_cnb,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_cnb = GridSearchCV(estimator=pipe_tfidf_cnb,
#                             param_grid=grid_params_tfidf_cnb,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# gs_cv_knn = GridSearchCV(estimator=pipe_cv_knn,
#                          param_grid=grid_params_cv_knn,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_knn = GridSearchCV(estimator=pipe_tfidf_knn,
#                             param_grid=grid_params_tfidf_knn,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# gs_cv_knc = GridSearchCV(estimator=pipe_cv_knc,
#                          param_grid=grid_params_cv_knc,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_knc = GridSearchCV(estimator=pipe_tfidf_knc,
#                             param_grid=grid_params_tfidf_knc,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# gs_cv_log = GridSearchCV(estimator=pipe_cv_log,
#                          param_grid=grid_params_cv_log,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_log = GridSearchCV(estimator=pipe_tfidf_log,
#                             param_grid=grid_params_tfidf_log,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# gs_cv_sgd = GridSearchCV(estimator=pipe_cv_sgd,
#                          param_grid=grid_params_cv_sgd,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_sgd = GridSearchCV(estimator=pipe_tfidf_sgd,
#                             param_grid=grid_params_tfidf_sgd,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# gs_cv_rfc = GridSearchCV(estimator=pipe_cv_rfc,
#                          param_grid=grid_params_cv_rfc,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_rfc = GridSearchCV(estimator=pipe_tfidf_rfc,
#                             param_grid=grid_params_tfidf_rfc,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# gs_cv_mlpc = GridSearchCV(estimator=pipe_cv_mlpc,
#                           param_grid=grid_params_cv_mlpc,
#                           scoring=scoring,
#                           cv=cv,
#                           n_jobs=jobs)
#
# gs_tfidf_mlpc = GridSearchCV(estimator=pipe_tfidf_mlpc,
#                              param_grid=grid_params_tfidf_mlpc,
#                              scoring=scoring,
#                              cv=cv,
#                              n_jobs=jobs)
#
# gs_cv_pac = GridSearchCV(estimator=pipe_cv_pac,
#                          param_grid=grid_params_cv_pac,
#                          scoring=scoring,
#                          cv=cv,
#                          n_jobs=jobs)
#
# gs_tfidf_pac = GridSearchCV(estimator=pipe_tfidf_pac,
#                             param_grid=grid_params_tfidf_pac,
#                             scoring=scoring,
#                             cv=cv,
#                             n_jobs=jobs)
#
# # List of pipelines for ease of iteration
# grids = [gs_cv_mlpc, gs_cv_rfc]
#     # gs_cv_cnb, gs_tfidf_cnb, gs_cv_knn, gs_tfidf_knn, gs_cv_log,  gs_tfidf_log, gs_cv_rfc, gs_tfidf_rfc,
#     #      gs_cv_mlpc, gs_tfidf_mlpc, gs_cv_knc, gs_tfidf_knc, gs_cv_sgd, gs_tfidf_sgd, gs_cv_pac, gs_tfidf_pac]
#
# # Dictionary of pipelines and classifier types for ease of reference
# grid_labels = ["cv_mlpc", "cv_rfc"]
#     # "cv_cnb", "tfidf_cnb", "cv_knn", "tfidf_knn", "cv_log", "tfidf_log", "cv_rfc", "tfidf_rfc",
#     #      "cv_mlpc", "tfidf_mlpc", "cv_knc", "tfidf_knc", "cv_sgd", "tfidf_sgd", "cv_pac", "tfidf_pac"]
#
# # Model Selection - Running Grid Searches
# model_selection(grids, X_500train, y_500train, [X_500test, X_1000test], [y_500test, y_1000test], grid_labels)
#
#
# # Load pickle files with fitted models
# gs_cv_cnb = load("./outputs/Pipeline_cv_cnb.pkl")
# gs_tfidf_cnb = load("./outputs/Pipeline_tfidf_cnb.pkl")
# gs_cv_knn = load("./outputs/Pipeline_cv_knn.pkl")
# gs_tfidf_knn = load("./outputs/Pipeline_tfidf_knn.pkl")
# gs_cv_knc = load("./outputs/Pipeline_cv_knc.pkl")
# gs_tfidf_knc = load("./outputs/Pipeline_tfidf_knc.pkl")
# gs_cv_log = load("./outputs/Pipeline_cv_log.pkl")
# gs_tfidf_log = load("./outputs/Pipeline_tfidf_log.pkl")
# # gs_cv_rfc = load("./outputs/Pipeline_cv_rfc.pkl")
# # gs_tfidf_rfc = load("./outputs/Pipeline_tfidf_rfc.pkl")
# gs_cv_sgd = load("./outputs/Pipeline_cv_sgd.pkl")
# gs_tfidf_sgd = load("./outputs/Pipeline_tfidf_sgd.pkl")
# # gs_cv_mlpc = load("./outputs/Pipeline_cv_mlpc.pkl")
# # gs_tfidf_mlpc = load("./outputs/Pipeline_tfidf_mlpc.pkl")
# gs_cv_pac = load("./outputs/Pipeline_cv_pac.pkl")
# gs_tfidf_pac = load("./outputs/Pipeline_tfidf_pac.pkl")
#
# # Ensemble
# # AdaBoost
# # clf = AdaBoostClassifier(base_estimator=[gs_cv_knc, gs_tfidf_knn], n_estimators=100, algorithm='SAMME', random_state=15)
# # clf.fit(X_500train, y_500train)
# # y_pred = clf.predict(X_500test)
# #
# # # Stacking classifier
# # clf = StackingClassifier(estimators=[gs_cv_knc, gs_tfidf_knn],
# #                          final_estimator=LogisticRegression(class_weight='balanced', multi_class='multinomial'))
# # clf.fit(X_500train, y_500train)
# # y_pred = clf.predict(X_500test)
# #
# # #  ('knn', gs_tfidf_knn), ('knc', gs_cv_knc)
# #
# # sclf = StackingCVClassifier(classifiers=[gs_cv_knc, gs_tfidf_knn],
# #                             meta_classifier=LogisticRegression(class_weight='balanced', multi_class='multinomial'),
# #                             random_state=15)
# # sclf.fit(X_500train, y_500train)
# # y_pred = sclf.predict(y_500test)
#
# #Vote Classifier
#     #Test ensemble combinations:
# def ensemble_vote(n_combinations, excerpt_size):
#     """
#     Parameters:
#
#     n_combinations : int
#         Number of models to be combined in the ensemble.
#
#     excerpt_size : ['short','large']
#
#         Size of the excerpt to be tested:
#         short = 500 words excerpt
#         large = 1000 words excerpt.
#         """
#
#
#     if excerpt_size == 'large':
#         X_train = X_500train
#         X_test = X_1000test
#         y_train = y_500train
#         y_test = y_1000test
#
#     elif excerpt_size == 'short':
#         X_train = X_500train
#         X_test = X_500test
#         y_train = y_500train
#         y_test = y_500test
#
#     models_list = [gs_cv_cnb, gs_tfidf_cnb, gs_cv_knn, gs_tfidf_knn, gs_cv_knc, gs_tfidf_knc,
#                    gs_cv_log,gs_tfidf_log, gs_cv_pac, gs_tfidf_pac] # gs_cv_rfc, gs_tfidf_rfc
#
#     models_labels = ['cv_cnb', 'tfidf_cnb', 'cv_knn', 'tfidf_knn','cv_knc','tfidf_knc',
#                      'cv_log','tfidf_log','cv_pac', 'tfidf_pac'] #'cv_rfc', 'tfidf_rfc'
#
#     models_comb = list(itertools.combinations(models_list, n_combinations))
#     labels_comb = list(itertools.combinations(models_labels, n_combinations))
#
#     score = []
#     for i in models_comb:
#         ensemble = EnsembleVoteClassifier(clfs=list(i),
#                                           voting='hard', refit=False)
#         ensemble.fit(X_500train, y_train)
#         y_pred = ensemble.predict(X_test)
#         score.append(f1_score(y_test, y_pred, average='macro'))
#
#         results = (dict(sorted(zip(score, labels_comb), reverse=True)))
#
#     print('The top 3 ensembles for combinations of {} and excerpt size "{}" are: \n{}'
#           .format(n_combinations,excerpt_size,results))
#
# ensemble_vote(3,'large') #short or large

tfidf_knn = pipe_tfidf_knn.fit(X_500train,X_500test)
cv_knc = pipe_cv_knc.fit(X_500train,X_500test)
cv_log = pipe_cv_log.fit(X_500train,X_500test)

#Fit ensemble:
ensemble = EnsembleVoteClassifier(clfs=[tfidf_knn,cv_knc,cv_log],
                                  voting='hard', refit=False)
ensemble.fit(X_500train, y_500train)
y_500pred = ensemble.predict(X_500test)
y_1000pred = ensemble.predict(X_1000test)

print('The score of the (tfidf_knn, cv_knc, cv_log) with 500 words is: ', f1_score(y_500test, y_500pred, average='macro'))
print('The score of the (tfidf_knn, cv_knc, cv_log) with 1000 is: ', f1_score(y_1000test, y_1000pred, average='macro'))

# Model Assessment
# ----------------------------------------------------------------------------------------------------------------------
# See reference:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
labels = {"tfidf": "TF-IDF",
          "cv": "Bag-of-Words",
          "cnb": "NaiveBayes",
          "knn": "KNearestNeighbors",
          "knc": "KNearestCentroid",
          "log": "LogisticRegression",
          "rfc": "RandomForest",
          "sgd": "StochasticGradientDescent",
          "lsvc": "LinearSupportVector",
          "mlpc": "MultiLayerPerceptron",
          "pac": "PassiveAggressive"}
xlsx_path = './outputs/Pipelines.xlsx'

# Comparing Test Error Score between Pipelines
model_assessment_vis(xlsx_path, labels)

# Creating excel with classification report of each model
# for i in grid_labels:
#     gs = load("./outputs/Pipeline_{}.pkl".format(i))
#     y_500pred, y_1000pred = gs.predict(X_500test), gs.predict(X_1000test)
#     cr_500 = pd.DataFrame(classification_report(y_500test, y_500pred, target_names=list(np.unique(y_500test)),
#                                                 output_dict=True))
#     cr_500.index = map(lambda x: x + "_500", cr_500.index.to_list())
#     cr_1000 = pd.DataFrame(classification_report(y_1000test, y_1000pred, target_names=list(np.unique(y_1000test)),
#                                                  output_dict=True))
#     cr_1000.index = map(lambda x: x + "_1000", cr_1000.index.to_list())
#     cr_total = pd.concat([cr_500, cr_1000])
#     save_excel(cr_total, i, "classification_reports")
#     del gs

# Comparing Test Error Score between Pipelines by Author
xlsx_path = './outputs/classification_reports.xlsx'
model_assessment_author_vis(xlsx_path, labels)

# Assessing best model
best_params = {'knn__n_neighbors': 5, 'knn__weights': 'distance', 'tfidf__binary': True,
               'tfidf__max_df': 0.8, 'tfidf__ngram_range': (1, 3), 'tfidf__stop_words': ['.', '...', '!', '?']}
best_gs = pipe_tfidf_knn.set_params(**best_params).fit(X_500train, y_500train)
y_500pred, y_1000pred = best_gs.predict(X_500test), best_gs.predict(X_1000test)

# Assessing emsemble
y_500pred = ensemble.predict(X_500test)
y_1000pred = ensemble.predict(X_1000test)

# classification report
print(classification_report(y_500test, y_500pred, target_names=list(np.unique(y_500test))))
print(classification_report(y_1000test, y_1000pred, target_names=list(np.unique(y_1000test))))

# import plotly.graph_objects as go
# import plotly.offline as pyo
#
# sampling_tests = pd.DataFrame({"Original": [0.92, 0.98],
#                                "OS-lms1.25": [0.93, 0.97],
#                                "OS-lms2": [0.9, 0.96],
#                                "OS-lms3": [0.89, 0.91],
#                                "OS-lms1.5 + US-jrs0.8": [0.93, 0.97],
#                                "OS-lms1.5 + US-jrs0.6": [0.94, 0.97],
#                                "OS-lms1.5 + US-jrs0.6_ccb0.8": [0.93, 0.97],
#                                "OS-lms1.5 + US-jrs0.8_ccb0.8": [0.94, 0.97],
#                                "OS-lms1.5 + US-jrs0.7_ccb0.7": [0.93, 0.96],
#                                "OS-lms2 + US-jrs0.8_ccb0.8": [0.91, 0.97]},
#                               index=["500_words", "1000_words"]).transpose()
#
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     name="500 words",
#     x=sampling_tests.index, y=sampling_tests["500_words"],
#     text=sampling_tests["500_words"]
# ))
# fig.add_trace(go.Bar(
#     name="1000 words",
#     x=sampling_tests.index, y=sampling_tests["1000_words"],
#     text=sampling_tests["1000_words"]
# ))
# fig.update_traces(
#         texttemplate='%{text:.2f}',
#         textposition='outside'
# )
# fig.update_layout(
#     barmode='group',
#     title="TF-IDF + KNearestNeighbors sampling trials comparison on test set error",
#     xaxis_title="Sampling trials",
#     yaxis_title="Test F1-Score"
# )
# pyo.plot(fig)

# plot confusion matrix
plot_cm(confusion_matrix(y_500test, y_500pred), np.unique(y_500test))
plot_cm(confusion_matrix(y_1000test, y_1000pred), np.unique(y_1000test))


# Prediction
# ----------------------------------------------------------------------------------------------------------------------
# Predict the author in the submission set
best_pipeline = gs_cv_knc
submission = cleaner.transform(submission_df["text"])
y_submission = best_pipeline.predict(submission)

# Creating csv file with predictions
submission = pd.Series(y_submission, index=submission_df.index)
submission.to_csv("./outputs/submission.csv", index=False)

# Extras
# ----------------------------------------------------------------------------------------------------------------------
# # Feature Engineering
# # Bag-of-words
# cv = CountVectorizer(max_df=0.9, binary=True, stop_words=[".", "...", "!", "?"])  # ignores terms with document
# # frequency above 0.9
# X = cv.fit_transform(X_train)
# y = y_train
# X_test = cv.transform(X_test)
#
# # N-Gram
# cv = CountVectorizer(
#     max_df=0.9,
#     binary=False,
#     stop_words=[".", "...", "!", "?"],
#     ngram_range=(1, 3)
# )
# X = cv.fit_transform(X_train)
# y = y_train
# X_test = cv.transform(X_test)
#
# top_df = get_top_n_grams(X_train, top_k=20, n=1)
# # top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=2)
# # top_df = get_top_n_grams(train_excerpt_df["text"], top_k=20, n=3)
#
# # TF-IDF
# cv = TfidfVectorizer(max_df=0.8, stop_words=[".", "...", "!", "?"], ngram_range=(1, 3))
# X = cv.fit_transform(X_train)
# y = y_train
# X_test = cv.transform(X_test)
#
# feature_names = cv.get_feature_names()
#
# # Model
# # Naive Bayes Classifier
# modelnaive = ComplementNB()  # ComplementNB appropriate for text data and imbalanced classes
# modelnaive.fit(X, y_train)
# y_pred = modelnaive.predict(X_test)
#
# # K-nearest neighbors
# modelknn = KNeighborsClassifier(n_neighbors=5,
#                                 weights='distance',
#                                 metric='cosine')
# modelknn.fit(X, y_train)
# y_pred = modelknn.predict(X_test)
#
# # Word2Vec
# # X_train = pd.DataFrame(X)
# # word2vec = Word2Vec(,min_count = 1, size = 100, window = 5)
# # y_pred = word2vec.predict(X_dev)
#
# # Logistic Regression
# log_reg = LogisticRegression(multi_class='multinomial', random_state=15).fit(X, y)
# y_pred = log_reg.predict(X_test)
#
# # Random Forest Classifier
# rfc = RandomForestClassifier(class_weight='balanced', random_state=15).fit(X, y)
# y_pred = rfc.predict(X_test)
#
# unique, counts = np.unique(y_pred, return_counts=True)
# print(np.asarray((unique, counts)).T)
#
# POS Tagging
# stok = nltk.data.load('tokenizers/punkt/portuguese.pickle')
#
# stok.tokenize(train_df["text"])
#
#  text = word_tokenize(test)
# pos = nltk.corpus.mac_morpho.tagged_words()
# nltk.corpus.mac_morpho.tagged_sents(text)

# text.tagged_words()
# X = pos.fit_transform(text)

# References
# ----------------------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/grid_search.html

#
# cv_f1500 = [0.643818771348588,0.810768480733043,0.946949887046312,0.930998002111493,0.930998080452325,0.926667]
# cv_f11000= [0.575557752051885,0.852971331291678,0.823053566426323,0.960478263714171,0.943030399251382,0.961667]
# sum(cv_f11000)/6
#
# tfidf_f1500 = [0.401747924904321,0.919370456993357,0.853246901281279,0.727165581182741,0.865584495601857,0.873333]
#
# tfidf_f11000 = [0.412220428836646,0.982182981239585,0.891593046662732,0.773217666840039,0.852135351692357,0.916667]
#
# sum(tfidf_f1500)/6
# sum(tfidf_f11000)/6
