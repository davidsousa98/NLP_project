from utility import get_files_zip, read_txt_zip, TextCleaner, update_df, sample_excerpts, plot_cm, word_counter, save_excel, \
    get_top_n_grams, model_selection, model_assessment_vis
from joblib import load
import itertools
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, RSLPStemmer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.classifier import EnsembleVoteClassifier

# nltk.download('rslp')
# nltk.download('punkt')

# TODO: Join some adjacent texts belonging to same book to obtain 1000 word excerpts
# TODO: Perform ensemble on best models
# TODO: Use word2vec to represent words


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

# Creating test_df
submission_df = df.loc[df["type"] == "test", ["file", "text"]].reset_index(drop=True)

# Preprocessing - Reference: http://www.nltk.org/howto/portuguese_en.html
# ----------------------------------------------------------------------------------------------------------------------
# TextCleaner behaves like a sklearn estimator. We built it this way only after realising we could integrate it in a
# sklearn Pipeline. However we realised afterwards that this would only work if we would have produced the excerpts
# before the cleaning instead of after. This would generate different data and since we had already fitted some models
# with the previous data we decided not to include it. However everything is built in order to integrate it in the
# Pipeline and fit a GridSearch to find out the overall best combination of parameters

cleaner = TextCleaner(
    punctuation=['$', '%', '&', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
                 '}', '~'] + [',', '.', '``', '?', '#', '!', "'", '"'],
    stoppers=[".", "...", "!", "?"],
    stopwords=stopwords.words('portuguese'),
    stemmer=SnowballStemmer('portuguese')  # RSLPStemmer('portuguese')
)

updates = cleaner.transform(train_df["text"])
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

# # Train / Test split
# X_train, X_test, y_train, y_test = train_test_split(train_excerpt_df['text'],
#                                                     train_excerpt_df['author'],
#                                                     test_size=0.3,
#                                                     random_state=15,
#                                                     shuffle=True,
#                                                     stratify=train_excerpt_df['author'])
#
# # Oversampling
# X_temp = X_train.loc[(y_train == 'AlmadaNegreiros') | (y_train == 'LuisaMarquesSilva')].values.reshape(-1, 1)
# y_temp = y_train.loc[(y_train == 'AlmadaNegreiros') | (y_train == 'LuisaMarquesSilva')].values.reshape(-1, 1)
#
# author_weights = {"AlmadaNegreiros": 200, "LuisaMarquesSilva": 200}
# ros = RandomOverSampler(sampling_strategy=author_weights, random_state=15)
# X_res, y_res = ros.fit_resample(X_temp, y_temp)
#
# max_index = train_excerpt_df.index.max() + 1
# X_train = X_train.append(pd.Series(X_res.flatten(), index=pd.RangeIndex(start=max_index,
#                                                                         stop=max_index + sum(author_weights.values()))))
# y_train = y_train.append(pd.Series(y_res.flatten(), index=pd.RangeIndex(start=max_index,
#                                                                         stop=max_index + sum(author_weights.values()))))
#
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

pipe_cv_knc = Pipeline([('cv', CountVectorizer()),
                        ('knc', NearestCentroid())])

pipe_tfidf_knc = Pipeline([('tfidf', TfidfVectorizer()),
                           ('knc', NearestCentroid())])

pipe_cv_log = Pipeline([('cv', CountVectorizer()),
                        ('log', SGDClassifier(random_state=15))])

pipe_tfidf_log = Pipeline([('tfidf', TfidfVectorizer()),
                           ('log', SGDClassifier(random_state=15))])

pipe_cv_sgd = Pipeline([('cv', CountVectorizer()),
                        ('sgd', SGDClassifier(random_state=15))])

pipe_tfidf_sgd = Pipeline([('tfidf', TfidfVectorizer()),
                           ('sgd', SGDClassifier(random_state=15))])

pipe_cv_rfc = Pipeline([('cv', CountVectorizer()),
                        ('rfc', RandomForestClassifier(class_weight='balanced', random_state=15))])

pipe_tfidf_rfc = Pipeline([('tfidf', TfidfVectorizer()),
                           ('rfc', RandomForestClassifier(class_weight='balanced', random_state=15))])

pipe_cv_lsvc = Pipeline([('cv', CountVectorizer()),
                         ('lsvc', LinearSVC(random_state=15))])

pipe_tfidf_lsvc = Pipeline([('cv', TfidfVectorizer()),
                            ('lsvc', LinearSVC(random_state=15))])

pipe_cv_mlpc = Pipeline([('cv', CountVectorizer()),
                         ('mlpc', MLPClassifier(random_state=15))])

pipe_tfidf_mlpc = Pipeline([('cv', TfidfVectorizer()),
                            ('mlpc', MLPClassifier(random_state=15))])

pipe_cv_pac = Pipeline([('cv', CountVectorizer()),
                        ('pac', PassiveAggressiveClassifier(class_weight='balanced', random_state=15))])

pipe_tfidf_pac = Pipeline([('tfidf', TfidfVectorizer()),
                           ('pac', PassiveAggressiveClassifier(class_weight='balanced', random_state=15))])

# Set grid search params
grid_params_cv_cnb = [{"cv__max_df": np.arange(0.8, 1.01, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "cnb__norm": [True, False]}]

grid_params_tfidf_cnb = [{"tfidf__max_df": np.arange(0.8, 1.01, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "cnb__norm": [True, False]}]

grid_params_cv_knn = [{"cv__max_df": np.arange(0.8, 1.01, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "knn__n_neighbors": np.arange(5, 31, 5),
                       "knn__weights": ["uniform", "distance"]}]

grid_params_tfidf_knn = [{"tfidf__max_df": np.arange(0.8, 1.01, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "knn__n_neighbors": np.arange(5, 31, 5),
                          "knn__weights": ["uniform", "distance"]}]

grid_params_cv_knc = [{"cv__max_df": np.arange(0.8, 1.01, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "knc__metric": ["euclidean", "manhattan"]}]

grid_params_tfidf_knc = [{"tfidf__max_df": np.arange(0.8, 1.01, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "knc__metric": ["euclidean", "manhattan"]}]

grid_params_cv_log = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "log__penalty": ['l1', 'l2'],
                       "log__alpha": np.logspace(-3, 3, 7)}]

grid_params_tfidf_log = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "log__penalty": ['l1', 'l2'],
                          "log__alpha": np.logspace(-3, 3, 7)}]

grid_params_cv_sgd = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "sgd__penalty": ['l2', 'elasticnet'],
                       "sgd__loss:": ["modified_huber", "squared_hinge", "perceptron"]}]

grid_params_tfidf_sgd = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "sgd__penalty": ['l2', 'elasticnet'],
                          "sgd__loss:": ["modified_huber", "squared_hinge", "perceptron"]}]

grid_params_cv_rfc = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "rfc__n_estimators": np.arange(100, 600, 100)}]

grid_params_tfidf_rfc = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "rfc__n_estimators": np.arange(100, 600, 100)}]

grid_params_cv_lsvc = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                        "cv__binary": [True, False],
                        "cv__stop_words": [[".", "...", "!", "?"], None],
                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                        }]  # Missing parameters

grid_params_tfidf_lsvc = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                           "tfidf__binary": [True, False],
                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                           }]  # Missing parameters

grid_params_cv_mlpc = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                        "cv__binary": [True, False],
                        "cv__stop_words": [[".", "...", "!", "?"], None],
                        "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                        "mlpc__hidden_layer_sizes": [(50, 50), (100,)],
                        "mlpc__activation": ['tanh', 'relu'],
                        'mlpc__alpha': [0.0001, 0.05],
                        'mlpc__learning_rate': ['constant', 'adaptive']}]

grid_params_tfidf_mlpc = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                           "tfidf__binary": [True, False],
                           "tfidf__stop_words": [[".", "...", "!", "?"], None],
                           "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                           "mlpc__hidden_layer_sizes": [(50, 50), (100,)],
                           "mlpc__activation": ['tanh', 'relu'],
                           'mlpc__alpha': [0.0001, 0.05],
                           'mlpc__learning_rate': ['constant', 'adaptive']}]

grid_params_cv_pac = [{"cv__max_df": np.arange(0.8, 1.05, 0.05),
                       "cv__binary": [True, False],
                       "cv__stop_words": [[".", "...", "!", "?"], None],
                       "cv__ngram_range": [(1, 1), (1, 2), (1, 3)],
                       "pac__early_stopping": [False, True],
                       "pac__warm_start": [False, True]}]

grid_params_tfidf_pac = [{"tfidf__max_df": np.arange(0.8, 1.05, 0.05),
                          "tfidf__binary": [True, False],
                          "tfidf__stop_words": [[".", "...", "!", "?"], None],
                          "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                          "pac__early_stopping": [False, True],
                          "pac__warm_start": [False, True]}]

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

gs_cv_knc = GridSearchCV(estimator=pipe_cv_knc,
                         param_grid=grid_params_cv_knc,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_knc = GridSearchCV(estimator=pipe_tfidf_knc,
                            param_grid=grid_params_tfidf_knc,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

gs_cv_log = GridSearchCV(estimator=pipe_cv_log,
                         param_grid=grid_params_cv_log,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_log = GridSearchCV(estimator=pipe_tfidf_log,
                            param_grid=grid_params_tfidf_log,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

gs_cv_sgd = GridSearchCV(estimator=pipe_cv_sgd,
                         param_grid=grid_params_cv_sgd,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_sgd = GridSearchCV(estimator=pipe_tfidf_sgd,
                            param_grid=grid_params_tfidf_sgd,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

gs_cv_rfc = GridSearchCV(estimator=pipe_cv_rfc,
                         param_grid=grid_params_cv_rfc,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_rfc = GridSearchCV(estimator=pipe_tfidf_rfc,
                            param_grid=grid_params_tfidf_rfc,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

gs_cv_lsvc = GridSearchCV(estimator=pipe_cv_lsvc,
                          param_grid=grid_params_cv_lsvc,
                          scoring=scoring,
                          cv=cv,
                          n_jobs=jobs)

gs_tfidf_lsvc = GridSearchCV(estimator=pipe_tfidf_lsvc,
                             param_grid=grid_params_tfidf_lsvc,
                             scoring=scoring,
                             cv=cv,
                             n_jobs=jobs)

gs_cv_mlpc = GridSearchCV(estimator=pipe_cv_mlpc,
                          param_grid=grid_params_cv_mlpc,
                          scoring=scoring,
                          cv=cv,
                          n_jobs=jobs)

gs_tfidf_mlpc = GridSearchCV(estimator=pipe_tfidf_mlpc,
                             param_grid=grid_params_tfidf_mlpc,
                             scoring=scoring,
                             cv=cv,
                             n_jobs=jobs)

gs_cv_pac = GridSearchCV(estimator=pipe_cv_pac,
                         param_grid=grid_params_cv_pac,
                         scoring=scoring,
                         cv=cv,
                         n_jobs=jobs)

gs_tfidf_pac = GridSearchCV(estimator=pipe_tfidf_pac,
                            param_grid=grid_params_tfidf_pac,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=jobs)

# List of pipelines for ease of iteration
grids = [gs_cv_cnb, gs_tfidf_cnb, gs_cv_knn, gs_tfidf_knn, gs_cv_knc, gs_tfidf_knc, gs_cv_log, gs_tfidf_log,
         gs_cv_rfc, gs_tfidf_rfc, gs_cv_sgd, gs_tfidf_sgd, gs_cv_lsvc, gs_tfidf_lsvc, gs_cv_mlpc, gs_tfidf_mlpc,
         gs_cv_pac, gs_tfidf_pac]

# Dictionary of pipelines and classifier types for ease of reference
grid_labels = ["cv_cnb", "tfidf_cnb", "cv_knn", "tfidf_knn", "cv_knc", "tfidf_knc", "cv_log", "tfidf_log",
               "cv_rfc", "tfidf_rfc", "cv_sgd", "tfidf_sgd", "cv_lsvc", "tfidf_lsvc", "cv_mlpc", "tfidf_mlpc",
               "cv_pac", "tfidf_pac"]

# Model Selection - Running Grid Searches
model_selection(grids, X_train, y_train, X_test, y_test, grid_labels)

# Load pickle files with fitted models
gs_cv_cnb = load("./outputs/Pipeline_cv_cnb.pkl")
gs_tfidf_cnb = load("./outputs/Pipeline_tfidf_cnb.pkl")
gs_cv_knn = load("./outputs/Pipeline_cv_knn.pkl")
gs_tfidf_knn = load("./outputs/Pipeline_tfidf_knn.pkl")
gs_cv_knc = load("./outputs/Pipeline_cv_knc.pkl")
gs_tfid_knc = load("./outputs/Pipeline_tfidf_knc.pkl")
gs_cv_log = load("./outputs/Pipeline_cv_log.pkl")
gs_tfidf_log = load("./outputs/Pipeline_tfidf_log.pkl")
gs_cv_rfc = load("./outputs/Pipeline_cv_rfc.pkl")
gs_tfidf_rfc = load("./outputs/Pipeline_tfidf_rfc.pkl")
gs_cv_sgd = load("./outputs/Pipeline_cv_sgd.pkl")
gs_tfidf_sgd = load("./outputs/Pipeline_tfidf_sgd.pkl")
gs_cv_lsvc = load("./outputs/Pipeline_cv_lsvc.pkl")
gs_tfidf_lsvc = load("./outputs/Pipeline_tfidf_lsvc.pkl")
gs_cv_mlpc = load("./outputs/Pipeline_cv_mlpc.pkl")
gs_tfidf_mlpc = load("./outputs/Pipeline_tfidf_mlpc.pkl")
gs_cv_pac = load("./outputs/Pipeline_cv_pac.pkl")
gs_tfidf_pac = load("./outputs/Pipeline_tfidf_pac.pkl")

# Ensemble

# AdaBoost
clf = AdaBoostClassifier(base_estimator=[gs_cv_knc, gs_tfidf_knn], n_estimators=100, algorithm='SAMME', random_state=15)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Stacking classifier
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=[gs_cv_knc, gs_tfidf_knn],
                         final_estimator=LogisticRegression(class_weight='balanced', multi_class='multinomial'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#('knn', gs_tfidf_knn), ('knc', gs_cv_knc)

from mlxtend.classifier import StackingCVClassifier
sclf = StackingCVClassifier(classifiers=[gs_cv_knc, gs_tfidf_knn],
                            meta_classifier=LogisticRegression(class_weight='balanced', multi_class='multinomial'),
                            random_state=15)
sclf.fit(X_train, y_train)
y_pred = sclf.predict(y_test)

models_list = [gs_cv_cnb,gs_tfidf_cnb,gs_cv_knn, gs_tfidf_knn,gs_cv_log,gs_tfidf_log,gs_cv_rfc,gs_tfidf_rfc,gs_cv_knc,gs_tfid_knc]
models_labels = ['cv_cnb', 'tfidf_cnb','cv_knn','tfidf_knn','cv_log','tfidf_log','cv_rfc','tfidf_rfc','cv_knc','tfidf_knc']

models_comb = list(itertools.combinations(models_list, 4))
labels_comb = list(itertools.combinations(models_labels, 4))


score = []
for i in models_comb:
    ensemble = EnsembleVoteClassifier(clfs=list(i),
                                      voting='hard', refit=False)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score.append(recall_score(y_test, y_pred, average='macro'))

    results = (dict(sorted(zip(score,labels_comb), reverse = True)[:3]))

print('The top 3 models are: ' , results, sep = '\n')


# sorted(zip(score, models_labels), reverse=True)[:2])
#
# ensemble = EnsembleVoteClassifier(clfs=[gs_tfidf_knn,gs_cv_log,gs_cv_knc],
#                                   voting='hard', refit=False)
#
# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)
# print('The recall of the (tfidf_cnb, cv_knn) is: ', recall_score(y_test, y_pred, average='macro'))


# Model Assessment
# ----------------------------------------------------------------------------------------------------------------------
# See reference:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
labels = {"tfidf": "TF-IDF",
          "cv": "CountVectorizer",
          "cnb": "NaiveBayes",
          "knn": "KNearestNeighbors",
          "log": "LogisticRegression",
          "rfc": "RandomForest",
          "sgd": "StocasticGradientDescent",
          "lsvc": "LinearSupportVector",
          "mlpc": "MultiLayerPerceptron",
          "pac": "PassiveAgressive"}
xlsx_path = './outputs/Pipelines.xlsx'

# Comparing Test Error Score between Pipelines
model_assessment_vis(xlsx_path, labels)

y_pred = gs_tfidf_knn.predict(X_test)

classification_report(y_test, y_pred, target_names=list(np.unique(y_test)))

# plot confusion matrix
plot_cm(confusion_matrix(y_test, y_pred), np.unique(y_test))

# evaluation_metrics = pd.DataFrame(classification_report(y_test, y_pred, target_names=list(np.unique(y_test)),
#                                                         output_dict=True))
# save_excel(evaluation_metrics, 'NGRAM13_KNN5')  # save the results in a excel file

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
