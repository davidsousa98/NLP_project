import os
import zipfile as zp
import io
import requests
import unicodedata
from openpyxl import load_workbook
# from tqdm import tqdm_notebook as tqdm
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from string import punctuation as punct
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

import matplotlib
matplotlib.use('TkAgg')

def get_files_zip():
    """
    Function that lists files inside the nlp_data.zip folder.

    Returns:
        - list of paths to files.
    """
    with zp.ZipFile("./nlp_data.zip") as myzip:
        list = myzip.namelist()
    return list


def read_txt_zip(files):
    """
    Function that reads files inside the nlp_data.zip folder.

    :param files: list of strings with paths to files.

    Returns:
        - list of strings with content of corresponding files.
    """
    df_list = []
    for file in files:
        with zp.ZipFile("./nlp_data.zip") as myzip:
            with myzip.open(file) as myfile:
                df_list.append(myfile.read().decode("utf-8"))
    return df_list


def clean(text_list, punctuation, stoppers, lemmatize=None, stemmer=None):
    """
    Function that a receives a list of strings and preprocesses it.

    :param text_list: list of strings.
    :param punctuation: list of punctuation to exclude.
    :param stoppers: list of punctuation that defines the end of an excerpt.
    :param lemmatize: lemmatizer to apply if provided.
    :param stemmer: stemmer to apply if provided.

    Returns:
        - list of cleaned strings.
    """
    updates = []
    for j in range(len(text_list)):

        text = text_list[j]

        # LOWERCASING
        text = text.lower()

        # REMOVING # SIGN (don't remove with punctuation as it will destroy tokens)
        text = text.replace("#", " ")

        # TOKENIZATION
        text = re.sub('\S+\$\d+( réis)*', ' #PRICE ', text)  # price token
        text = re.sub('(\d{1,2}\D\d{1,2}\D\d{2,4})|(\d{4}\D\d{1,2}\D\d{1,2})|(\d{1,2} de [a-zA-Z]+ de \d{2,4})',
                      " #DATE ", text)  # date token
        text = re.sub('[0-9]+', ' #NUMBER ', text)  # number token
        # cities token
        url = "https://simplemaps.com/static/data/country-cities/pt/pt.csv"
        cities = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')))["city"].to_list()
        cities.append("lisboa")
        cities = list(map(lambda x: x.lower(), cities))
        for i in cities:
            if i in text:
                text = text.replace(i, ' #CITY ')
        # week days token
        week_days = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado',
                     'domingo']
        for i in week_days:
            if i in text:
                text = text.replace(i, ' #WDAY ')

        # REMOVE PUNCTUATION, SEPARATE KEEP PUNCTUATION WITH SPACES AND KEEP TOKENS #
        def repl(matchobj):
            keep_punct = list(set(punct) - (set(punctuation) - set(stoppers)) - set("#"))
            if matchobj.group(0) in keep_punct:
                return " {} ".format(matchobj.group(0))
            elif matchobj.group(0) == "#":
                return '#'
            else:
                return ' '
        text = re.sub("([^a-zA-Z0-9])", repl, text)

        # REMOVE STOPWORDS
        stop = set(stopwords.words('portuguese'))
        text = " ".join(" " if word in stop else word for word in text.split())

        # REMOVE HTML TAGS
        text = BeautifulSoup(text, features="lxml").get_text()

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
    """
    Function that updates a DataFrame's "text" column inplace.

    :param dataframe: DataFrame to update.
    :param list_updated: list to replace the "text" column.
    """
    dataframe.update(pd.Series(list_updated, name="text"))  # Modify in place using non-NA values from another DataFrame


def sample_excerpts(dataframe, stoppers):
    """
    Function that receives a DataFrame and creates subsets of ~500 words of each
    string in its "text" column.

    :param dataframe: DataFrame to create subsets.
    :param stoppers: list of punctuation that defines the end of an excerpt.

    Returns:
        - DataFrame object with a new text column that contains the subsets.
    """
    data = []
    for _, row in dataframe.iterrows():
        words = word_tokenize(row["text"])
        groups = []
        group = []
        for word in words:
            group.append(word)
            if (len(group) >= 485) & (group[-1] in stoppers):
                groups.append((row["book_id"], row["author"], " ".join(group)))
                group = []
            elif len(group) >= 600:  # Dealing with lack of punctuation
                groups.append((row["book_id"], row["author"], " ".join(group)))
                group = []
        data += groups
    return pd.DataFrame(data, columns=["book_id", "author", "text"])


def plot_cm(confusion_matrix: np.array, class_names: list):
    """
    Function that creates a confusion matrix plot using the Wikipedia convention for the axis.

    :param confusion_matrix: confusion matrix that will be plotted
    :param class_names: labels of the classes

    Returns:
        - Plot of the Confusion Matrix
    """

    fig, ax = plt.subplots()
    plt.imshow(confusion_matrix, cmap=plt.get_cmap('cividis'))
    plt.colorbar()

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    plt.ylabel('Targets', fontweight='bold')
    plt.xlabel('Predictions', fontweight='bold')
    plt.ylim(top=len(class_names) - 0.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-0.5)  # adjust the bottom leaving top unchanged
    plt.tight_layout()
    return plt.show()


def word_counter(text_list, top=None):
    """
    Function that creates the ranking of word frequencies in every string in a
    list of strings.

    :param text_list: list of strings.
    :param top: integer that indicates how many top ranks to display.

    Returns:
        - Series with the value counts of each top word.
    """
    split = [i.split() for i in text_list]
    words_in_df = list(itertools.chain.from_iterable(split))  # merges lists in list
    freq = pd.Series(words_in_df).value_counts()
    if top:
        return freq.head(top)
    else:
        return freq


def save_excel(dataframe, sheetname, filename="metrics"):
    """
    Function that saves the evaluation metrics into a excel file.

    :param dataframe: dataframe that contains the metrics.
    :param sheetname: name of the sheet containing the parameterization.
    :param filename: specifies the name of the xlsx file.

    """
    if not os.path.isfile("./outputs/{}.xlsx".format(filename)):
        mode = 'w'
    else:
        mode = 'a'
    writer = pd.ExcelWriter('./outputs/{}.xlsx'.format(filename), engine='openpyxl', mode=mode)
    dataframe.to_excel(writer, sheet_name=sheetname)
    writer.save()


def get_top_n_grams(corpus, top_k, n):
    """
    Function that receives a list of documents (corpus) extracts
        the top k most frequent n-grams for that corpus and plots the frequencies in a bar plot.

    :param corpus: list of texts
    :param top_k: int with the number of n-grams that we want to extract
    :param n: n gram type to be considered
             (if n=1 extracts unigrams, if n=2 extracts bigrams, ...)

    :return: Returns a sorted dataframe in which the first column
        contains the extracted ngrams and the second column contains
        the respective counts
    """
    vec = CountVectorizer(ngram_range=(n, n), max_features=2000).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = []
    for word, idx in vec.vocabulary_.items():
        words_freq.append((word, sum_words[0, idx]))

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:top_k])
    top_df.columns = ["Ngram", "Freq"]

    x_labels = top_df["Ngram"][:30]
    y_pos = np.arange(len(x_labels))
    values = top_df["Freq"][:30]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel('Frequencies')
    plt.title('Words')
    plt.xticks(rotation=90)
    plt.show()
    return top_df


def model_selection(grids, X_train, y_train, X_test, y_test, grid_labels):
    """
        Function receives a list of GridSearch objects and fits them to the data. Returns the best parameters in each
        grid together with the respective macro recall score. Writes and excel file with information on the best
        parameters for each Pipeline and exports the Pipelines to pickle files.

        :param grids: list of GridSearch objects
        :param X_train: X train
        :param y_train: target train
        :param X_test: X test
        :param y_test: target test
        :param grid_labels: list with a label for each GridSearch object

        """
    print('Performing model optimizations...')
    for gs, label in zip(grids, grid_labels):
        print('\nEstimator: %s' % label)
        if os.path.isfile("./outputs/Pipelines.xlsx"):
            wb = load_workbook("./outputs/Pipelines.xlsx", read_only=True)
            if label in wb.sheetnames:
                print('Grid already fitted. Continue to next grid.')
                continue
        # Fit grid search
        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data score
        print('Best training score: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # Test data score of model with best params
        print('Test set score for best params: %.3f ' % recall_score(y_test, y_pred, average="macro"))
        # Save Pipeline name and best parameters to excel
        grid_results = pd.DataFrame(gs.cv_results_)
        grid_results["best_index"] = pd.Series(np.zeros(grid_results.shape[0]))
        grid_results.loc[gs.best_index_, "best_index"] = 1
        save_excel(grid_results, label, "Pipelines")
        # Save fitted grid to pickle file
        dump(gs, "./outputs/Pipeline_{}.pkl".format(label), compress=1)
    print("Model Selection process finished")

