import zipfile as zp
import pandas as pd
import itertools
import re
import nltk
import string

# nltk.download('rslp')
# nltk.download('punkt')


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




def sample_excerpts(data):

    def extract_excerpts(row):
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
        return groups

    dataframe = []
    for _, row in data.iterrows():
        dataframe += extract_excerpts(row)
    return pd.DataFrame(dataframe, columns=["book_id", "author", "text"])


# Sampling the texts
samples = sample_excerpts(train_df)
samples["word_count"] = samples["text"].apply(lambda x: len(nltk.word_tokenize(x)))  # creating number of words column
# samples["word_count"].describe()

# Updating train_df
train_df = samples

# TODO: clean texts before sampling to reduce word_count variance.
# TODO: Join some adjacent texts belonging to same book to obtain 1000 word excerpts
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
# TODO: exclude the text corresponding to the beginning of the book
# TODO: tokenize week days (e.g. sexta-feira, quinta-feira, sabado, ...), cidades portuguesas, datas, ...
# TODO: include punctuation as features (atenção às reticências)
# TODO: Perform POS filtering (Viterbi algorithm)
# 1. Normalization (dates), 2. Lowercasing, 3. Tokenization (compounds, punctuation), 4. Stop-words, 5. Stemming and Lemmatisation, 6. POS filtering
# TODO: build baseline with simple KNN (slides and labs)
# a = train.loc[:, "text"].apply(lambda x: re.findall("\n{3,5}([\s\S]*)", x))

# Reference: http://www.nltk.org/howto/portuguese_en.html
stopw = set(nltk.corpus.stopwords.words('portuguese'))
punct = set(string.punctuation) # Even if we end up not removing punctuation we need to watch out with some symbols (e.g. *, %, >)
stemmer = nltk.stem.RSLPStemmer()
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
def price_token(text):
    token = re.sub('\S+\$\d+( réis)*','#Price',text)
    return token

def date_token(text):
    token = re.sub('(\d{1,2}\D\d{1,2}\D\d{2,4})|(\d{4}\D\d{1,2}\D\d{1,2})|(\d{1,2} de [a-zA-Z]+ de \d{2,4})', "#DATE", text)
    return token

# df['texts'].iloc[18] = token(df['texts'].iloc[18])

a = df['texts'].apply(lambda x: re.findall('\$',x))

# Bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

cv = CountVectorizer(max_df=0.9, binary=True)

X = cv.fit_transform(train["text"])
y = np.array(train["author"])

# K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

modelknn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                         metric='cosine', metric_params=None, n_jobs=1)
modelknn.fit(X,y)

test4 = cv.transform(test.text)

predict = modelknn.predict(test4)
predict
