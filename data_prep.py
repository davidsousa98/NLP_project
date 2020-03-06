import zipfile as zp
import pandas as pd
import itertools
import re

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

text_names = list(filter(lambda x: ".txt" in x, get_files_zip()))
texts = read_txt_zip(text_names)
df = pd.DataFrame({"names": text_names, "texts": texts})


files = list(filter(lambda x: ".txt" in x, get_files_zip()))
texts = read_txt_zip(files)
df = pd.DataFrame({"file": files, "text": texts})

df["type"] = df["file"].apply(lambda x: re.findall(r"(^[a-z]+)/", x)[0])
train = df.loc[df["type"] == "train", ["file", "text"]].reset_index(drop=True)
train["author"] = train["file"].apply(lambda x: re.findall(r"/([a-zA-Z]+)/", x)[0])
train = train[["file", "author", "text"]]
test = df.loc[df["type"] == "test", ["file", "text"]].reset_index(drop=True)

# TODO: sample the books to create excerpts of 500 or 1000 words for each author
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

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup # useful in dealing with html
import string

stop = set(stopwords.words('portuguese'))
# exclude = set(string.punctuation)
lemma = WordNetLemmatizer('portuguese')
snowball_stemmer = SnowballStemmer('portuguese')
list_of_words = ["consultoria", "consultores", "consulta", "consultava"]
for word in list_of_words:
    print("Lemma of {} is {}".format(word,lemma.lemmatize(word)))
    print("Stem of {} is {}".format(word,snowball_stemmer.stem(word)))
    print("----")

def rm_nonalphanumeric(string):
    re.su

df["text"].apply(rm_nonalphanumeric)


#Clean data
def token(text):
    token = re.sub('\S+\$\d+( réis)*','#Price',text)
    return token

df['texts'].iloc[18] = token(df['texts'].iloc[18])

a = df['texts'].apply(lambda x: re.findall('\$',x))
# b = df['texts'].apply(lambda x: re.findall('\/',x))


