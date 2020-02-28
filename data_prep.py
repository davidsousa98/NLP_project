import zipfile as zp
import pandas as pd

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