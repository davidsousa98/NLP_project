import zipfile as zp
import pandas as pd

def get_files_zip():
    with zp.ZipFile("./nlp_data.zip") as myzip:
        list = myzip.namelist()
    return list


def read_text_zip(files):
    df_list = []
    for file in files:
        with zp.ZipFile("./nlp_data.zip") as myzip:
            with myzip.open(file) as myfile:
                df_list.append(open(myfile,'w+'))
    return df_list


# Reading files from zip without extracting them
# Warning is returned however it's not concerning!

get_files_zip()

read_text_zip()