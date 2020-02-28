
def get_files_zip():
    with zp.ZipFile("./data/nlp_data.zip") as myzip:
        list = myzip.namelist()
    return list


def read_csv_zip(files):
    df_list = []
    for file in files:
        with zp.ZipFile("./data/airbnb_data.zip") as myzip:
            with myzip.open(file) as myfile:
                df_list.append(pd.read_csv(myfile))
    return df_list


# Reading files from zip without extracting them
# Warning is returned however it's not concerning!
get_files_zip()