import csv
import gzip
import os
import shutil
import glob

import pandas as pd

class_dir = '/Users/sohumberry/Downloads/Gaia/gdr3/Variability/vari_classifier_result/'
summ_dir = '/Users/sohumberry/Downloads/Gaia/gdr3/Variability/vari_summary/'

dirs = [class_dir, summ_dir]


def gz_extract(directory):
    extension = ".gz"
    os.chdir(directory)
    for item in os.listdir(directory):  # loop through items in dir
        if item.endswith(extension):  # check for ".gz" extension
            gz_name = os.path.abspath(item)  # get full path of files
            print(gz_name)
            file_name = (os.path.basename(gz_name)).rsplit('.', 1)[0]  #get file name for file within
            with gzip.open(gz_name, "rb") as f_in, open(file_name, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print('deleting zipped file')
            os.remove(gz_name)  # delete zipped file


def combine_csvs(folder):
    csv_files = glob.glob(folder + '*.{}'.format('csv'))
    print(csv_files)
    # df_append = pd.DataFrame()
    df_concat = pd.concat([pd.read_csv(f, comment='#') for f in csv_files], ignore_index=True)
    #append all files together
    # for file in csv_files:
    #     df_temp = pd.read_csv(file)
    #     df_append = df_append.append(df_temp, ignore_index=True, comment='#')
    print(df_concat)
    df_concat.to_csv(f'{folder.split('/')[-2]}.csv', index=False)


def combine_dataframes(d1, d2, write_to_csv=True):
    groupby = 'source_id'
    d1k = d1.assign(key=d1.groupby(groupby).cumcount())
    d2k = d2.assign(key=d2.groupby(groupby).cumcount())

    d3 = d1k.merge(d2k, on=[groupby, 'key'], how='outer').sort_values(groupby)
    d3 = d3.drop('key', axis=1)
    if write_to_csv:
        d3.to_csv('combined.csv', index=False, na_rep='N/A')
        print('combined data written')
    return d3


def clean_data(data=pd.DataFrame, threshold=0.2):
    classes = ['ECL', 'LPV', 'SOLAR_LIKE', 'DSCT|GDOR|SXPHE', 'AGN', 'RS', 'S', 'RR']
    # init_len = data.shape
    # print(init_len)
    # data = data.dropna(axis=0, how='any')
    # second_len = data.shape
    # print(second_len)
    # if init_len != second_len:
    #     data.to_csv('total_combined_clean.csv')
    data = data.drop(data[data.best_class_name not in classes].index)
    print(data.shape)
    # if second_len != data.shape:
    data.to_csv(f'{threshold}_clean.csv')


def filter_classes(data, threshold):
    allowed_values = ['ECL', 'LPV', 'SOLAR_LIKE', 'DSCT|GDOR|SXPHE', 'AGN', 'RS', 'S', 'RR']
    try:
        data['best_class_name'] = data['best_class_name'].apply(lambda x: 'OTHER' if x not in allowed_values else x)
    except IndexError:
        print('index not found')
    except Exception:
        print('error')
    data.to_csv(f'preprocessed data/{threshold}.csv')


def filter_confidence(filename, threshold, name, filter_na):
    rows = getstuff(filename, threshold)
    # df = df.drop(df[df.score < threshold].index)


def getstuff(filename, criterion):
    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        yield next(datareader)  # yield the header row
        for row in datareader:
            if float(row[-1]) == criterion:
                yield row
        return

# def closest_square_dimensions(x):
#     # Start with the square root of x and round down
#     side = int(math.sqrt(x))
#
#     # Find the closest factor pair
#     for i in range(side, 0, -1):
#         if x % i == 0:
#             return i, x // i