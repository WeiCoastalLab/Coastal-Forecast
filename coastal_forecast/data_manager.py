# Created by Andrew Davison
# Used to scrape and clean long term training_data and short term data for use in model training and predictions
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import requests


def hello():
    return "Hello from Component data_manager"


def get_request(url: str) -> list:
    """
    Sends request for data from url.

    :param url: url of data source.
    :return: list of scraped data for further processing.
    """
    print(f'Fetching data from {url}...')
    with requests.Session() as s:
        download = s.get(url)
        decoded_content = download.content.decode('utf-8')
        cr = list(csv.reader(decoded_content.splitlines(), delimiter=','))
        data_list = []
        for item in cr:
            list_item = [words for segments in item for words in segments.split()]
            data_list.append(list_item)

    return data_list


def fetch_lt_data(station_id: str) -> None:
    """
    Fetches long term data from specific NOAA station and prepares data for ML model training.

    :param station_id: string representation of NOAA station identifier, default is Station 41013, Frying Pan Shoals.
    :return: None
    """
    # create a list of the past five available years
    years = [datetime.today().year - x - 1 for x in range(5)]

    # unpack returned results from helper function
    year_one, year_two, year_three, year_four, year_five = fetch_lt_data_helper(station_id, years)

    # set returned results to dataframes list and combine
    dataframes = [year_one, year_two, year_three, year_four, year_five]
    lt_data = pd.concat(dataframes)

    # clean the dataset and save for model training
    save_lt_data(lt_data, station_id)


def fetch_lt_data_helper(station_id: str, years: list) -> pd.DataFrame:
    """
    Helper function to scrape historical NOAA station data from a list of past years.

    :param station_id: string representation of NOAA station identifier.
    :param years: list of past target years to scrape data for.
    :return: data from single year for further processing
    """
    # initialize urls with station_id and requested years
    url1 = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{years[0]}' \
           f'.txt.gz&dir=data/historical/stdmet/'
    url2 = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{years[1]}' \
           f'.txt.gz&dir=data/historical/stdmet/'
    url3 = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{years[2]}' \
           f'.txt.gz&dir=data/historical/stdmet/'
    url4 = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{years[3]}' \
           f'.txt.gz&dir=data/historical/stdmet/'
    url5 = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{years[4]}' \
           f'.txt.gz&dir=data/historical/stdmet/'

    # identify columns for dataframe
    column_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP',
                    'WTMP', 'DEWP', 'VIS', 'TIDE']

    # loop through each url and pull data
    for url in [url1, url2, url3, url4, url5]:
        data_list = get_request(url)
        # set pulled data as dataframe and return
        data = pd.DataFrame(data_list[2:], columns=column_names)

        yield data


# refactoring needed for new clean short term data with the proper columns for model
def fetch_data(station_id: str) -> pd.DataFrame:
    """
    Fetches short term data from specific NOAA station and prepares data for ML model.

    :param station_id: NOAA station to fetch data from.
    :return: pandas DataFrame of prepared data.
    """
    # set url for the requested station data
    url = f'https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt'

    # identify columns for dataframe
    column_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP',
                    'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']

    # pull data from url
    data_list = get_request(url)

    # set pulled data as a dataframe and return cleaned data
    data = pd.DataFrame(data_list[2:], columns=column_names)

    return clean_data(data)


def save_lt_data(data: pd.DataFrame, station_id: str) -> None:
    """
    Cleans and saves long term data to csv file for later model training.

    :param data: dataframe of long term data.
    :param station_id: string of NOAA station identification number.
    :return: None
    """
    column_names_kept = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                         'ATMP', 'WTMP', 'DEWP']
    column_drops = ['MM', 'DD', 'YY', 'hh', 'mm', 'Datetime']
    columns = ['Time', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP']

    # preprocesses data with columns from columns list
    data = preprocess_data(data, column_names_kept, column_drops)

    # set numerical values to floats
    for col in columns[1:]:
        data.loc[:, col] = data.loc[:, col].astype(float)

    # replace bad data with NaN
    data = data.replace({'WDIR': 999, 'WSPD': 99, 'GST': 99,
                         'WVHT': 99, 'APD': 99, 'DPD': 99, 'MWD': 999,
                         'PRES': 9999, 'ATMP': 999, 'WTMP': 999, 'DEWP': 999}, np.nan)

    # reduces dataframe to good values of wave height
    data = data[data['WVHT'].notna()]

    # replace all NaN values with median value of the column
    null_columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'APD', 'DPD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP']
    for col in null_columns:
        data.loc[:, col].fillna(data[col].median(), inplace=True)  # method='pad', inplace=True)

    # resample data by every hour and reset indexing
    data_sampled = data.set_index('Time').resample('60T').pad()
    data_sampled.reset_index(inplace=True)

    # mask first record from dataset and reset dataframe
    mask = data_sampled['Time'] >= f'{datetime.today().year - 5}-01-01 01:00:00'
    data_sampled = data_sampled.loc[mask]
    data_sampled.reset_index(inplace=True, drop=True)

    # rearrange columns
    column_finals = ['Time', 'WDIR', 'WSPD', 'GST', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'WVHT', 'DPD', 'APD', 'MWD']
    data_sampled = data_sampled[column_finals]
    #
    # print(data_sampled.info())
    # print(data_sampled['DEWP'].head(20))
    # quit()
    # save data to csv file without indexes
    data_sampled.to_csv(f'../training_data/{station_id}_lt_clean.csv', index=False)


# used for cleaning short term data, might need to be refactored
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess cleaning of data for use in a trained ML model.

    :param data: dataframe of data to be cleaned
    :return: clean preprocessed data.
    """
    # initialize columns for cleaning steps
    column_names_kept = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'PRES', 'WVHT', 'APD', 'MWD']
    column_drops = ['MM', 'DD', 'YY', 'hh', 'mm', 'Datetime']
    column_finals = ['Time', 'WDIR', 'WSPD', 'PRES', 'WVHT', 'APD', 'MWD']

    data = preprocess_data(data, column_names_kept, column_drops)

    data.drop(data[data.loc[:, 'WVHT'] == 'MM'].index, inplace=True)
    data = data.replace('MM', np.nan)

    # reset indexing, 0-n and set dataframe to only the final columns needed
    data.reset_index(inplace=True)
    data = data.loc[:, column_finals]

    # set all measured values to floating point numbers and replace NaN values with previous value
    for col in column_finals[1:]:
        data.loc[:, col] = data.loc[:, col].astype(float)
        data.loc[:, col].fillna(method='pad', inplace=True)

    # finalize dataset to be indexed by time at every hour and reset indexes
    data_sampled = data.set_index('Time').resample('60T').pad()
    data_sampled.reset_index(inplace=True)

    return data_sampled


# preprocesses dataframe to index by timestamp, should be good
def preprocess_data(data: pd.DataFrame, columns_kept: list, columns_drop: list) -> pd.DataFrame:
    """
    Preprocessing data step to consolidate columns to a time column and sort and reindex dataframe.

    :param data: dataframe to be preprocessed
    :param columns_kept: list of columns to keep in the dataframe
    :param columns_drop: list of columns to drop in the dataframe
    :return: preprocessed dataframe
    """
    # keep required columns
    data = data.loc[:, columns_kept]

    # set date/time columns to strings
    for name in ['YY', 'MM', 'DD', 'hh', 'mm']:
        data.loc[:, name] = data.loc[:, name].astype(str).str.zfill(2)

    # create Datetime column for data and format to year, month, day, hours, minutes
    data.loc[:, 'Datetime'] = data.loc[:, 'YY'] + data.loc[:, 'MM'] + data.loc[:, 'DD'] + data.loc[:, 'hh'] + data.loc[
                                                                                                              :, 'mm']
    data.loc[:, 'Time'] = pd.to_datetime(data.loc[:, 'Datetime'].astype(str), format='%Y%m%d%H%M')

    # drop not required columns and move Time column to first position followed by all other columns
    data.drop(columns_drop, inplace=True, axis=1)
    data = data[['Time'] + [col for col in data.columns if col != 'Time']]

    # sort Time column in ascending order, earliest time to present
    data = data.sort_values('Time')

    return data


if __name__ == "__main__":
    fetch_lt_data('41008')
    # fetch_data('41013')
    # for station in ['41008', '41009', '41013', '44013']:
    #     fetch_lt_data(station)
