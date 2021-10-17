# Created by Andrew Davison
# Used to scrape and clean the training_data for use in model
import csv
import datetime

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


def fetch_lt_data(station_id: str = '41013') -> None:
    """
    Fetches long term data from specific NOAA station and prepares data for ML model training.

    :param station_id: string representation of NOAA station identifier, default is Station 41013, Frying Pan Shoals.
    :return: None
    """
    # create a list of the past five available years
    years = [datetime.datetime.today().year - x - 1 for x in range(5)]

    # unpack returned results from helper function
    year_one, year_two, year_three, year_four, year_five = fetch_lt_data_helper(station_id, years)

    # set returned results to dataframes list and combine
    dataframes = [year_one, year_two, year_three, year_four, year_five]
    lt_data = pd.concat(dataframes)

    # clean the dataset and save for model training
    cleaned_dataset = clean_data(lt_data)
    cleaned_dataset.to_csv(f'../training_data/{station_id}_lt_clean.csv')  # noqa


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

    # keep required columns
    data = data.loc[:, column_names_kept]

    # set date/time columns to strings
    for name in ['YY', 'MM', 'DD', 'hh', 'mm']:
        data.loc[:, name] = data.loc[:, name].astype(str).str.zfill(2)

    # create Datetime column for data and format to year, month, day, hours, minutes
    data.loc[:, 'Datetime'] = data.loc[:, 'YY'] + data.loc[:, 'MM'] + data.loc[:, 'DD'] + data.loc[:, 'hh'] + data.loc[
                                                                                                              :, 'mm']
    data.loc[:, 'Time'] = pd.to_datetime(data.loc[:, 'Datetime'].astype(str), format='%Y%m%d%H%M')

    # drop not required columns and move Time column to first position followed by all other columns
    data.drop(column_drops, inplace=True, axis=1)
    data = data[['Time'] + [col for col in data.columns if col != 'Time']]

    # sort Time column in ascending order, earliest time to present
    data = data.sort_values('Time')

    # drop WVHT columns marked with 'MM', replace all other 'MM' occurances to NaN
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

    # return prepped data
    return data_sampled


# if __name__ == "__main__":
    # fetch_data('41013')
    # fetch_lt_data()
