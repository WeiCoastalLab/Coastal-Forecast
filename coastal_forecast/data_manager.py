# Created by Andrew Davison
# Used to scrape and clean the training_data for use in model
import csv

import numpy as np
import pandas as pd
import requests


def hello():
    return "Hello from Component data_manager"


def fetch_data(station_id: str) -> pd.DataFrame:
    """
    Fetches short term data from specific NOAA station and prepares data for ML model.

    Preparation of data sets data to one hour increments from earliest data time to
    present with the format of[Time, WDIR, WSPD, PRES, WVHT, APD, MWD], where the
    columns are the following:
    \nTime: Datetime of the observation (Year, month, day, hour, minute) in UTC.
    \nWDIR: Wind direction (degrees).
    \nWSPD: Wind speed (m/s).
    \nPRES: Barometric pressure (hPa).
    \nWVHT: Wave height (target value for ML model) in meters.
    \nAPD: Average wave period (target value for ML model) in seconds.
    \nMWD: Mean wave direction (target value for ML model) in degrees.

    :param station_id: NOAA station to fetch data from.
    :return: pandas DataFrame of prepared data.
    """
    # set url for the requested station data
    url = f'https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt'
    print(f'Fetching data from: {url}')

    # identify columns for processing
    column_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP',
                    'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']
    column_names_kept = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'PRES', 'WVHT', 'APD', 'MWD']
    column_drops = ['MM', 'DD', 'YY', 'hh', 'mm', 'Datetime']
    column_finals = ['Time', 'WDIR', 'WSPD', 'PRES', 'WVHT', 'APD', 'MWD']

    # begin request session
    with requests.Session() as s:
        download = s.get(url)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')

        # create list for setting decoded data to dataframe
        my_list, new_list = list(cr), []
        for item in my_list:
            my_list_item = [words for segments in item for words in segments.split()]
            new_list.append(my_list_item)

    # create dataframe of data
    realtime_data = pd.DataFrame(new_list[2:], columns=column_names)

    # keep required columns
    data = realtime_data.loc[:, column_names_kept]

    # set date/time columns to strings
    for name in ['YY', 'MM', 'DD', 'hh', 'mm']:
        data.loc[:, name] = data.loc[:, name].astype(str).str.zfill(2)

    # create Datetime column for data and format to year, month, day, hours, minutes
    data.loc[:, 'Datetime'] = data.loc[:, 'YY']+data.loc[:, 'MM']+data.loc[:, 'DD']+data.loc[:, 'hh']+data.loc[:, 'mm']
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
#     fetch_data('41013')
