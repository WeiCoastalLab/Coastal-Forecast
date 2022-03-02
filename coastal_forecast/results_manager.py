# Created by Andrew Davison
from datetime import datetime

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

dpi = 300
matplotlib.rc("savefig", dpi=dpi)


def plot_results(results: pd.DataFrame, station_id: str, filepath: str, n_inputs: int, n_outputs: int) -> None:
    """
    Creates and saves a comparison plot of ground truth and predictions over time
    :param results: dataframe of observed and predicted values
    :param station_id: string of NOAA station identification number
    :param filepath: string of filepath, with filename, to save comparison plots
    :param n_inputs: number of inputs used in model
    :param n_outputs: number of outputs predicted from model
    :return: None
    """
    print(f"Plotting station {station_id} results...\n")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), facecolor='w', edgecolor='k', sharex='all')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)
    results.plot(x='Time', y='WVHT', color='blue', label='NOAA Measurement', ax=ax1)
    results.plot(x='Time', y='WVHT_LSTM', color='red', label='LSTM Prediction', ax=ax1)
    ax1.legend();  # noqa
    ax1.set_xlabel(' ')
    ax1.set_ylabel('WVHT (m)')
    title_wvht = f'(a) Significant wave height (WVHT) comparison at St# {station_id} for {n_outputs}-hour ' \
                 f'prediction with {n_inputs}-hour input'
    ax1.set_title(title_wvht)  # noqa
    ax1.grid(b=True, which='major', color='#666666', linestyle='-')
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax1.get_legend().remove()

    results.plot(x='Time', y='APD', color='blue', label='NOAA Measurement', ax=ax2, legend=False)
    results.plot(x='Time', y='APD_LSTM', color='red', label='LSTM Prediction', ax=ax2, legend=False)
    ax2.legend();  # noqa
    ax2.set_xlabel(' ')
    ax2.set_ylabel('APD (s)')
    title_apd = f'(b) Averaged wave period (APD) comparison at St# {station_id} for {n_outputs}' \
                f'-hour prediction with {n_inputs}-hour input'
    ax2.set_title(title_apd)  # noqa
    ax2.grid(b=True, which='major', color='#666666', linestyle='-')
    ax2.minorticks_on()
    ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax2.get_legend().remove()

    results.plot(x='Time', y='MWD', color='blue', label='NOAA Measurement', ax=ax3)
    results.plot(x='Time', y='MWD_LSTM', color='red', label='LSTM Prediction', ax=ax3)
    ax3.legend();  # noqa
    ax3.set_xlabel('Time (UTC)')
    title_mwd = f'(c) Mean Wave Direction (MWD) comparison at St# {station_id} for {n_outputs}' \
                f'-hour prediction with {n_inputs}-hour input'
    ax3.set_title(title_mwd)
    ax3.set_ylim([0, 360])
    ax3.set_ylabel('MWD (degree)')
    ax3.grid(b=True, which='major', color='#666666', linestyle='-')
    ax3.minorticks_on()
    ax3.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax3.legend(loc="lower center", ncol=2)

    meta = {'Title': filepath,
            'Station': station_id,
            'Time': datetime.utcnow().strftime('%m/%d/%Y %H:%M')}

    plt.savefig(filepath, bbox_inches='tight', metadata=meta)
    plt.show()
