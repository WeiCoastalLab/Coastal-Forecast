# Created by Andrew Davison
from flask import render_template
from coastal_forecast import app
from PIL import Image


def process_params(station_id: str) -> tuple[str, str]:
    """
    Processes and returns current prediction time, next prediction time,
    and plot image location for rendering of template
    :param station_id: identification number of station to process
    :return: tuple of current prediction time, next prediction time, and relative plot image path
    """
    with Image.open(f'coastal_forecast/static/{station_id}_system_prediction.png') as img:
        img.load()
        current_time = img.info['Time Created']
        next_time = img.info['Time Offset']
    print(current_time, next_time)
    return current_time, next_time


# home page
@app.route("/")
@app.route("/41013")
def home():
    title = '41013'
    current_time, next_time = process_params(title)
    return render_template('41013.html', title=title, current=current_time, next=next_time)


@app.route("/41009")
def canaveral():
    title = '41009'
    current_time, next_time = process_params(title)
    return render_template('41009.html', title=title, current=current_time, next=next_time)


@app.route("/44013")
def boston():
    title = '44013'
    current_time, next_time = process_params(title)
    return render_template('44013.html', title=title, current=current_time, next=next_time)


@app.route("/41008")
def grays():
    title = '41008'
    current_time, next_time = process_params(title)
    return render_template('41008.html', title=title, current=current_time, next=next_time)
