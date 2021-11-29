# Created by Andrew Davison
from flask import render_template
from datetime import datetime, timedelta
from coastal_forecast import app, prediction_manager as pm

stations = ['41008', '41009', '41013', '44013']
pred_time = dict.fromkeys(stations, datetime.utcnow().strftime('%m/%d/%Y %H:%M'))


# home page
@app.route("/")
@app.route("/41013")
def home():
    return render_template('41013.html', title='41013', current=pred_time['41013'],
                           next=(datetime.strptime(pred_time['41013'], '%m/%d/%Y %H:%M')
                                 + timedelta(hours=6)).strftime('%m/%d/%Y %H:%M'))


@app.route("/41009")
def canaveral():
    return render_template('41009.html', title='41009', current=pred_time['41009'],
                           next=(datetime.strptime(pred_time['41009'], '%m/%d/%Y %H:%M')
                                 + timedelta(hours=6)).strftime('%m/%d/%Y %H:%M'))


@app.route("/44013")
def boston():
    return render_template('44013.html', title='44013', current=pred_time['44013'],
                           next=(datetime.strptime(pred_time['44013'], '%m/%d/%Y %H:%M')
                                 + timedelta(hours=6)).strftime('%m/%d/%Y %H:%M'))


@app.route("/41008")
def grays():
    return render_template('41008.html', title='41008', current=pred_time['41008'],
                           next=(datetime.strptime(pred_time['41008'], '%m/%d/%Y %H:%M')
                                 + timedelta(hours=6)).strftime('%m/%d/%Y %H:%M'))


def scheduled_task():
    for station in stations:
        pm.get_prediction(station, 9, 3)
        pred_time[station] = datetime.utcnow().strftime('%m/%d/%Y %H:%M')
