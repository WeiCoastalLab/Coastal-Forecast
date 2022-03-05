# Created by Andrew Davison
from flask import Flask
from whitenoise import WhiteNoise
from apscheduler.schedulers.background import BackgroundScheduler
from coastal_forecast import prediction_manager as pm


def timed_job():
    stations = ['41008', '41009', '41013', '44013']
    for station in stations:
        pm.get_prediction(station, 9, 3)


timed_job()
scheduler = BackgroundScheduler(timezone='UTC', daemon=True)
app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='coastal_forecast/static/')
try:
    print("Starting scheduler...")
    scheduler.start()
except KeyboardInterrupt or SystemExit:
    scheduler.shutdown(wait=False)

from coastal_forecast import routes  # noqa E402
