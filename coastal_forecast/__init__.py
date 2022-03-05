# Created by Andrew Davison
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from coastal_forecast import prediction_manager as pm


def timed_job():
    stations = ['41008', '41009', '41013', '44013']
    for station in stations:
        pm.get_prediction(station, 9, 3)


timed_job()
app = Flask(__name__)
scheduler = BackgroundScheduler(timezone='UTC', daemon=True)
scheduler.add_job(timed_job, 'interval', hours=6)
try:
    print("starting scheduler...")
    scheduler.start()
except KeyboardInterrupt or SystemExit:
    scheduler.shutdown(wait=False)

from coastal_forecast import routes  # noqa E402
