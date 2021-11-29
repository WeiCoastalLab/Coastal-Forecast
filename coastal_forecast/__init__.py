# Created by Andrew Davison
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
scheduler = BackgroundScheduler()

from coastal_forecast import routes  # noqa E402
