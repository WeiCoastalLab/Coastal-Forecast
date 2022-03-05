# Created by Andrew Davison
from flask import Flask
from whitenoise import WhiteNoise

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='coastal_forecast/static/')

from coastal_forecast import routes  # noqa E402
