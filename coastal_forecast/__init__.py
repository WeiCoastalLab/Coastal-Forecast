from flask import Flask

app = Flask(__name__)

from coastal_forecast import routes  # noqa E402
