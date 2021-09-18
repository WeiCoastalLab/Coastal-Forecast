from flask import render_template
from coastal_forecast import app


# home page
@app.route("/")
def home():
    return render_template('layout.html')
