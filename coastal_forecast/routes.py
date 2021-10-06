from flask import render_template
from coastal_forecast import app


# home page
@app.route("/")
@app.route("/41013")
def home():
    return render_template('41013.html', title='41013')


@app.route("/41009")
def canaveral():
    return render_template('41009.html', title='41009')


@app.route("/44013")
def boston():
    return render_template('44013.html', title='44013')


@app.route("/41008")
def grays():
    return render_template('41008.html', title='41008')
