from flask import render_template
from datetime import datetime, timedelta
from coastal_forecast import app


# home page
@app.route("/")
@app.route("/41013", methods=['GET', 'POST'])
def home():
    return render_template('41013.html', title='41013', current=datetime.utcnow().strftime('%H:%M'),
                           next=(datetime.utcnow() + timedelta(hours=6)).strftime('%H:%M'))


@app.route("/41009", methods=['GET', 'POST'])
def canaveral():
    return render_template('41009.html', title='41009', current=datetime.utcnow().strftime('%H:%M'),
                           next=(datetime.utcnow() + timedelta(hours=6)).strftime('%H:%M'))


@app.route("/44013", methods=['GET', 'POST'])
def boston():
    return render_template('44013.html', title='44013', current=datetime.utcnow().strftime('%H:%M'),
                           next=(datetime.utcnow() + timedelta(hours=6)).strftime('%H:%M'))


@app.route("/41008", methods=['GET', 'POST'])
def grays():
    return render_template('41008.html', title='41008', current=datetime.utcnow().strftime('%H:%M'),
                           next=(datetime.utcnow() + timedelta(hours=6)).strftime('%H:%M'))
