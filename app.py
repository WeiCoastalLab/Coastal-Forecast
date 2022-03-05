# Created by Andrew Davison
from apscheduler.schedulers.background import BackgroundScheduler
from coastal_forecast import app, prediction_manager as pm


def timed_job():
    stations = ['41008', '41009', '41013', '44013']
    for station in stations:
        pm.get_prediction(station, 9, 3)


if __name__ == '__main__':
    timed_job()
    scheduler = BackgroundScheduler(timezone='UTC', daemon=True)
    scheduler.add_job(timed_job, 'interval', hours=6)
    try:
        print("Starting scheduler...")
        scheduler.start()
    except KeyboardInterrupt or SystemExit:
        scheduler.shutdown(wait=False)
    app.run(debug=True, use_reloader=False)
