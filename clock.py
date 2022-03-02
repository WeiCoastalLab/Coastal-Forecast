import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from coastal_forecast import prediction_manager as pm

sched = BlockingScheduler(timezone=pytz.UTC)


@sched.scheduled_job('interval', hours=6)
def timed_job():
    stations = ['41008', '41009', '41013', '44013']
    for station in stations:
        pm.get_prediction(station, 9, 3)


timed_job()
sched.start()
