from apscheduler.schedulers.blocking import BlockingScheduler
from coastal_forecast.routes import run_predictions

sched = BlockingScheduler()


@sched.scheduled_job('interval', hours=6)
def timed_job():
    run_predictions()


sched.start()
