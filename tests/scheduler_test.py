import datetime
import sched
import time

from coastal_forecast.prediction_manager import get_prediction

s = sched.scheduler(time.time, time.sleep)


def do_something(sc):
    print(f'Starting system prediction at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    # something to do
    stations = ['41008', '41009', '41013', '44013']
    for station in stations:
        get_prediction(station, 9, 3)
    s.enter(5, 1, do_something, (sc,))


print("Calling scheduler...")
s.enter(0, 1, do_something, (s,))
s.run()
