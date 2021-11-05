import sched
import time

s = sched.scheduler(time.time, time.sleep)


def something_else(sec=0):
    time.sleep(10)
    sec += 10
    print(sec, end=" ", flush=True)
    if sec < 60:
        something_else(sec)
    return


def do_something(sc):
    print("Doing stuff...")
    # something to do
    something_else()
    s.enter(1, 1, do_something, (sc,))


print("Calling scheduler...")
s.enter(1, 1, do_something, (s,))
s.run()
