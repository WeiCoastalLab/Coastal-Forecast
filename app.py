# Created by Andrew Davison
from coastal_forecast import app, scheduler, timed_job


if __name__ == "__main__":
    scheduler.add_job(timed_job, 'interval', minutes=2)  # hours=6)
    try:
        scheduler.start()
    except KeyboardInterrupt or SystemExit:
        scheduler.shutdown(wait=False)
    app.run(debug=True, use_reloader=False)
