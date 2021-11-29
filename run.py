# Created by Andrew Davison
from coastal_forecast import app, scheduler
from coastal_forecast.routes import scheduled_task

if __name__ == "__main__":
    scheduled_task()
    scheduler.add_job(scheduled_task, 'interval', hours=6)
    scheduler.start()
    app.run(debug=True, use_reloader=False)
