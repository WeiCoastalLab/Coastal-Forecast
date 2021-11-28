from coastal_forecast import app, scheduler
from coastal_forecast.routes import scheduled_task

if __name__ == "__main__":
    scheduler.add_job(scheduled_task, 'interval', seconds=5)
    scheduler.start()
    app.run()  #debug=True)
