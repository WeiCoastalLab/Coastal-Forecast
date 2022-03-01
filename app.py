# Created by Andrew Davison
from coastal_forecast import app, scheduler
from coastal_forecast.routes import run_predictions

if __name__ == "__main__":
    run_predictions()
    # app.run(debug=True, use_reloader=False)
    scheduler.add_job(run_predictions, 'interval', hours=6)
    scheduler.start()
    app.run(debug=True, use_reloader=False)
