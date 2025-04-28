# scheduler.py - Simple daily runner script
import schedule
import time
from execution.daily_retrain import daily_pipeline

def main():
    schedule.every().day.at("07:00").do(daily_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()