# scheduler.py - Simple daily runner script
import schedule
import time
from execution.daily_retrain import daily_pipeline
import traceback
from datetime import datetime

def job():
    print(f"[{datetime.now()}] Starting daily pipeline...")
    try:
        daily_pipeline()
        print(f"[{datetime.now()}] Pipeline completed successfully.")
    except Exception as e:
        print(f"[{datetime.now()}] Error occurred:")
        traceback.print_exc()

def main():
    schedule.every().day.at("07:00").do(job)
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()