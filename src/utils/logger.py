import logging
import os
from datetime import datetime

def setup_logger(name='trading_ai_logger', log_dir='./logs/'):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{datetime.today().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger