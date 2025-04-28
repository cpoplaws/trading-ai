# logger.py - Simple structured logger
import logging
import os

def setup_logger(logfile='./logs/system.log'):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    return logging.getLogger()

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger initialized successfully.")