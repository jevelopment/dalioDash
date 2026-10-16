import logging
import os
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "dashboard.log")


def get_logger(name: str = "dashboard") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file: 1 MB per file, keep 5 backups
    fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Also print to stdout so cron captures it
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
