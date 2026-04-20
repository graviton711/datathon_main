import logging
import sys
from src.config import config

def setup_logger(name="datathon"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(config.LOGS_DIR / "project.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
