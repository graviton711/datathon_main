import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from src.config import config
from src.data.loader import DataLoader

def main():
    logger = setup_logger()
    logger.info("Initializing Datathon 2026 Pipeline...")
    
    loader = DataLoader()
    logger.info(f"Project Root: {config.PROJECT_ROOT}")
    
    # Ready for the first EDA notebook
    logger.info("Structure verification complete. Ready for development.")

if __name__ == "__main__":
    main()
