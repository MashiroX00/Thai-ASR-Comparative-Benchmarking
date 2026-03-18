#Check Device Compatibilities
import torch
import os
import logging
from utils.logger import get_logger

logger = get_logger("compatibility_check")

def check_gpu_compatibility():
    try:
        if torch.cuda.is_available():
            logger.info("GPU is available.")
            return True
        else:
            logger.warning("GPU is not available.")
            return False
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False
    
if __name__ == "__main__":
    logger.info("Checking GPU compatibility...")
    check_gpu_compatibility()