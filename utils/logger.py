import logging
import os
import datetime
import sys

_global_file_handler = None

def setup_logger(name="asr_benchmark", log_level=logging.INFO, log_to_file=True):
    """
    Sets up a logger that handles both console and file output.
    If a file handler already exists, it will reuse it to ensure single log file per run.
    """
    global _global_file_handler
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        if _global_file_handler is None:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_file = os.path.join(log_dir, f"run_{timestamp}.log")
            _global_file_handler = logging.FileHandler(log_file, encoding='utf-8')
            _global_file_handler.setLevel(log_level)
            _global_file_handler.setFormatter(formatter)
        
        logger.addHandler(_global_file_handler)

    return logger

def get_logger(name):
    """
    Returns a logger with the given name, ensuring it's properly set up.
    """
    return setup_logger(name)
