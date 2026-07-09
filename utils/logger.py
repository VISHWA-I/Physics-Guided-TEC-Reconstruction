import logging
from pathlib import Path
from typing import Optional


def get_model_logger(name: str = "ModelFramework", log_file: str = "logs/model.log", level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger specifically for the model framework.
    Records events to the specified log file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File Handler
        fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
        fh.setLevel(level)

        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

