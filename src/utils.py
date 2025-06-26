import logging
import time

from pathlib import Path

def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

log_file_path = Path("logs") / f"{get_timestamp()}.log"

def get_logger(name: str, args: str) -> logging.Logger:
    """
    Create a logger with the specified name and level.
    
    Args:
        name (str): The name of the logger.
        level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'DEBUG'.
        file_path (Path): The path to the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """

    if args.local_rank != 0:
        return None
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not log_file_path.exists():
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    ch = logging.StreamHandler()
    ch.setLevel(logger.level)

    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logger.level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger