import logging


def setup_logger(log_path="results.log"):
    # Create a logger
    logger = logging.getLogger("training_results")
    logger.setLevel(logging.INFO)

    # Create file processor
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Create console processor
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a renderer
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add processor to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
