import logging
from logging import Logger


def et_get_logger() -> Logger:
    logger = logging.getLogger("erictransformer")
    handler = logging.StreamHandler()
    handler.addFilter(logging.Filter("erictransformer"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[handler],
    )
    return logger
