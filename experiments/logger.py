import logging


def configure_logger(filename=None, level=logging.DEBUG):
    """Configure logger to log to file, if desired, and stdout."""
    handlers = [logging.StreamHandler()]
    if filename is not None:
        handlers += [logging.FileHandler(filename, mode="a")]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )
