import logging
import sys
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from loggingFormatter import JsonFormatter
from pythonjsonlogger import jsonlogger


def init_logger(file_name: str) -> None:
    """
    Initialize logging, creating necessary folder and file if it doesn't already exist

    Parameters
    ______________
    file_name: str
        the name of the file for logging

    """
    # Assume script is called from top-level directory
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configue handlers to print logs to file and std out
    file_handler = logging.FileHandler(filename=f"logs/{file_name}")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stdout_handler],
    )


def init_logger_json(file_name: str) -> None:
    """
    Initialize logger - output is a json file.
    :param file_name:
        Name of the file, example my_log
    :return:
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    logger = logging.getLogger()
    # create a file handler
    file_handler = logging.FileHandler(log_dir / f"{file_name}.json")
    json_formatter = JsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    file_handler.setFormatter(json_formatter)
    # add the file handler to the logger
    logger.addHandler(file_handler)
    # set the log level
    logger.setLevel(logging.INFO)


def persist_df(df: pd.DataFrame,
               name: str,
               file_path: str = './data',) -> None:
    """
    Persist a dataframe to a specified file path and file name

    Parameters
    __________________
    df: pd.DataFrame
        the dataframe to persist

    file_path: str
        the path to create if it does not exist example your/path

    name: str
        name of the file

    Return
    __________________
    None
    """
    # persist file
    today = datetime.now().isoformat()
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = f"{file_path}/{name}_{today}.parq"
    logging.info(f"persisting file:\t{file_name}")
    df.to_parquet(file_name)