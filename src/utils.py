import json
import logging
import os
import sys
from datetime import datetime
from contextlib import ContextDecorator


class HiddenMessage(ContextDecorator):
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.original_stdout

hidden_message = HiddenMessage()


def makedirs(func):
    def wrap(data, path, *args, **kwarg):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        func(data, path, *args, **kwarg)
    return wrap


def get_datetime():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        level=logging.INFO,
    )
    return logging.getLogger()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@makedirs
def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return
