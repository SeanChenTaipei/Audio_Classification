import os
import sys
import json


def hidden_message(func):
    def wrap(path, *args):
        sys.stdout = open(os.devnull, 'w')
        func(path, *args)
        sys.stdout.close()
    return wrap


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
