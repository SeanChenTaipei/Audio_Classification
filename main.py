# Basic usage
import os
import csv
import random
import string
import numpy as np
import pandas as pd

# Logger & Parser
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output_path", 
        type=str, 
        default='./', 
        help="Path to store the generation."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ## Parsing Arguments
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    outpath = os.path.join(args.output_path, 'submission.csv')
    e2 = pd.read_csv('Result/ensemble2_proba.csv', index_col=0)
    e1 = pd.read_csv('Result/ensemble1_proba.csv', index_col=0)
    y_pred = np.argmax(e1+e2, axis=1)+1
    pd.DataFrame(y_pred, index=e1.index).to_csv(outpath, header=False)
