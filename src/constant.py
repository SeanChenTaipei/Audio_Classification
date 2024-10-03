import os
from collections import OrderedDict


# file
DATA_FOLDER = "dataset"
TRAIN_TABULAR_FILE = "tabular_train.csv"
TRAIN_WAVLM_FILE = "wavlm_train_ms.json"
TRAIN_WAV2VEC_FILE = "wav2vec2_train_s3prl.json"
PUBLIC_WAVLM_FILE = "wavlm_public_ms.json"
PUBLIC_TABULAR_FILE = "tabular_public.csv"
PUBLIC_WAV2VEC_FILE = "wav2vec2_public_s3prl.json"
PRIVATE_WAVLM_FILE = "wavlm_private_ms.json"
PRIVATE_TABULAR_FILE = "tabular_private.csv"
PRIVATE_WAV2VEC_FILE = "wav2vec2_private_s3prl.json"
CHECKPOINTS = "checkpoints"
CONFIG_FILE = "config.json"
OUTPUT_FOLDER = "prediction"
OUTPUT_FILE = "submissions.csv"

TRAIN_TABULAR_PATH = os.path.join(DATA_FOLDER, TRAIN_TABULAR_FILE)
TRAIN_WAVLM_PATH = os.path.join(DATA_FOLDER, TRAIN_WAVLM_FILE)
TRAIN_WAV2VEC_PATH = os.path.join(DATA_FOLDER, TRAIN_WAV2VEC_FILE)
PUBLIC_TABULAR_PATH = os.path.join(DATA_FOLDER, PUBLIC_TABULAR_FILE)
PUBLIC_WAVLM_PATH = os.path.join(DATA_FOLDER, PUBLIC_WAVLM_FILE)
PUBLIC_WAV2VEC_PATH = os.path.join(DATA_FOLDER, PUBLIC_WAV2VEC_FILE)
PRIVATE_WAVLM_PATH = os.path.join(DATA_FOLDER, PRIVATE_WAVLM_FILE)
PRIVATE_TABULAR_PATH = os.path.join(DATA_FOLDER, PRIVATE_TABULAR_FILE)
PRIVATE_WAV2VEC_PATH = os.path.join(DATA_FOLDER, PRIVATE_WAV2VEC_FILE)
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

# data
TABULAR = "tabular"
WAVLM_ENCODE = "wavlm-encode"
WAV2VEC_ENCODE = "wav2vec-encode"
LABEL_COLUMN = "Disease category"

TRAIN_DATA = OrderedDict(
    [
        (TABULAR, TRAIN_TABULAR_PATH),
        (WAVLM_ENCODE, TRAIN_WAVLM_PATH),
        (WAV2VEC_ENCODE, TRAIN_WAV2VEC_PATH),
    ]
)

PUBLIC_DATA = OrderedDict(
    [
        (TABULAR, PUBLIC_TABULAR_PATH),
        (WAVLM_ENCODE, PUBLIC_WAVLM_PATH),
        (WAV2VEC_ENCODE, PUBLIC_WAV2VEC_PATH),
    ]
)

PRIVATE_DATA = OrderedDict(
    [
        (TABULAR, PRIVATE_TABULAR_PATH),
        (WAVLM_ENCODE, PRIVATE_WAVLM_PATH),
        (WAV2VEC_ENCODE, PRIVATE_WAV2VEC_PATH),
    ]
)

# model
BRF = "brf"
LGBM = "lgbm"
TABPFN = "tabpfn"
BBC = "bbc"
BRF_BBC = "brf-bbc"
LGBM_BBC = "lgbm-bbc"
TABPFN_BBC = "tabpfn-bbc"
