import os
import json
import math
import random
import scipy
import torch
import logging
import librosa

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Union
from collection import OrderedDict

from pathlib import Path
from argparse import ArgumentParser, Namespace
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, balanced_accuracy_score
from librosa.feature import spectral_contrast, spectral_flatness, spectral_rolloff

from transformers.data.data_collator import DataCollatorWithPadding
from transformers import AutoProcessor, AutoFeatureExtractor, AutoConfig, PreTrainedModel, TrainingArguments, Trainer, Wav2Vec2Model, Wav2Vec2FeatureExtractor, WavLMModel, Wav2Vec2Processor


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
mapping = {
    0: 'train',
    1: 'public', 
    2: 'private'
}

t_mapping = {
    0: 48000,
    1: 32000, 
    2: 32000
}

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_wav", type=str, default=None,
        help="The path of directory of the training audio datas."
    )
    parser.add_argument(
        "--train_csv", type=str, default=None,
        help="The path of the training csv."
    )
    parser.add_argument(
        "--public_wav", type=str, default=None,
        help="The path ofdirectory of the training audio datas."
    )
    parser.add_argument(
        "--public_csv", type=str, default=None,
        help="The path of the public csv."
    )
    parser.add_argument(
        "--private_wav", type=str, default=None,
        help="The path ofdirectory of the training audio datas."
    )
    parser.add_argument(
        "--private_csv", type=str, default=None,
        help="The path of the private csv."
    )
    parser.add_argument(
        "--output_path", type=str, default='./',
        help="Path to store the generation."
    )
    args = parser.parse_args()

    return args


def describe_freq(x):
    freqs = np.fft.fftfreq(x.size)
    p1, p2 = librosa.feature.poly_features(y=x, sr=16000).mean(axis=1)
    freq_stat = OrderedDict([
        ('mean', np.mean(freqs)),
        ('median', np.median(freqs)),
        ('skew', scipy.stats.skew(freqs)),
        ('contrast', spectral_contrast(y=x, sr=16000).mean()),
        ('flatness', spectral_flatness(y=x).mean()),
        ('rolloff', spectral_rolloff(y=x, sr=16000).mean()),
        ('p1', p1),
        ('p2', p2),
    ])
    return freq_stat


def get_dataset(path: str, df_MH: pd.DataFrame) -> Dataset:
    datalist = []
    metalist=[]
    id_list = df_MH.index.to_list()
    for index, name in enumerate(tqdm(id_list)):
        data = {}
        file = path+'/'+name+'.wav'
        sample = librosa.load(file, sr=16000, duration=3)[0]
        data['file'] = file
        data['audio'] = dict(path=file, array=torch.as_tensor(sample), sampling_rate=16000)
        data['speaker_id'] = name
        data['additional_data'] = df_MH.values[index]
        datalist.append(data)   
        metalist.append(describe_freq(sample)) 
    stats = pd.DataFrame(metalist, index=id_list)

    return Dataset.from_list(datalist), df_MH.join(stats).fillna(0)


def preprocess_function(data, feature_extractor, max_len=48000):
    audio_arrays = [x["array"] for x in data["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    return inputs


if __name__ == "__main__":
    args = parse_args()
    outpath = args.output_path
    os.makedirs(outpath, exist_ok=True)

    # Pretrained WavLM
    logging.info(f'Loding Pretrained WavLMModel')
    PRETRAINED_PATH = "microsoft/wavlm-base-plus"
    processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_PATH)
    model = WavLMModel.from_pretrained(PRETRAINED_PATH)
    for index, (wav_path, csv_path) in enumerate(zip([args.train_wav, args.public_wav, args.private_wav],
                                                     [args.train_csv, args.public_csv, args.private_csv])):
        logging.info(f'Encoding {mapping.get(index)} audio data - WavLM')
        data = {}
        ## Dataset generation
        d, df = get_dataset(wav_path, pd.read_csv(csv_path, index_col=0))
        df.to_csv(f'{outpath}/{mapping.get(index)}_all.csv')
        dset = d.map(lambda data: preprocess_function(data, feature_extractor, t_mapping.get(index)), remove_columns="audio", batched=True)
        # Audio file is decoded on the fly

        for i, data in tqdm(enumerate(dset)):
            idd = dset[i]['speaker_id']
            if mapping.get(index)=='train':
                inputs = processor(d[i]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
            else:
                inputs = processor(dset[i]['input_values'], sampling_rate=16000, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state

            data[idd] = last_hidden_states.squeeze().mean(dim=0).tolist()
        path = os.path.join(outpath, f'wavlm_{mapping.get(index)}_ms.json')

        with open(path, 'w') as fp:
            json.dump(data, fp)

    # Pretrained Wav2Vec2
    logging.info(f'Loding Pretrained Wav2Vec2Model')
    PRETRAINED_PATH = "superb/wav2vec2-base-superb-er"
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_PATH)
    model = Wav2Vec2Model.from_pretrained(PRETRAINED_PATH)
    for index, (wav_path, csv_path) in enumerate(zip([args.train_wav, args.public_wav, args.private_wav],
                                                     [args.train_csv, args.public_csv, args.private_csv])):
        logging.info(f'Encoding {mapping.get(index)} audio data - Wav2Vec2')
        data = {}
        ## Dataset generation
        d, _ = get_dataset(wav_path, pd.read_csv(csv_path, index_col=0))
        dset = d.map(lambda data: preprocess_function(data, feature_extractor), remove_columns="audio", batched=True)
        # Audio file is decoded on the fly
        for i in tqdm(range(len(dset))):
            idd = dset[i]['speaker_id']
            # inputs = processor(d[i]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
            inputs = processor(dset[i]['input_values'], sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            data[idd] = last_hidden_states.squeeze().mean(dim=0).tolist()
        path = os.path.join(outpath, f'wav2vec2_{mapping.get(index)}_s3prl.json')

        with open(path, 'w') as fp:
            json.dump(data, fp)
