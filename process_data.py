import os
import json
import scipy
import torch
import librosa

import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict

from argparse import ArgumentParser
from librosa.feature import spectral_contrast, spectral_flatness, spectral_rolloff
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    WavLMModel,
)

from src.utils import write_json


mapping = {0: "train", 1: "public", 2: "private"}
t_mapping = {0: 48000, 1: 32000, 2: 32000}


SAMPLING_RATE = 16000
TRAIN_WAV_PATH = "dataset/Training Dataset/training_voice_data"
TRAIN_TAB_PATH = "dataset/Training Dataset/training datalist.csv"
PUBLIC_WAV_PATH = "dataset/test_data_public"
PUBLIC_TAB_PATH = "dataset/test_datalist_public.csv"
PRIVATE_WAV_PATH = None
PRIVATE_TAB_PATH = None
OUTPUT_PATH = ""
FEATURE_EXTRACTOR_PRETRAIN = "microsoft/wavlm-base-plus" #"superb/wav2vec2-base-superb-er"
PROCESSOR_PRTRIN_NAME = {
    "wavlm": "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
    "wav2vec": "facebook/wav2vec2-base",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="wavlm",
        help="wavlm or wav2vec.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        help="train or public or private."
    )

    return parser.parse_args()


def get_audio(id_list: list) -> dict:
    audio_data = dict()
    for id_ in id_list:
        audio_path = os.path.join(PUBLIC_WAV_PATH, f"{id_}.wav")
        sample = librosa.load(audio_path, sr=SAMPLING_RATE, duration=3)[0]
        audio_data[id_] = sample
    return audio_data


def get_freq_stats(id_, audio):
    freqs = np.fft.fftfreq(audio.size)
    p1_intercept, p1_slope = librosa.feature.poly_features(
        y=audio, sr=SAMPLING_RATE
    ).mean(axis=1)
    return OrderedDict(
        [
            ("ID", id_),
            ("freq_mean", np.mean(freqs)),
            ("freq_median", np.median(freqs)),
            ("freq_skew", scipy.stats.skew(freqs)),
            ("freq_flatness", spectral_flatness(y=audio).mean()),
            ("freq_contrast", spectral_contrast(y=audio, sr=SAMPLING_RATE).mean()),
            ("freq_rolloff", spectral_rolloff(y=audio, sr=SAMPLING_RATE).mean()),
            ("freq_p1_intercept", p1_intercept),
            ("freq_p1_slope", p1_slope),
        ]
    )


def get_audio_stats_feature(audio_data: dict) -> list:
    audio_feature_list = []
    for id_, audio in audio_data.items():
        audio_feature_list.append(get_freq_stats(id_, audio))
    return audio_feature_list


@torch.no_grad()
def get_audio_latent_feature(audio_data: dict, feature_extractor, processor, model) -> dict:
    audio_feature_extraction = {}
    for id_, audio in tqdm(audio_data.items()):

        audio = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=32000,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_values"][0]


        input_data = processor(
            audio,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt"
        )

        outputs = model(**input_data)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states_mean = last_hidden_states.squeeze().mean(dim=0)
        audio_feature_extraction[id_] = last_hidden_states_mean.tolist()
    return audio_feature_extraction


def get_wav_model(name):
    wav_model_dict = dict(
        wavlm=WavLMModel.from_pretrained("microsoft/wavlm-base-plus"),
        wav2vec=Wav2Vec2Model.from_pretrained("superb/wav2vec2-base-superb-er")
    )
    return wav_model_dict[name]


def preprocess_function(data, feature_extractor, max_len=48000):
    audio_arrays = [x["array"] for x in data["audio"]]
    return feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )


if __name__ == "__main__":
    args = parse_args()

    if OUTPUT_PATH:
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    df = pd.read_csv(PUBLIC_TAB_PATH)
    id_list = df["ID"].to_list()
    audio_data = get_audio(id_list)

    # Extract statistical frequency features
    audio_stats_feature = get_audio_stats_feature(audio_data)
    extra_df = pd.DataFrame(audio_stats_feature)
    merge_df = df.merge(extra_df, on="ID")
    merge_df.to_csv(os.path.join(OUTPUT_PATH, "train.csv"))

    # Extract latent features by deep learning model
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PRTRIN_NAME[args.model_name])
    feature_extractor = AutoFeatureExtractor.from_pretrained(FEATURE_EXTRACTOR_PRETRAIN)
    model = get_wav_model(args.model_name)
    audio_latent_feature = get_audio_latent_feature(
        audio_data,
        feature_extractor=feature_extractor,
        processor=processor,
        model=model,
    )
    write_json(audio_latent_feature, "latent_feature.json")
