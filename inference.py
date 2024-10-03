import pandas as pd
import os
import warnings

from src.constant import (
    PUBLIC_DATA, PRIVATE_DATA,
    OUTPUT_FOLDER,
)
from src.model_setting import EnsembleModelSetting
from src.pipeline import EnsemblePipeline
from src.utils import makedirs, get_logger


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
logger = get_logger()


@makedirs
def save_pred(pred_df, path):
    pred_df.to_csv(path)


if __name__ == "__main__":
    checkpoint = os.path.join("checkpoints", "20240718153747")
    config_path = os.path.join(checkpoint, "config.json")
    model_settings = EnsembleModelSetting.load(config_path)
    pipe = EnsemblePipeline.load(checkpoint, model_settings)

    public_pred = pipe.predict(PUBLIC_DATA)
    public_pred_df = pd.DataFrame(public_pred)
    save_pred(public_pred_df, os.path.join(OUTPUT_FOLDER, checkpoint, "public.csv"))

    private_pred = pipe.predict(PRIVATE_DATA)
    private_pred_df = pd.DataFrame(private_pred)
    save_pred(private_pred_df, os.path.join(OUTPUT_FOLDER, checkpoint, "private.csv"))
