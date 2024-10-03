import warnings
import os

from src.constant import CHECKPOINTS, CONFIG_FILE, TRAIN_DATA
from src.pipeline import EnsemblePipeline
from src.model_setting import EnsembleModelSetting, ModelSetting
from src.utils import get_datetime, get_logger


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
logger = get_logger()


def get_model_setting_list():
    return [
        EnsembleModelSetting(
            [
                ModelSetting(
                    trans="v1",
                    algo="brf-bbc",
                ),
                ModelSetting(
                    trans="v1",
                    algo='lgbm-bbc',
                ),
                ModelSetting(
                    trans="v1",
                    algo='tabpfn-bbc',
                ),
            ]
        ),
        EnsembleModelSetting(
            [
                ModelSetting(
                    trans="v2",
                    algo="brf-bbc",
                    weight=0.125,
                ),
                ModelSetting(
                    trans="v2",
                    algo='lgbm-bbc',
                    weight=0.5,
                ),
            ]
        ),
    ]


def main():
    for model_settings in get_model_setting_list():
        checkpoint_folder = os.path.join(CHECKPOINTS, get_datetime())
        os.makedirs(checkpoint_folder, exist_ok=True)
        model_settings.save(os.path.join(checkpoint_folder, CONFIG_FILE))

        pipe = EnsemblePipeline(model_settings, logger)
        pipe.train(TRAIN_DATA)
        pipe.save(checkpoint_folder)


if __name__ == "__main__":
    main()
