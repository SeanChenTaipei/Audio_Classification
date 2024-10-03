from src.utils import write_json, read_json
from src.model_params import MODEL_PARAMS


class ModelSetting:
    def __init__(self, trans, algo, name=None, params=None, weight=None):
        self.trans = trans
        self.algo = algo
        self.name = name if name is not None else f"{algo}_{trans}"
        self.params = params if params is not None else MODEL_PARAMS[algo]
        self.weight = weight if weight is not None else 1

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n" + \
            "".join([f"  {k}={v},\n" for k, v in self.__dict__.items()]) + \
            ")"
        )

    def get_config(self):
        return dict(self.__dict__.items())


class EnsembleModelSetting:
    def __init__(self, model_settings):
        self.model_settings = model_settings

    def __iter__(self):
        return iter(self.model_settings)

    def __getitem__(self, i):
        return self.model_settings[i]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n" + \
            "".join(
                [
                    f"  {setting.__repr__().replace('  ', '    ').replace(')', '  )')},\n"
                    for setting in self.model_settings
                ]
            ) + ")"
        )

    def get_configs(self):
        return [setting.get_config() for setting in self.model_settings]

    def save(self, path):
        configs = self.get_configs()
        write_json(configs, path)

    def load(path):
        configs = read_json(path)
        return EnsembleModelSetting(
            [ModelSetting(**config) for config in configs]
        )
