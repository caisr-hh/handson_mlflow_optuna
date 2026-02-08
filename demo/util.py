import yaml
from pathlib import Path
from demo.constants import CONFIG_DIR
from configs.model_config import ModelConfig
from configs.mlflow_config import MlflowServiceConfig
from configs.optuna_config import OptunaRunnerConfig


def read_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(path=CONFIG_DIR.MODEL.value) -> ModelConfig:

    config = ModelConfig.model_validate(read_config(path))
    return config


def load_mlflow_config(path=CONFIG_DIR.MLFLOW.value) -> MlflowServiceConfig:

    config = MlflowServiceConfig.model_validate(read_config(path))
    return config


def load_optuna_config(path=CONFIG_DIR.OPTUNA.value) -> OptunaRunnerConfig:

    config = OptunaRunnerConfig.model_validate(read_config(path))
    return config
