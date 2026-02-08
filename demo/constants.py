from enum import Enum


class LOGGERS(Enum):
    OPTUNA = "Optuna"
    MLFLOW = "MLFlow"
    LOCAL = "Local"
    FINAL = "Final"


class CONFIG_DIR(Enum):
    MODEL = "configs/model_config.yaml"
    OPTUNA = "configs/optuna.yaml"
    MLFLOW = "configs/mlflow.yaml"

OPTUNA_STUDY_NAME = "Tutorialstudy_MLP"

MLFLOW_EXPERIMENT_NAME = "Tutorial_tooling_MLP"

REGISTERED_MODEL_NAME = "MLP_circles"
