from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
from configs.model_config import ModelConfig
from dataclasses import dataclass
from typing import Any
import torch
import yaml


class ModelData(BaseModel):
    config: ModelConfig
    training_loader: Any
    test_loader: Any


@dataclass
class EpochMetrics:
    epoch_loss: float
    epoch_accuracy: float


@dataclass
class TestMetrics:
    test_loss: float
    test_accuracy: float


def load_data_config(path="config/DefaultDataConfig.yaml"):
    with open(path, "r") as f:
        config = ModelConfig.model_validate(yaml.safe_load(f))
        return config


def construct_data(config: ModelConfig) -> ModelData:
    input, labels = make_circles(
        n_samples=config.n_samples,
        noise=config.noise,
        random_state=config.random_state,
        factor=config.factor,
    )

    input_train, input_test, label_train, label_test = train_test_split(
        input, labels, train_size=config.train_split, random_state=config.random_state
    )

    input_train = torch.tensor(input_train, dtype=torch.float32)
    input_test = torch.tensor(input_test, dtype=torch.float32)
    label_train = torch.tensor(label_train, dtype=torch.float32).unsqueeze(1)
    label_test = torch.tensor(label_test, dtype=torch.float32).unsqueeze(1)

    dataset_train = TensorDataset(input_train, label_train)
    dataset_test = TensorDataset(input_test, label_test)

    training_loader = DataLoader(dataset_train, batch_size=config.batch_size)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size)

    data = ModelData(
        config=config, training_loader=training_loader, test_loader=test_loader
    )
    return data
