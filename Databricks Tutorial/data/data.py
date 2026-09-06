from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
from configs.pipeline_config import PipelineConfig, DataConfig
from dataclasses import dataclass
from typing import Any
import torch
import yaml
import pandas as pd
from pyspark.sql import SparkSession
import numpy as np

class ModelData(BaseModel):
    config: DataConfig
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

"""
def load_data_config(path="config/DefaultDataConfig.yaml"):
    with open(path, "r") as f:
        config = ModelConfig.model_validate(yaml.safe_load(f))
        return config

"""



def construct_data(config: DataConfig) -> ModelData:
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



def generate_data_db(config: DataConfig) -> ModelData:
    input, labels = make_circles(
        n_samples=config.n_samples,
        noise=config.noise,
        random_state=config.random_state,
        factor=config.factor,
    )

    input_train, input_test, label_train, label_test = train_test_split(
        input, labels, train_size=config.train_split, random_state=config.random_state
    )

    scale_x = (5*np.random.rand()+1)/6
    scale_y = (5*np.random.rand()+1)/6
    input_test[:,0] = input_test[:,0] * scale_x
    input_test[:,1] = input_test[:,1] * scale_y
    input_train[:,0] = input_train[:,0] * scale_x
    input_train[:,1] = input_train[:,1] * scale_y

    df_train = pd.DataFrame(input_train, columns=["x", "y"])
    df_test = pd.DataFrame(input_test, columns=["x", "y"])
    df_train["label"] = label_train
    df_test["label"] = label_test
    df_train["set"] = 0
    df_test["set"] = 1


    df_full = pd.concat([df_train, df_test], axis=0)
    spark = SparkSession.getActiveSession()
    sdf = spark.createDataFrame(df_full)
    sdf.write.mode("overwrite").saveAsTable(config.table)
    return

def get_data_db(config: DataConfig) -> ModelData:
    spark = SparkSession.getActiveSession()
    sdf = spark.table(config.table)
    pdf = sdf.toPandas()
    input_train = pdf[pdf["set"] == 0][["x", "y"]].values
    input_test = pdf[pdf["set"] == 1][["x", "y"]].values
    label_train = pdf[pdf["set"] == 0]["label"].values
    label_test = pdf[pdf["set"] == 1]["label"].values

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
