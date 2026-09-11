from pydantic import BaseModel, Field, computed_field
from typing import Any
import yaml


class DatagenConfig(BaseModel):
    # Model data relevant parameters
    table : str
    n_samples: int = 10000
    noise: float = 0.1
    factor: float = 0.02
    train_split: float = 0.7
    random_state: int = 42
    batch_size: int = 32

    @classmethod
    def from_file(cls, path = "configs/datagen_config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        return cls(**yaml_config)


    


