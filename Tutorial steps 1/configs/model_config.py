from pydantic import BaseModel
from typing import Any


class ModelConfig(BaseModel):
    # Model architecture
    n_width: int
    n_depth: int
    learning_rate: float = 1e-3
    epoch_max: int = 50
    seed: int = 42

    # Model data relevant parameters
    n_samples: int = 10000
    noise: float = 0.1
    factor: float = 0.02
    train_split: float = 0.7
    random_state: int = 42
    batch_size: int = 32


class RunInfo(BaseModel):
    run_id: str | None = None
    run_id_parent: str | None = None
    trial: Any | None = None
