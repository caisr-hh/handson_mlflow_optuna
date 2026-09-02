from pydantic import BaseModel, Field, computed_field
from typing import Any

# Todo: Move to loggers
class RunInfo(BaseModel):
    run_id: str | None = None
    parent_run_id: str | None = None
    trial: Any | None = None
    study: Any | None = None

class ModelConfig(BaseModel):
    # Model architecture
    n_width: int
    n_depth: int
    learning_rate: float = 1e-3
    epoch_max: int = 50
    seed: int = 42

class DataConfig(BaseModel):
    # Model data relevant parameters
    table : str
    n_samples: int = 10000
    noise: float = 0.1
    factor: float = 0.02
    train_split: float = 0.7
    random_state: int = 42
    batch_size: int = 32

class LoggerConfig(BaseModel):

    experiment_root: str
    verbosity: int = 1

    @computed_field
    @property
    def model_name(self) -> str:
        return self.experiment_root + "_model"
    
    @computed_field
    @property
    def experiment_hpo(self) -> str:
        return "/" + self.experiment_root + "_hpo"

    @computed_field
    @property
    def experiment_train(self) -> str:
        return "/" + self.experiment_root
    

 

class PipelineConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    logger: LoggerConfig
    


