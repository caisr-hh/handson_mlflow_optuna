"""Optuna Configuration"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class StudyConfig(BaseModel):
    """Optuna Study configs."""

    study: str
    seed: int  # for sampler
    load_if_exists: bool  # resume and existing study if true

    @property
    def name(self) -> str:
        """Study name constructed from model name, database, and study"""
        return f"{self.model_name}__{self.database_name}__{self.study}"


class StorageConfig(BaseModel):
    """Optuna Storage and DB config"""

    dir_path: Path
    db_name: str

    def model_post_init(  # pylint: disable=arguments-differ
        self, __context: Any | None = None
    ) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def uri(self) -> str:
        """Storage SQLite DB URI"""
        return f"sqlite:///{self.dir_path}/{self.db_name}"


class OptunaRunnerConfig(BaseModel):
    """Configuration of the runner"""

    n_trials: int
    n_jobs: int
    timeout: int
    n_startup_trials: int
    n_warmup_steps: int

    study: StudyConfig
    storage: StorageConfig
