"""MLflow configuration module."""

from pathlib import Path

from pydantic import BaseModel


class MlflowServiceConfig(BaseModel):
    """Configuration for the MLflow service"""

    scheme: str
    host: str
    port: int
    backend_kind: str
    backend_path: Path
    artifacts_root: Path

    @property
    def tracking_uri(self) -> str:
        """Return MLflow tracking URI"""
        return f"{self.scheme}://{self.host}:{self.port}"
