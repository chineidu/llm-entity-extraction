from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from ner_extraction import PACKAGE_ROOT

from ..schemas import BaseSchema


class Data(BaseSchema):
    """Data configuration class."""

    data_path: str = Field(description="Training data path")
    download_data_path: str = Field(description="Path to download data from Google Drive")
    bucket_name: str = Field(description="Google Cloud Storage bucket name")


class Database(BaseSchema):
    """Database configuration class."""

    filename: str = Field(description="Database filename")
    db_path: str = Field(description="Database path")


class Prompts(BaseSchema):
    """Prompts configuration class."""

    prompt_path: str = Field(description="Prompt path")
    thinking_mode: bool = Field(description="Whether to use thinking mode")
    num_entity: str = Field(description="Number of entities to extract per transaction")


class ModelHyperparams(BaseSchema):
    """Model hyperparameters class."""

    temperature: float = Field(description="Temperature for model sampling")
    seed: int = Field(description="Seed for model sampling")


class Inference(BaseSchema):
    """Inference configuration class."""

    batch_size: int = Field(ge=5, le=500, description="Batch size for model inference")
    concurrency_limit: int = Field(ge=1, le=30, description="Concurrency limit for model inference")
    max_connections: int = Field(ge=1, le=20, description="Max connections for model inference")
    mini_batch_size: int = Field(ge=5, le=50,description="Mini batch size for model inference")
    use_vllm: bool = Field(description="Whether to use VLLM for model inference")
    model: ModelHyperparams = Field(description="Model hyperparameters")


class AppConfig(BaseSchema):
    """Application configuration class."""

    data: Data = Field(description="Data configuration")
    database: Database = Field(description="Database configuration")
    prompts: Prompts = Field(description="Prompts configuration")
    inference: Inference = Field(description="Inference configuration")


config_path: Path = PACKAGE_ROOT / "config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).app_config
app_config: AppConfig = AppConfig(**config)  # type: ignore
