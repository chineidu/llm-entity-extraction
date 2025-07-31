from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET_KEY = SecretStr("empty")


class BaseSettingsConfig(BaseSettings):
    """Base configuration class for settings.

    This class extends BaseSettings to provide common configuration options
    for environment variable loading and processing.

    Attributes
    ----------
    model_config : SettingsConfigDict
        Configuration dictionary for the settings model specifying env file location,
        encoding and other processing options.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
    )


class Settings(BaseSettingsConfig):
    """Application settings class containing database and other credentials."""

    # ENV
    ENVIRONMENT: Literal["staging", "development", "production"] = "development"
    # BASE
    INDICINA_API_KEY: SecretStr = DEFAULT_SECRET_KEY
    INDICINA_BASE_URL: str = "http://0.0.0.0"
    INDICINA_PORT: int = 8000

    # LANGFUSE
    LANGFUSE_SECRET_KEY: SecretStr = DEFAULT_SECRET_KEY
    LANGFUSE_PUBLIC_KEY: SecretStr = DEFAULT_SECRET_KEY
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # OPENROUTER
    OPENROUTER_API_KEY: SecretStr = DEFAULT_SECRET_KEY
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    # HUGGINGFACE
    HUGGINGFACE_API_KEY: SecretStr = DEFAULT_SECRET_KEY

    # RUNPOD
    RUNPOD_API_KEY: SecretStr = DEFAULT_SECRET_KEY
    RUNPOD_BASE_URL: str = "https://runpod.ai/api/v1"
    RUNPOD_ENDPOINT_ID: str = ""

    @property
    def INDICINA_URL(self) -> str:  # noqa: N802
        """Construct the full base URL including host and port.

        Returns
        -------
        str
            The complete base URL in the format 'http://<BASE_URL>:<BASE_PORT>/v1'
            where BASE_URL and BASE_PORT are configuration values.
        """
        return f"{self.INDICINA_BASE_URL}:{self.INDICINA_PORT}/v1"

    @property
    def RUNPOD_URL(self) -> str:  # noqa: N802
        """Construct the full base URL including host and port.

        Returns
        -------
        str
            The complete base URL in the format 'http://<BASE_URL>:<BASE_PORT>/v1'
            where BASE_URL and BASE_PORT are configuration values.
        """
        return f"{self.RUNPOD_BASE_URL}/v2/{self.RUNPOD_ENDPOINT_ID}/openai/v1"


def refresh_settings() -> Settings:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    Settings
        A new Settings instance with refreshed environment variables
    """
    load_dotenv(override=True)
    return Settings()  # type: ignore
