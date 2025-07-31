import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import click
from pydantic.validate_call_decorator import validate_call
from sqlalchemy.orm import Session

from ner_extraction import create_logger
from ner_extraction.config import app_config
from ner_extraction.schemas import AllEntitySchemaResponse, ModelEnum
from ner_extraction.schemas.ner_models import engine, init_db
from ner_extraction.utilities.entity_utils import BASE_URL, ENVIRONMENT, async_get_batch_entities
from ner_extraction.utilities.utils import (
    async_download_file_from_gdrive,
    async_load_json_data,
    async_timer,
    clean_text,
    create_path,
    list_files,
    upload_file_to_gcs,
)

logger = create_logger(name="get_predictions")
timestamp: str = datetime.now().strftime("%Y_%m_%d__%H_%M")


DOWNLOAD_DATA_PATH: str = app_config.data.download_data_path
FILENAME: str = f"{app_config.database.filename}-{timestamp}.db"
BATCH_SIZE: int = app_config.inference.batch_size
BUCKET_NAME: str = app_config.data.bucket_name


def prepare_data(texts_with_idx: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Process and clean input texts with their indices.

    Parameters
    ----------
    texts_with_idx : list[tuple[str, str]]
        List of tuples containing (index, text) pairs to be processed.

    Returns
    -------
    list[tuple[str, str]]
        List of tuples containing (index, cleaned_text) pairs.

    Raises
    ------
    ValueError
        If any input tuple doesn't contain exactly 2 elements.
    """
    is_invalid: list[bool] = [len(row) != 2 for row in texts_with_idx]
    if any(is_invalid):
        raise ValueError("Invalid data format")
    cleaned_texts: list[tuple[str, str]] = [(idx, clean_text(text)) for idx, text in texts_with_idx]

    return cleaned_texts


@async_timer
@validate_call
async def get_predictions(
    data_path: Path | str = DOWNLOAD_DATA_PATH,
    batch_size: int = 5,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Process text data to extract named entities using batch processing.

    Parameters
    ----------
    batch_size : int, optional
        Number of texts to process in each batch, by default 5.

    Returns
    -------
    list[dict[str, Any]] | dict[str, Any]
    """
    global MODEL

    init_db()

    response_model = AllEntitySchemaResponse

    if isinstance(data_path, str):
        data_path = Path(data_path)

    format_type: Literal["json_file", "jsonl_file"] | None = (
        "json_file"
        if data_path.suffix[1:] == "json"
        else ("jsonl_file" if data_path.suffix[1:] == "jsonl" else None)
    )
    if format_type is None:
        return {
            "error": "Invalid format type",
            "message": "Supported formats: json_file and jsonl_file",
        }

    data: list[tuple[str, str]] = await async_load_json_data(data_path, format_type=format_type)
    data = prepare_data(data)
    model_str: ModelEnum = MODEL

    with Session(engine) as session:
        all_results: list[dict[str, Any]] = await async_get_batch_entities(
            texts=data,
            model_str=model_str,
            response_model=response_model,  # type: ignore
            db=session,
            batch_size=batch_size,
        )

        return all_results


@click.command()
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=BATCH_SIZE,
    help="Number of texts to process in each batch.",
)
def get_predictions_cli(batch_size: int = BATCH_SIZE) -> list[dict[str, Any]] | dict[str, Any]:
    """Process text data to extract named entities using batch processing through CLI.

    Parameters:\n
        batch_size : int, optional
            Number of texts to process in each batch, by default 5.

    Returns:\n
        list[dict[str, Any]] | dict[str, Any]
    """
    all_results: list[dict[str, Any]] = asyncio.run(get_predictions(batch_size=batch_size))

    return all_results


if __name__ == "__main__":
    logger.info(f"🚀 Starting extraction and data upload process.\nBase_url: {BASE_URL!r}")
    logger.info(f"Environment: {ENVIRONMENT!r}")
    MODEL: ModelEnum = ModelEnum.OPENROUTER_MODEL
    file_id: str = "1q0OWcRWTX9vna34MM3xOcILheUR7iIvp"

    # Create the destination path if it doesn't exist
    create_path(DOWNLOAD_DATA_PATH)

    try:
        # asyncio.run(
        #     async_download_file_from_gdrive(
        #         file_id=file_id,
        #         destination=DOWNLOAD_DATA_PATH,
        #     )
        # )
        get_predictions_cli()
        if ENVIRONMENT == "production":
            upload_file_to_gcs(file_path=FILENAME, bucket_name=BUCKET_NAME, blob_name=None)
        logger.info("✅ Extraction and data upload complete")

    except KeyboardInterrupt:
        logger.info("❌ Process interrupted by user.")
