import asyncio
import json
import logging
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Literal

import anyio
import requests  # type: ignore
from google.cloud import storage  # type: ignore
from google.cloud.exceptions import Forbidden, NotFound
from tenacity import (
    AsyncRetrying,
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)
from tqdm import tqdm  # type: ignore

from ner_extraction import create_logger
from ner_extraction.config.settings import Settings, refresh_settings

logger = create_logger(name="utils", log_level=logging.INFO)
# Load Settings
SETTINGS: Settings = refresh_settings()


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and whitespace.

    Parameters
    ----------
    text : str
        Input text string to be cleaned

    Returns
    -------
    str
    """
    pattern: str = r"\b(nip)\b"

    text = re.sub(pattern, " ", text, flags=re.I)
    text = re.sub(r"[\d\W_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def create_path(path: str | Path) -> None:
    """
    Create parent directories for the given path if they don't exist.

    Parameters
    ----------
    path : str | Path
        The file path for which to create parent directories.
    """
    # Convert to Path object if it's a string
    path_obj: Path = Path(path) if isinstance(path, str) else path

    # Get the parent directory and create it if it doesn't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)


def get_batch(data: list[Any], *, batch_size: int) -> Generator[list[Any], None, None]:
    """
    Get a batch of data from a list.

    Parameters
    ----------
    data : list[Any]
        Input list containing elements to be batched.
    batch_size : int
        Size of each batch.

    Returns
    -------
    Generator[list[Any], None, None]
        Generator yielding batches of data as lists.

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5]
    >>> list(get_batch(data, batch_size=2))
    [[1, 2], [3, 4], [5]]
    """
    for idx in range(0, len(data), batch_size):
        yield data[idx : idx + batch_size]


def save_json_data(filename: str | Path, data: Any) -> None:
    """
    Save data to a JSON file.

    Parameters
    ----------
    filename : str | Path
        Path to the file where data will be saved.
    data : Any
        Data to be saved in JSON format.

    Returns
    -------
    None
    """
    if isinstance(filename, str):
        filename = Path(filename)
    create_path(filename)

    with filename.open(mode="w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {filename}", end="")


def load_json_data(
    data_path: Path | str, format_type: Literal["json_file", "jsonl_file"] = "json_file"
) -> list[Any] | Any:
    """
    Load and parse JSON data from a file.

    Parameters
    ----------
    data_path : Path | str
        Path to the JSON file to be loaded.
    format_type : Literal["json_file", "jsonl_file"]
        Type of JSON file to load. Can be either "json_file" for standard JSON
        or "jsonl_file" for JSON Lines format.

    Returns
    -------
    Parsed JSON data.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    if format_type == "json_file":
        # Load ALL the data
        with data_path.open(mode="r") as f:
            return json.load(f)

    with data_path.open(mode="r") as f:
        # Load one line at a time
        data: list[Any] = [json.loads(line) for line in f]
    return data


async def async_load_json_data(
    data_path: Path | str, format_type: Literal["json_file", "jsonl_file"] = "json_file"
) -> list[Any] | Any:
    """
    Load and parse JSON data from a file asynchronously.

    Parameters
    ----------
    data_path : Path | str
        Path to the JSON file to be loaded.
    format_type : Literal["json_file", "jsonl_file"]
        Type of JSON file to load. Can be either "json_file" for standard JSON
        or "jsonl_file" for JSON Lines format.

    Returns
    -------
    list[Any] | Any
        For "json_file", returns parsed JSON data of any type.
        For "jsonl_file", returns a list of parsed JSON objects.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    """
    # Ensure data_path is a Path object
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    if format_type == "json_file":
        # Read the file asynchronously with anyio
        async with await anyio.open_file(data_path, "r") as f:
            # Read ALL the contents of the file
            contents: str = await f.read()  # type: ignore
            # Parse the JSON data
            return json.loads(contents)

    async with await anyio.open_file(data_path, "r") as f:
        # Read one line at a time
        data: list[Any] = [json.loads(line) async for line in f]
    return data


def async_timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that measures and prints the execution time of an async function.

    Parameters
    ----------
    func : Callable
        The async function to be timed.

    Returns
    -------
    Callable
        A wrapped async function that prints execution time.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function that times the execution of the decorated function.
        """
        start_time: float = time.perf_counter()
        result = await func(*args, **kwargs)
        duration: float = time.perf_counter() - start_time
        logger.info(f"{func.__name__} executed in {duration:.2f} seconds")  # type: ignore
        return result

    return wrapper


def get_async_retrying(
    logger: logging.Logger,
    max_attempts: int = 5,
) -> AsyncRetrying:
    """
    Creates an AsyncRetrying instance with configured retry behavior and logging.
    Used as a callable function.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for recording retry attempts and failures.

    Returns
    -------
    AsyncRetrying
        Configured AsyncRetrying instance with retry policies, wait strategies,
        stop conditions and logging hooks.

    """
    STOP_AFTER_DELAY: int = 300  # seconds
    MAX_WAIT: int = 1  # seconds

    return AsyncRetrying(
        retry=retry_if_exception_type(ConnectionError),  # Retry on any exception
        wait=wait_fixed(MAX_WAIT),
        # ==== Compound stop conditions ====
        stop=(stop_after_attempt(max_attempts) | stop_after_delay(STOP_AFTER_DELAY)),
        # ==== Logging hooks ====
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.INFO),
        # ==== Add custom statistics ====
        retry_error_callback=lambda state: logger.error(
            f"Final failure after {state.attempt_number} attempts: {state.outcome.exception()}"  # type: ignore
        ),
    )


def async_retry_decorator(
    logger: logging.Logger,
    max_attempts: int = 5,
) -> Callable[[Any], Any]:
    """
    Creates a retry decorator with configured retry behavior and logging.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for recording retry attempts and failures.

    Returns
    -------
    Callable[[Any], Any]
        A retry decorator function that can be applied to other functions.
    """
    STOP_AFTER_DELAY: int = 300  # seconds
    MAX_WAIT: int = 1  # seconds

    return retry(
        wait=wait_fixed(MAX_WAIT),
        # ==== Compound stop conditions ====
        stop=(stop_after_attempt(max_attempts) | stop_after_delay(STOP_AFTER_DELAY)),
        # ==== Logging hooks ====
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.INFO),
        # ==== Add custom statistics ====
        retry_error_callback=lambda state: logger.error(
            f"Final failure after [{state.attempt_number}] attempts: {state.outcome.exception()}"  # type: ignore
        ),
    )


def download_file_from_gdrive(file_id: str | Path, destination: str | Path) -> None:
    """This is used to download files from the Google Drive

    Parameters
    ----------
    file_id : str | Path
        The ID of the file to download
    destination : str | Path
        The path to save the downloaded file
    """
    chunk_size: int = 8_192
    download_url: str = f"https://drive.google.com//uc?export=download&id={file_id}"
    response = requests.get(download_url, stream=True)
    # Raise an exception for bad status codes
    response.raise_for_status()

    # Get the total file size from headers if available
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file:
        # Create progress bar
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading file") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Downloaded file to {destination!r}")


async def async_download_file_from_gdrive(file_id: str | Path, destination: str | Path) -> None:
    """This is used to asynchronously download files from the Google Drive

    Parameters
    ----------
    file_id : str | Path
        The ID of the file to download
    destination : str | Path
        The path to save the downloaded file
    """
    return await asyncio.to_thread(download_file_from_gdrive, file_id, destination)


def upload_file_to_gcs(
    file_path: str | Path,
    bucket_name: str,
    blob_name: str | None = None,
) -> str:
    """Upload a file to Google Cloud Storage bucket with progress tracking.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to upload
    bucket_name : str
        Name of the GCS bucket
    blob_name : str, optional
        Name for the file in GCS. If None, uses the original filename.

    Returns
    -------
    str
        The public URL of the uploaded file

    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    NotFound
        If the bucket doesn't exist
    Forbidden
        If you don't have permission to upload to the bucket
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use original filename if blob_name not specified
    if blob_name is None:
        prefix: str = "ner_data"
        blob_name = f"{prefix}/{file_path.name}"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        file_size = file_path.stat().st_size

        # Upload with progress tracking
        with open(file_path, "rb") as file_obj:
            blob.upload_from_file(file_obj, rewind=True, size=file_size, checksum="md5")

        # Get the public URL
        public_uri = f"gs://{bucket_name}/{blob_name}"

        logger.info(f"Successfully uploaded {file_path.name} to {public_uri}")
        return public_uri

    except NotFound:
        logger.error(f"Bucket '{bucket_name}' not found")
        raise
    except Forbidden:
        logger.error(f"Permission denied to upload to bucket '{bucket_name}'")
        raise
    except Exception as error:
        logger.error(f"An error occurred during upload: {error}")
        raise

