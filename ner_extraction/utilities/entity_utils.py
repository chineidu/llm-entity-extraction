import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Type, TypeVar

import httpx
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tqdm import tqdm  # type: ignore

from ner_extraction import PACKAGE_ROOT, create_logger
from ner_extraction.config.settings import refresh_settings
from ner_extraction.schemas.ner_models import bulk_insert_records

from ..config import app_config
from ..schemas import (
    EntitySchemaResponse,
    EntityType,
    ModelEnum,
    ResponseGenerator,
)
from .jinja_utils import (
    Environment,
    load_and_render_template,
    setup_jinja_environment,
)
from .keywords import (
    AIRTIME_OR_DATA,
    banking_organizations,
    cabletv_streaming,
    fees,
    gambling_or_betting,
    health,
    loan_repayment,
    location,
    organization,
    pensions,
    person,
    religious_activity,
    savings_and_investments,
)
from .utils import (
    get_async_retrying,
    get_batch,
    load_json_data,
)

settings = refresh_settings()
logger = create_logger(name="entity_utils", log_level=logging.DEBUG)
T = TypeVar("T", bound=BaseModel)

# ======= CLIENT CONFIGURATION ========
API_KEY: str = settings.OPENROUTER_API_KEY.get_secret_value()
BASE_URL: str = settings.OPENROUTER_URL
ENVIRONMENT: str = settings.ENVIRONMENT

# ======= INFERENCE =======
CONCURRENCY_LIMIT: int = app_config.inference.concurrency_limit
MAX_CONNECTIONS: int = app_config.inference.max_connections
MINI_BATCH_SIZE: int = app_config.inference.mini_batch_size
USE_VLLM: bool = app_config.inference.use_vllm

# ======= MODEL CONFIGURATION =======
TEMPERATURE: float = app_config.inference.model.temperature
SEED: int = app_config.inference.model.seed

search_path: Path = PACKAGE_ROOT / "prompts"
data_path: Path = PACKAGE_ROOT / app_config.data.data_path
template_file: str = app_config.prompts.prompt_path

# ======= PROMPT CONFIGURATION =======
THINKING_MODE: bool = app_config.prompts.thinking_mode
template_env: Environment = setup_jinja_environment(searchpath=search_path)
LABELS: str = [entity.value for entity in EntityType]  # type: ignore
NUM_ENTITY: str = app_config.prompts.num_entity
TRANSACTIONS: list[dict[str, Any]] = load_json_data(data_path=data_path, format_type="jsonl_file")

RELIGIOUS_ACTIVITY_STRING: str = "|".join(religious_activity)
FEES_STRING: str = "|".join(fees)
GAMBLING_AND_BETTING_STRING: str = "|".join(gambling_or_betting)
CABLETV_STREAMING_SUBSCRIPTIONS_STRING: str = "|".join(cabletv_streaming)
LOAN_REPAYMENT_STRING: str = "|".join(loan_repayment)
SAVINGS_AND_INVESTMENTS_STRING: str = "|".join(savings_and_investments)
PENSIONS_STRING: str = "|".join(pensions)
HEALTH_STRING: str = "|".join(health)
TOPUP_OR_DATA_STRING: str = "|".join(AIRTIME_OR_DATA)
BANKING_ORGANIZATION_STRING: str = "|".join(banking_organizations)
PERSON_STRING: str = "|".join(person)
ORGANIZATION_STRING: str = "|".join(organization)
LOCATION_STRING: str = "|".join(location)

context: dict[str, Any] = {
    "THINKING_MODE": THINKING_MODE,
    "LABELS": LABELS,
    "NUM_ENTITY": NUM_ENTITY,
    "TRANSACTIONS": TRANSACTIONS,
    # Keywords
    "RELIGIOUS_ACTIVITY_STRING": RELIGIOUS_ACTIVITY_STRING,
    "FEES_STRING": FEES_STRING,
    "GAMBLING_AND_BETTING_STRING": GAMBLING_AND_BETTING_STRING,
    "CABLETV_STREAMING_SUBSCRIPTIONS_STRING": CABLETV_STREAMING_SUBSCRIPTIONS_STRING,
    "LOAN_REPAYMENT_STRING": LOAN_REPAYMENT_STRING,
    "SAVINGS_AND_INVESTMENTS_STRING": SAVINGS_AND_INVESTMENTS_STRING,
    "PENSIONS_STRING": PENSIONS_STRING,
    "HEALTH_STRING": HEALTH_STRING,
    "TOPUP_OR_DATA_STRING": TOPUP_OR_DATA_STRING,
    "BANKING_ORGANIZATION_STRING": BANKING_ORGANIZATION_STRING,
    "PERSON_STRING": PERSON_STRING,
    "ORGANIZATION_STRING": ORGANIZATION_STRING,
    "LOCATION_STRING": LOCATION_STRING,
}
SYSTEM_PROMPT: str = load_and_render_template(env=template_env, template_file=template_file, context=context)

concurrency_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
retry_policy = get_async_retrying(logger=logger, max_attempts=5)


def _base_response(data: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Returns a base response with empty text for each input."""
    return [
        EntitySchemaResponse(**{"id": str(text[0]), "text": text[1]}).model_dump()  # type: ignore
        for text in data
    ]


class ClientManager:
    """
    A client manager for handling AsyncOpenAI client connections.

    Parameters
    ----------
    base_url : str
        The base URL for the API endpoint.
    api_key : str
        The authentication API key.

    Attributes
    ----------
    client : AsyncOpenAI
        The AsyncOpenAI client instance.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=MAX_CONNECTIONS,
                ),
                timeout=httpx.Timeout(120.0),
            ),
        )

    async def close(self) -> None:
        """
        Close the underlying HTTP client connection.

        Returns
        -------
        None
        """
        await self.client._client.aclose()  # noqa: SLF001


async def _make_request_vllm(
    data: list[tuple[str, str]],
    client: AsyncOpenAI,
    model_str: ModelEnum,
    response_model: Type[T],
) -> list[dict[str, Any]]:
    model: str = model_str.value
    user_message: str = "\n".join([f"<txn> {row} </txn>" for row in data])
    json_schema: dict[str, str] = response_model.model_json_schema()

    try:
        response = await client.chat.completions.create(  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            extra_body={
                "enable_thinking": THINKING_MODE,
                "guided_json": json_schema,
                # Add request ID for better tracking
                "request_id": f"req-{random.randint(10000, 99999)}",
            },
            temperature=TEMPERATURE,
            seed=SEED,
        )
        content = response.choices[0].message.content
        if response == "" or content is None:
            return _base_response(data=data)
        return json.loads(content)["data"]
    except Exception as e:
        logger.warning(f"⚠️ An exception occurred: {e}")
        raise ConnectionError(f"Wrapped as retryable: {e}") from e


async def _make_request_non_vllm(
    data: list[tuple[str, str]],
    client: AsyncOpenAI,
    model_str: ModelEnum,
    response_model: Type[T],  # noqa: ARG001
) -> list[dict[str, Any]]:
    model: str = model_str.value
    user_message: str = "\n".join([f"<txn> {row} </txn>" for row in data])

    try:
        # For non-vllm inference
        client = instructor.from_openai(client, mode=instructor.Mode.JSON)  # type: ignore
        response: Type[T] = await client.chat.completions.create(  # type: ignore
            model=model,
            response_model=ResponseGenerator,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            seed=SEED,
            # max_tokens=MAX_TOKENS,
            stream=True,
        )
        if response is None:
            return _base_response(data=data)
        return [result.model_dump() async for result in response]  # type: ignore

    except Exception as e:
        logger.warning(f"⚠️ An exception occurred: {e}")
        raise ConnectionError(f"Wrapped as retryable: {e}") from e


async def async_safe_entity_extraction_request(
    data: list[tuple[str, str]],
    client: AsyncOpenAI,
    model_str: ModelEnum,
    response_model: Type[T],
    use_vllm: bool = USE_VLLM,
) -> list[dict[str, Any]]:
    """Improved version with client pool"""
    try:
        async for attempt in retry_policy:
            with attempt:
                if use_vllm:
                    return await _make_request_vllm(data, client, model_str, response_model)
                return await _make_request_non_vllm(data, client, model_str, response_model)

    except Exception as e:
        logger.error(f"❌ Final failure after retries: {e}")
        return _base_response(data=data)

    return _base_response(data=data)


async def async_get_batch_entities(
    texts: list[tuple[str, str]],
    model_str: ModelEnum,
    response_model: Type[T],
    db: Session,
    batch_size: int = 20,
) -> list[dict[str, Any]] | None:
    """
    Processes multiple texts in parallel to extract entities using the specified model.

    Parameters
    ----------
    texts : list[tuple[str, str]]
        List of text strings to process for entity extraction
    model_str : ModelEnum
        Enum specifying which model to use for processing
    response_model : Type[T]
        The model to use for the response.
    db : Session
        The database session object
    batch_size : int, optional
        Number of texts to process in each batch, by default 20

    Returns
    -------
    list[dict[str, Any]] | None
    """

    client_manager = ClientManager(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    client = client_manager.client

    results: list[dict[str, Any]] = []
    print(f" Using model: {model_str.value!r}")

    try:
        total_batches: int = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts), batch_size):
            batch_num: int = (batch_idx // batch_size) + 1
            batch: list[tuple[str, str]] = texts[batch_idx : batch_idx + batch_size]
            mini_batches = list(get_batch(data=batch, batch_size=MINI_BATCH_SIZE))

            tasks = [
                async_safe_entity_extraction_request(
                    data=data,
                    client=client,
                    model_str=model_str,
                    response_model=response_model,
                )
                for data in mini_batches
            ]

            with tqdm(
                total=len(tasks),
                desc=f"[🔄 Batch {batch_num}/{total_batches}] Processing texts",
                unit="text",
            ) as pbar:
                batch_results: list[dict[str, Any]] = []

                for coro in asyncio.as_completed(tasks):
                    result: dict[str, Any] = await coro  # type: ignore
                    batch_results.extend(result)  # type: ignore
                    pbar.update(1)

            # Save intermediate results
            bulk_insert_records(db=db, data_list=batch_results)
            results.extend(batch_results)

            # Small delay between batches to prevent overwhelming
            await asyncio.sleep(0.1)

        return results

    except Exception as e:
        logger.error(f"Error processing batches: {e}")
        return None

    finally:
        await client_manager.close()
