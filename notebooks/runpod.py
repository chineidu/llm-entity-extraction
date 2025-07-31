import asyncio
import json
import logging
import random
import time
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
    cabletv_streaming_subscriptions,
    fees,
    fuel,
    gambling_and_betting,
    health,
    internet_and_airtime,
    leisure_lifestyle_recreation,
    loan_repayment,
    misc_list,
    religious_activity,
    savings_and_investments,
    txn_reasons,
)
from .utils import (
    get_async_retrying,
    load_json_data,
)

settings = refresh_settings()
logger = create_logger(name="entity_utils", log_level=logging.DEBUG)
T = TypeVar("T", bound=BaseModel)

# ======= CLIENT CONFIGURATION ========
API_KEY: str = settings.INDICINA_API_KEY.get_secret_value()
BASE_URL: str = settings.INDICINA_URL
ENVIRONMENT: str = settings.ENVIRONMENT

# ======= INFERENCE =======
CONCURRENCY_LIMIT: int = app_config.inference.concurrency_limit
MAX_CONNECTIONS: int = app_config.inference.max_connections
MAX_TOKENS: int = app_config.inference.max_tokens
POOL_SIZE: int = app_config.inference.pool_size
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
INTERNET_AND_AIRTIME_STRING: str = "|".join(internet_and_airtime)
FEES_STRING: str = "|".join(fees)
GAMBLING_AND_BETTING_STRING: str = "|".join(gambling_and_betting)
LOAN_REPAYMENT_STRING: str = "|".join(loan_repayment)
RELIGIOUS_ACTIVITY_STRING: str = "|".join(religious_activity)
FUEL_STRING: str = "|".join(fuel)
LEISURE_LIFESTYLE_RECREATION_STRING: str = "|".join(leisure_lifestyle_recreation)
HEALTH_STRING: str = "|".join(health)
CABLETV_STREAMING_SUBSCRIPTIONS_STRING: str = "|".join(cabletv_streaming_subscriptions)
SAVINGS_AND_INVESTMENTS_STRING: str = "|".join(savings_and_investments)
TXN_REASON_STRING: str = "|".join(txn_reasons)
MISC_STRING: str = "|".join(misc_list)
NUM_ENTITY: str = app_config.prompts.num_entity
TRANSACTIONS: list[dict[str, Any]] = load_json_data(data_path=data_path, format_type="jsonl_file")

context: dict[str, Any] = {
    "THINKING_MODE": THINKING_MODE,
    "LABELS": LABELS,
    "INTERNET_AND_AIRTIME_STRING": INTERNET_AND_AIRTIME_STRING,
    "FEES_STRING": FEES_STRING,
    "GAMBLING_AND_BETTING_STRING": GAMBLING_AND_BETTING_STRING,
    "LOAN_REPAYMENT_STRING": LOAN_REPAYMENT_STRING,
    "RELIGIOUS_ACTIVITY_STRING": RELIGIOUS_ACTIVITY_STRING,
    "FUEL_STRING": FUEL_STRING,
    "LEISURE_LIFESTYLE_RECREATION_STRING": LEISURE_LIFESTYLE_RECREATION_STRING,
    "HEALTH_STRING": HEALTH_STRING,
    "CABLETV_STREAMING_SUBSCRIPTIONS_STRING": CABLETV_STREAMING_SUBSCRIPTIONS_STRING,
    "SAVINGS_AND_INVESTMENTS_STRING": SAVINGS_AND_INVESTMENTS_STRING,
    "MISC_STRING": MISC_STRING,
    "TXN_REASON_STRING": TXN_REASON_STRING,
    "NUM_ENTITY": NUM_ENTITY,
    "TRANSACTIONS": TRANSACTIONS,
}
SYSTEM_PROMPT: str = load_and_render_template(
    env=template_env, template_file=template_file, context=context
)


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
            max_tokens=MAX_TOKENS,
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
    print(f"PROMPT: {user_message}\n\n\n")

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
) -> None:
    # Initialize client manager properly
    client_manager = ClientManager(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    client = client_manager.client  # Get the client instance

    results: list[dict[str, Any]] = []
    logger.info(f"Using model: {model_str.value} | Batch size: {batch_size}")

    try:
        # Create batches of transactions (each batch contains multiple transactions)
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches for {len(texts)} transactions")
        
        # Create tasks for each BATCH of transactions
        batch_tasks = []
        for batch in batches:
            # Each task processes a BATCH of transactions
            task = async_safe_entity_extraction_request(
                data=batch,
                client=client,
                model_str=model_str,
                response_model=response_model,
            )
            batch_tasks.append(task)

        # Process all batches
        with tqdm(
            total=len(batch_tasks),
            desc=f"[🔄] Processing {len(texts)} texts in {len(batches)} batches",
            unit="batch",
        ) as pbar:
            for i, task in enumerate(asyncio.as_completed(batch_tasks)):
                start_time = time.perf_counter()
                batch_results = await task
                # print(batch_results)
                proc_time = time.perf_counter() - start_time

                # Calculate performance metrics
                num_items = len(batch_results)  # Should be batch_size (except last batch)
                input_tokens = sum(len(text[1]) // 4 for _, text in batches[i])  # Use current batch
                output_tokens = num_items * MAX_TOKENS
                total_tokens = input_tokens + output_tokens
                tokens_sec = total_tokens / proc_time if proc_time > 0 else 0

                # Save results
                bulk_insert_records(db=db, data_list=batch_results)
                results.extend(batch_results)

                # Update progress
                pbar.set_postfix({
                    "batch": f"{i+1}/{len(batches)}",
                    "tokens/s": f"{tokens_sec:.0f}",
                    "items": num_items,
                    "time": f"{proc_time:.2f}s",
                })
                pbar.update(1)

                # Small delay between batches
                await asyncio.sleep(1)
        
        logger.info(f"Processed {len(results)} transactions total")

    except Exception as e:
        logger.exception(f"Error processing batches: {str(e)}")
        return

    finally:
        await client_manager.close()  # Close properly through manager