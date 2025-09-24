# import asyncio
import json
import re
from pathlib import Path
from typing import Annotated, Any, Type

import instructor
import polars as pl  # type: ignore
from openai import AsyncOpenAI
from pydantic import BaseModel, BeforeValidator, Field
from rich.console import Console
from rich.theme import Theme

from ner_extraction.config.settings import refresh_settings
from ner_extraction.schemas import ModelEnum
from ner_extraction.utilities.utils import async_timer  # type: ignore
from notebooks.llm_utils import LLMResponse

custom_theme = Theme(
    {
        "white": "#FFFFFF",  # Bright white
        "info": "#00FF00",  # Bright green
        "warning": "#FFD700",  # Bright gold
        "error": "#FF1493",  # Deep pink
        "success": "#00FFFF",  # Cyan
        "highlight": "#FF4500",  # Orange-red
    }
)
console = Console(theme=custom_theme)
settings = refresh_settings()


def string_constraints(value: str) -> str:
    """Apply custom string formatting."""

    return value.strip().title()


ConstrainedStr = Annotated[str, BeforeValidator(string_constraints)]


class Person(BaseModel):
    name: ConstrainedStr = Field(description="The name of the person")
    role: ConstrainedStr = Field(description="The role of the person")
    salary: float = Field(description="The yearly salary of the person")
    organisation: ConstrainedStr = Field(description="The organisation the person works for")


class Persons(BaseModel):
    persons: list[Person]


json_schema = Persons.model_json_schema()

# model = "/data/indicinaaa/Qwen3-8B-unsloth-bnb-4bit-fp16"
model = ModelEnum.OPENROUTER_MODEL
aclient = AsyncOpenAI(
    api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
    base_url=settings.OPENROUTER_URL,
)


@async_timer
async def get_response(message: str) -> tuple[Type[BaseModel], Any] | tuple[dict[str, str], Any]:
    """Get response from OpenAI API."""
    try:
        client = instructor.from_openai(aclient, mode=instructor.Mode.JSON)  # type: ignore
        response, completions = await client.chat.completions.create_with_completion(  # type: ignore
            model=model,
            response_model=Persons,
            messages=[
                {
                    "role": "system",
                    "content": "You're an AI assistant that helps people find information.",
                },
                {"role": "user", "content": message},
            ],
            extra_body={"enable_thinking": False},
            temperature=0,
            seed=42,
            max_retries=3,
        )

        return response, completions  # type: ignore

    except Exception as e:
        return ({"status": "error", "error": str(e)}, None)


@async_timer
async def get_response_base(message: str) -> Type[BaseModel] | dict[str, str]:
    """Get response from OpenAI API."""

    try:
        return await aclient.chat.completions.create(  # type: ignore
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You're an AI assistant that helps people find information.",
                },
                {"role": "user", "content": message},
            ],
            seed=42,
            temperature=0,
            extra_body={"enable_thinking": False, "guided_json": json_schema},
        )

    except Exception as e:
        print(e)
        return {"status": "error", "error": str(e)}


@async_timer
async def get_custom_response(message: str) -> tuple[Type[BaseModel], Type[BaseModel]]:
    """Get response from OpenAI API."""
    llm = LLMResponse(
        api_key=settings.INDICINA_API_KEY,  # type: ignore
        base_url=settings.INDICINA_URL,
        model=model,
        use_vllm=True,
    )
    structured_output, raw_response = await llm.get_structured_response(message=message, response_model=Persons)
    return structured_output, raw_response  # type: ignore


if __name__ == "__main__":
    message: str = (
        "Michael Scott is a software engineer at Anery Limited. He has 4 years "
        "of experience. He currently earns $100,000 a year. James Kayode has been "
        "an instructor at the Centre For Artificial Intelligence for the last 5 years."
        "At 27 years old, Adaugo is a student at the University of Lagos who believes in "
        "fighting for justice and equality. Kareem just released a new album called 'The Journey'. "
        "It has sold over 17 thousands copies in the first day of release. "
    )
    # response = asyncio.run(get_response_base(message))
    # result = response.choices[0].message.content
    # console.print(Persons.model_validate(json.loads(result)))
    # print("----" * 50)
    # print()

    # response, completions = asyncio.run(get_response(message))
    # console.print(response)


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and converting to uppercase.

    Parameters
    ----------
    text : str
        Input text string to be cleaned

    Returns
    -------
    str
    """
    patterns: str = r"\b(nip)\b"
    regex_pattern: str = rf"[\d\W\s_]+|{patterns}"
    return re.sub(regex_pattern, " ", text, flags=re.I).strip()


def remove_invalid_characters(filepath: str, output_path: str) -> None:
    """Remove invalid characters from a CSV file and save the cleaned data."""
    df: pl.DataFrame = pl.read_csv(filepath, infer_schema=False)
    df = df.with_columns(pl.col("text").map_elements(clean_text, return_dtype=pl.Utf8).alias("text"))
    df.write_csv(output_path)


def process_data(fp: str, sp: Path) -> None:
    """
    Process CSV data and write to JSONL file.

    Parameters
    ----------
    fp : str
        Path to input CSV file.
    sp : Path
        Path to output JSONL file.

    Returns
    -------
    None
        Writes processed data to JSONL file.
    """
    try:
        df: pl.DataFrame = pl.read_csv(fp, truncate_ragged_lines=True).drop(["id", "analysisId", "createdAt"])
    except Exception:
        df = pl.read_csv(fp, truncate_ragged_lines=True).drop(["analysisId"])

    print(df.head())

    # Error handling for JSON decoding
    try:
        df = df.with_columns(pl.col("entities").str.json_decode().alias("entities"))
    except Exception as e:
        print(f"Error decoding JSON in entities column: {e}")
        # Check for problematic JSON entries
        entities_col = df.select("entities").to_series()
        for i, entity_str in enumerate(entities_col):
            if entity_str and not entity_str.strip().startswith("["):
                print(f"Row {i}: Non-JSON entities value: {entity_str[:100]}...")
        raise

    # Drop duplicates
    df = df.unique(["text"])

    df = df.with_columns(
        pl.int_range(0, len(df)).alias("id"),
    ).select(["id", "text", "entities"])
    print(df.head())

    new: list[dict[str, Any]] = df.to_dicts()

    with sp.open("w") as f:
        for item in new:
            f.write(json.dumps(item) + "\n")


# Input file paths
fp: str = "data/data/*.csv" 
# fp: str = "./data/results.csv"
# sp: Path = Path("data/data/pred_data.jsonl")
sp: Path = Path("./data/location_data.jsonl")


# Process the data
# remove_invalid_characters("./data/results.csv", output_path="./data/results_cleaned.csv")
process_data(fp, sp)
