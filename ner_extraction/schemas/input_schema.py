from enum import Enum
from typing import Annotated

from pydantic import (  # type: ignore
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StringConstraints,
)
from pydantic.alias_generators import to_camel


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Returns:
        float: Rounded value.
    """
    if isinstance(value, float):
        return round(value, 2)
    return value


Float = Annotated[float, BeforeValidator(round_probability)]
String = Annotated[
    str,
    StringConstraints(strip_whitespace=True, strict=True, min_length=1, max_length=80),
]


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


class ModelEnum(str, Enum):
    """Enumeration of available model endpoints and their costs."""

    # Change this when new models are added
    INDICINA_QWEN = "/data/Qwen/Qwen3-8B"
    RUNPOD_MODEL = "indicinaaa/Qwen3-4B-unsloth-bnb-4bit-fp16"
    OPENROUTER_MODEL = "google/gemini-2.0-flash-001"  # "google/gemini-2.5-flash-preview-05-20"

    # For testing (remove later)
    QWEN_3p0_8B_REMOTE = "qwen/qwen3-8b"  # $0.035/1M tokens
    MISTRAL_NEMO_REMOTE = "mistralai/mistral-nemo"  # $0.01/1M tokens
    LLAMA_3p2_3B_INSTRUCT_REMOTE = "meta-llama/llama-3.2-3b-instruct"  # $0.01/1M tokens
    LLAMA_3p1_8B_INSTRUCT_REMOTE = "meta-llama/llama-3.1-8b-instruct"  # $0.02/1M tokens


class EntityType(str, Enum):
    RELIGIOUS_ACTIVITY = "religiousActivity"
    LEVIES_AND_CHARGES = "leviesAndCharges"
    BETTING_OR_GAMBLING = "bettingOrGambling"
    CABLE_TV_OR_STREAMING = "cableTvOrStreaming"
    LOAN_REPAYMENT = "loanRepayment"
    SAVINGS_AND_INVESTMENTS = "savingsAndInvestments"
    PENSIONS = "pensions"
    HEALTH_ACTIVITY = "healthActivity"
    INTERNET_AND_TELECOM = "topUpOrData"
    PERSON = "person"
    BANKING_ORGANIZATION = "bankOrFinancialOrganization"
    ORGANIZATION = "organizationOrEnterprise"
    LOCATION = "location"
