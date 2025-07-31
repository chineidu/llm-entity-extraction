import json
import re
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from openai import AsyncOpenAI  # type: ignore
from pydantic import BaseModel, SecretStr, validate_call

T = TypeVar("T", bound=BaseModel)
SYSTEM_MESSAGE: str = """
<system>
/no_think

<role>
You are a data extraction expert. Extract information from the provided text and return ONLY a 
valid JSON object that matches this exact schema:

<schema>
{json_schema}
</schema>
</role>

<guidelines>
- Return only valid JSON - no explanations, markdown, or additional text
- Extract data precisely as it appears in the source text
- Do not include fields not present in the schema
- For missing required fields, use these defaults:
  * Numbers: 0
  * Strings: null
  * Booleans: false
  * Arrays: []
  * Objects: {{}}
- Preserve original data types and formatting where possible
- If text contains ambiguous information, choose the most likely interpretation
</guidelines>

<validation>
Response must:
- Be parseable by json.loads
- Contain only fields defined in the schema
- Use correct data types for each field
- Have no trailing commas or syntax errors
</validation>

</system>
"""

SYSTEM_MESSAGE_VLLM: str = """
<system>
/no_think

<role>
You're an AI assistant that helps people extract relevant information.
</role>

<validation>
Response must:
- Not contain made up words or phrases
</validation>

</system>
"""


def _clean_response_text_single_regex(text: str) -> str:
    """
    Clean response text by removing XML-like tags and backticks using regex pattern.

    Parameters
    ----------
    text : str
        Input text containing XML-like tags and backticks to be cleaned.

    Returns
    -------
    str
        Cleaned text with XML-like tags and backticks removed.
    """
    pattern: str = r"<think>.*?</think>|`+json|`+"
    cleaned_text: str = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text.strip()


@dataclass
class LLMResponse:
    """Class for handling LLM API responses.

    Parameters
    ----------
    api_key : SecretStr
        The API key for authentication.
    base_url : str
        The base URL for the API endpoint.
    model : str
        The name of the LLM model to use.
    """

    api_key: SecretStr
    base_url: str
    model: str
    use_vllm: bool = False

    def _get_client(self) -> AsyncOpenAI:
        """Get an instance of the OpenAI client."""
        return AsyncOpenAI(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_url,
            max_retries=3,
            timeout=180,  # type: ignore
        )

    @validate_call
    async def ainvoke(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, Type[T]] | tuple[None, dict[str, str]]:
        """Asynchronously invoke the LLM API with the given messages.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries containing role and content.

        Returns
        -------
        tuple[str, Type[T]] | tuple[None, dict[str, str]]
            A tuple containing either:
            - (content, raw_response)
            - (None, error_info)

        """
        try:
            aclient: AsyncOpenAI = self._get_client()

            raw_response = await aclient.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,  # type: ignore
                temperature=0,
                seed=42,
            )

            content = _clean_response_text_single_regex(raw_response.choices[0].message.content)  # type: ignore
            return (content, raw_response)  # type: ignore

        except Exception as e:
            return (None, {"status": "error", "error": str(e)})  # type: ignore

    @validate_call
    async def get_structured_response(
        self, message: str, response_model: Type[T]
    ) -> tuple[Type[T], Type[T]] | tuple[None, dict[str, str]]:
        """Get structured response from OpenAI API.

        Parameters
        ----------
            message : str
                The user message to send to the API.
            response_model : Type[T]
                The Pydantic model class to validate the response.

        Returns
        -------
        A tuple containing either:
        - (structured_output, raw_response)
        - (None, error_info)
        """
        try:
            aclient: AsyncOpenAI = self._get_client()
            json_schema: dict[str, str] = response_model.model_json_schema()
            if not self.use_vllm:
                raw_response: Type[T] = await aclient.chat.completions.create(  # type: ignore
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_MESSAGE.format(json_schema=json_schema),
                        },
                        {"role": "user", "content": message},
                    ],
                    response_format={"type": "json_schema", "schema": json_schema, "strict": True},
                    temperature=0,
                    seed=42,
                )

                _value = _clean_response_text_single_regex(raw_response.choices[0].message.content)  # type: ignore
                structured_output: Type[T] = response_model.model_validate(json.loads(_value))  # type: ignore
                return (structured_output, raw_response)  # type: ignore

            raw_response = await aclient.chat.completions.create(  # type: ignore
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE_VLLM},
                    {"role": "user", "content": message},
                ],
                extra_body={"enable_thinking": False, "guided_json": json_schema},
                temperature=0,
                seed=42,
            )

            _value = _clean_response_text_single_regex(raw_response.choices[0].message.content)  # type: ignore
            structured_output = response_model.model_validate(json.loads(_value))  # type: ignore
            return (structured_output, raw_response)  # type: ignore

        except Exception as e:
            return (
                None,
                {"status": "error", "error": str(e)},
            )


@validate_call
def convert_openai_messages_to_string(messages: list[dict[str, Any]]) -> str:
    """
    Convert a list of OpenAI messages to a formatted string representation.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        List of OpenAI message dictionaries containing 'role' and 'content' keys.

    Returns
    -------
    str
        A formatted string with each message's role and content on separate lines.
    """
    msgs: list[str] = [f"\nRole: {msg['role']}\nContent: {msg['content']}" for msg in messages]
    return "\n".join(msgs)
