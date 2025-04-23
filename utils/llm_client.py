"""LLM client for Anthropic models."""

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Tuple, cast
from dataclasses_json import DataClassJsonMixin
import anthropic
import openai
from anthropic import (
    NOT_GIVEN as Anthropic_NOT_GIVEN,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,  # pyright: ignore[reportPrivateImportUsage]
)
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
    ThinkingBlock as AnthropicThinkingBlock,
    RedactedThinkingBlock as AnthropicRedactedThinkingBlock,
)
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types import (
    ToolResultBlockParam as AnthropicToolResultBlockParam,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from anthropic.types.message_create_params import (
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
)

from openai import (
    APIConnectionError as OpenAI_APIConnectionError,
)
from openai import (
    InternalServerError as OpenAI_InternalServerError,
)
from openai import (
    RateLimitError as OpenAI_RateLimitError,
)
from openai._types import (
    NOT_GIVEN as OpenAI_NOT_GIVEN,  # pyright: ignore[reportPrivateImportUsage]
)

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ToolParam(DataClassJsonMixin):
    """Internal representation of LLM tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall(DataClassJsonMixin):
    """Internal representation of LLM-generated tool call."""

    tool_call_id: str
    tool_name: str
    tool_input: Any


@dataclass
class ToolResult(DataClassJsonMixin):
    """Internal representation of LLM tool result."""

    tool_call_id: str
    tool_name: str
    tool_output: Any


@dataclass
class ToolFormattedResult(DataClassJsonMixin):
    """Internal representation of formatted LLM tool result."""

    tool_call_id: str
    tool_name: str
    tool_output: str


@dataclass
class TextPrompt(DataClassJsonMixin):
    """Internal representation of user-generated text prompt."""

    text: str
    image_url: str = None


@dataclass
class TextResult(DataClassJsonMixin):
    """Internal representation of LLM-generated text result."""

    text: str


AssistantContentBlock = (
    TextResult | ToolCall | AnthropicRedactedThinkingBlock | AnthropicThinkingBlock
)
UserContentBlock = TextPrompt | ToolFormattedResult
GeneralContentBlock = UserContentBlock | AssistantContentBlock
LLMMessages = list[list[GeneralContentBlock]]


class LLMClient:
    """A client for LLM APIs for the use in agents."""

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """
        raise NotImplementedError


def recursively_remove_invoke_tag(obj):
    """Recursively remove the </invoke> tag from a dictionary or list."""
    result_obj = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result_obj[key] = recursively_remove_invoke_tag(value)
    elif isinstance(obj, list):
        result_obj = [recursively_remove_invoke_tag(item) for item in obj]
    elif isinstance(obj, str):
        if "</invoke>" in obj:
            result_obj = json.loads(obj.replace("</invoke>", ""))
        else:
            result_obj = obj
    else:
        result_obj = obj
    return result_obj


class AnthropicDirectClient(LLMClient):
    """Use Anthropic models via first party API."""

    def __init__(
        self,
        model_name="claude-3-7-sonnet-20250219",
        max_retries=2,
        use_caching=True,
        use_low_qos_server: bool = False,
        thinking_tokens: int = 0,
    ):
        """Initialize the Anthropic first party client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        # Disable retries since we are handling retries ourselves.
        self.client = anthropic.Anthropic(
            api_key=api_key, max_retries=1, timeout=60 * 5
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.prompt_caching_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        self.thinking_tokens = thinking_tokens

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """

        # Turn GeneralContentBlock into Anthropic message format
        anthropic_messages = []
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"
            message_content_list = []
            for message in message_list:
                # Check string type to avoid import issues particularly with reloads.
                if str(type(message)) == str(TextPrompt):
                    message = cast(TextPrompt, message)
                    message_content = AnthropicTextBlock(
                        type="text",
                        text=message.text,
                    )
                elif str(type(message)) == str(TextResult):
                    message = cast(TextResult, message)
                    message_content = AnthropicTextBlock(
                        type="text",
                        text=message.text,
                    )
                elif str(type(message)) == str(ToolCall):
                    message = cast(ToolCall, message)
                    message_content = AnthropicToolUseBlock(
                        type="tool_use",
                        id=message.tool_call_id,
                        name=message.tool_name,
                        input=message.tool_input,
                    )
                elif str(type(message)) == str(ToolFormattedResult):
                    message = cast(ToolFormattedResult, message)
                    message_content = AnthropicToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=message.tool_call_id,
                        content=message.tool_output,
                    )
                elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
                    message = cast(AnthropicRedactedThinkingBlock, message)
                    message_content = message
                elif str(type(message)) == str(AnthropicThinkingBlock):
                    message = cast(AnthropicThinkingBlock, message)
                    message_content = message
                else:
                    print(
                        f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                    )
                    raise ValueError(
                        f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                    )
                message_content_list.append(message_content)

            # Anthropic supports up to 4 cache breakpoints, so we put them on the last 4 messages.
            if self.use_caching and idx >= len(messages) - 4:
                if isinstance(message_content_list[-1], dict):
                    message_content_list[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    message_content_list[-1].cache_control = {"type": "ephemeral"}

            anthropic_messages.append(
                {
                    "role": role,
                    "content": message_content_list,
                }
            )

        if self.use_caching:
            extra_headers = self.prompt_caching_headers
        else:
            extra_headers = None

        # Turn tool_choice into Anthropic tool_choice format
        if tool_choice is None:
            tool_choice_param = Anthropic_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = ToolChoiceToolChoiceAny(type="any")
        elif tool_choice["type"] == "auto":
            tool_choice_param = ToolChoiceToolChoiceAuto(type="auto")
        elif tool_choice["type"] == "tool":
            tool_choice_param = ToolChoiceToolChoiceTool(
                type="tool", name=tool_choice["name"]
            )
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        if len(tools) == 0:
            tool_params = Anthropic_NOT_GIVEN
        else:
            tool_params = [
                AnthropicToolParam(
                    input_schema=tool.input_schema,
                    name=tool.name,
                    description=tool.description,
                )
                for tool in tools
            ]

        response = None

        if thinking_tokens is None:
            thinking_tokens = self.thinking_tokens
        if thinking_tokens and thinking_tokens > 0:
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_tokens}
            }
            temperature = 1
            assert max_tokens >= 32_000 and thinking_tokens <= 8192, (
                f"As a heuristic, max tokens {max_tokens} must be >= 32k and thinking tokens {thinking_tokens} must be < 8k"
            )
        else:
            extra_body = None

        for retry in range(self.max_retries):
            try:
                response = self.client.messages.create(  # type: ignore
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    model=self.model_name,
                    temperature=temperature,
                    system=system_prompt or Anthropic_NOT_GIVEN,
                    tool_choice=tool_choice_param,  # type: ignore
                    tools=tool_params,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                )
                break
            except (
                AnthropicAPIConnectionError,
                AnthropicInternalServerError,
                AnthropicRateLimitError,
                AnthropicOverloadedError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed Anthropic request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying LLM request: {retry + 1}/{self.max_retries}")
                    # Sleep 4-6 seconds with jitter to avoid thundering herd.
                    time.sleep(5 * random.uniform(0.8, 1.2))

        # Convert messages back to Augment format
        augment_messages = []
        assert response is not None
        for message in response.content:
            if "</invoke>" in str(message):
                warning_msg = "\n".join(
                    ["!" * 80, "WARNING: Unexpected 'invoke' in message", "!" * 80]
                )
                print(warning_msg)

            if str(type(message)) == str(AnthropicTextBlock):
                message = cast(AnthropicTextBlock, message)
                augment_messages.append(TextResult(text=message.text))
            elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
                augment_messages.append(message)
            elif str(type(message)) == str(AnthropicThinkingBlock):
                message = cast(AnthropicThinkingBlock, message)
                augment_messages.append(message)
            elif str(type(message)) == str(AnthropicToolUseBlock):
                message = cast(AnthropicToolUseBlock, message)
                augment_messages.append(
                    ToolCall(
                        tool_call_id=message.id,
                        tool_name=message.name,
                        tool_input=recursively_remove_invoke_tag(message.input),
                    )
                )
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", -1
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", -1
            ),
        }

        return augment_messages, message_metadata


class OpenAIDirectClient(LLMClient):
    """Use OpenAI models via first party API."""

    def __init__(self, model_name: str, max_retries=2, cot_model: bool = True):
        """Initialize the OpenAI first party client."""
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(
            api_key=api_key,
            max_retries=1,
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.cot_model = cot_model

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            system_prompt: A system prompt.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """
        assert thinking_tokens is None, "Not implemented for OpenAI"

        # Turn GeneralContentBlock into OpenAI message format
        openai_messages = []
        if system_prompt is not None:
            if self.cot_model:
                raise NotImplementedError("System prompt not supported for cot model")
            system_message = {"role": "system", "content": system_prompt}
            openai_messages.append(system_message)
        for idx, message_list in enumerate(messages):
            if len(message_list) > 1:
                raise ValueError("Only one entry per message supported for openai")
            augment_message = message_list[0]
            if str(type(augment_message)) == str(TextPrompt):
                augment_message = cast(TextPrompt, augment_message)
                message_content = {"type": "text", "text": augment_message.text}
                openai_message = {"role": "user", "content": [message_content]}
            elif str(type(augment_message)) == str(TextResult):
                augment_message = cast(TextResult, augment_message)
                message_content = {"type": "text", "text": augment_message.text}
                openai_message = {"role": "assistant", "content": [message_content]}
            elif str(type(augment_message)) == str(ToolCall):
                augment_message = cast(ToolCall, augment_message)
                tool_call = {
                    "type": "function",
                    "id": augment_message.tool_call_id,
                    "function": {
                        "name": augment_message.tool_name,
                        "arguments": augment_message.tool_input,
                    },
                }
                openai_message = {
                    "role": "assistant",
                    "tool_calls": [tool_call],
                }
            elif str(type(augment_message)) == str(ToolFormattedResult):
                augment_message = cast(ToolFormattedResult, augment_message)
                openai_message = {
                    "role": "tool",
                    "tool_call_id": augment_message.tool_call_id,
                    "content": augment_message.tool_output,
                }
            else:
                print(
                    f"Unknown message type: {type(augment_message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                )
                raise ValueError(f"Unknown message type: {type(augment_message)}")
            openai_messages.append(openai_message)

        # Turn tool_choice into OpenAI tool_choice format
        if tool_choice is None:
            tool_choice_param = OpenAI_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = "required"
        elif tool_choice["type"] == "auto":
            tool_choice_param = "auto"
        elif tool_choice["type"] == "tool":
            tool_choice_param = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        # Turn tools into OpenAI tool format
        openai_tools = []
        for tool in tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            tool_def["parameters"]["strict"] = True
            openai_tool_object = {
                "type": "function",
                "function": tool_def,
            }
            openai_tools.append(openai_tool_object)

        response = None
        for retry in range(self.max_retries):
            try:
                extra_body = {}
                openai_max_tokens = max_tokens
                openai_temperature = temperature
                if self.cot_model:
                    extra_body["max_completion_tokens"] = max_tokens
                    openai_max_tokens = OpenAI_NOT_GIVEN
                    openai_temperature = OpenAI_NOT_GIVEN

                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model_name,
                    messages=openai_messages,
                    temperature=openai_temperature,
                    tools=openai_tools if len(openai_tools) > 0 else OpenAI_NOT_GIVEN,
                    tool_choice=tool_choice_param,  # type: ignore
                    max_tokens=openai_max_tokens,
                    extra_body=extra_body,
                )
                break
            except (
                OpenAI_APIConnectionError,
                OpenAI_InternalServerError,
                OpenAI_RateLimitError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed OpenAI request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying OpenAI request: {retry + 1}/{self.max_retries}")
                    # Sleep 8-12 seconds with jitter to avoid thundering herd.
                    time.sleep(10 * random.uniform(0.8, 1.2))

        # Convert messages back to Augment format
        augment_messages = []
        assert response is not None
        openai_response_messages = response.choices
        if len(openai_response_messages) > 1:
            raise ValueError("Only one message supported for OpenAI")
        openai_response_message = openai_response_messages[0].message
        tool_calls = openai_response_message.tool_calls
        content = openai_response_message.content

        # Exactly one of tool_calls or content should be present
        if tool_calls and content:
            raise ValueError("Only one of tool_calls or content should be present")
        elif not tool_calls and not content:
            raise ValueError("Either tool_calls or content should be present")

        if tool_calls:
            if len(tool_calls) > 1:
                raise ValueError("Only one tool call supported for OpenAI")
            tool_call = tool_calls[0]
            try:
                # Parse the JSON string into a dictionary
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool arguments: {tool_call.function.arguments}")
                print(f"JSON parse error: {str(e)}")
                raise ValueError(f"Invalid JSON in tool arguments: {str(e)}") from e

            augment_messages.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    tool_input=tool_input,
                    tool_call_id=tool_call.id,
                )
            )
        elif content:
            augment_messages.append(TextResult(text=content))
        else:
            raise ValueError(f"Unknown message type: {openai_response_message}")

        assert response.usage is not None
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return augment_messages, message_metadata


class GeminiDirectClient(LLMClient):
    """Use Gemini models via OpenAI API.

    This client is designed to work with Gemini models through OpenAI's API.
    It's specifically tailored for Gemini-2.5-pro-exp and similar models.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro-exp-03-25", max_retries=2):
        """Initialize the Gemini client via OpenAI API.

        Args:
            model_name: The name of the Gemini model to use (default: gemini-2.5-pro-exp)
            max_retries: Maximum number of retries for API calls
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(
            api_key=api_key,
            max_retries=1,
        )
        self.model_name = model_name
        self.max_retries = max_retries

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses using Gemini models via OpenAI API.

        Args:
            messages: A list of messages
            max_tokens: The maximum number of tokens to generate
            system_prompt: A system prompt
            temperature: The temperature
            tools: A list of tools
            tool_choice: A tool choice
            thinking_tokens: Not used for Gemini

        Returns:
            A tuple of (generated response, metadata)
        """
        assert thinking_tokens is None, "Thinking tokens not supported for Gemini"

        # Convert messages to OpenAI format
        openai_messages = []

        # Add system prompt if provided
        if system_prompt is not None:
            system_message = {"role": "system", "content": system_prompt}
            openai_messages.append(system_message)

        # Process message lists
        for idx, message_list in enumerate(messages):
            # Handle multiple messages in a list by processing only the last one
            # This is a workaround for Gemini via OpenAI which doesn't support multiple entries per message
            # if len(message_list) > 1:
            #     print(f"Warning: Multiple entries in message list {idx}, using only the last one")

            # Use the last message in the list
            augment_message = message_list[-1]

            # Handle different message types
            if str(type(augment_message)) == str(TextPrompt):
                augment_message = cast(TextPrompt, augment_message)
                # Handle image URLs if present
                if augment_message.image_url:
                    openai_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": augment_message.text},
                            {"type": "image_url", "image_url": {"url": augment_message.image_url}}
                        ]
                    }
                else:
                    openai_message = {"role": "user", "content": augment_message.text}
                openai_messages.append(openai_message)
            elif str(type(augment_message)) == str(TextResult):
                augment_message = cast(TextResult, augment_message)
                openai_message = {"role": "assistant", "content": augment_message.text}
                openai_messages.append(openai_message)
            elif str(type(augment_message)) == str(ToolCall):
                augment_message = cast(ToolCall, augment_message)
                tool_call = {
                    "id": augment_message.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": augment_message.tool_name,
                        "arguments": json.dumps(augment_message.tool_input),
                    },
                }
                openai_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                }
                openai_messages.append(openai_message)
            elif str(type(augment_message)) == str(ToolFormattedResult):
                augment_message = cast(ToolFormattedResult, augment_message)
                openai_message = {
                    "role": "tool",
                    "tool_call_id": augment_message.tool_call_id,
                    "content": augment_message.tool_output,
                }
                openai_messages.append(openai_message)
            else:
                raise ValueError(f"Unknown message type: {type(augment_message)}")

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            openai_tool_object = {
                "type": "function",
                "function": tool_def,
            }
            openai_tools.append(openai_tool_object)

        # Handle tool_choice
        if tool_choice is None:
            tool_choice_param = OpenAI_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = "required"
        elif tool_choice["type"] == "auto":
            tool_choice_param = "auto"
        elif tool_choice["type"] == "tool":
            tool_choice_param = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        # Make API call with retries
        response = None
        for retry in range(self.max_retries):
            try:
                print(f"Making API call to Gemini model: {self.model_name}")
                # Don't print the full messages with base64 images
                sanitized_messages = []
                for msg in openai_messages:
                    if isinstance(msg, dict) and 'content' in msg and isinstance(msg['content'], list):
                        # Check for image URLs in content
                        sanitized_content = []
                        for item in msg['content']:
                            if isinstance(item, dict) and item.get('type') == 'image_url' and 'image_url' in item:
                                # Replace the base64 data with a placeholder
                                if 'url' in item['image_url'] and item['image_url']['url'].startswith('data:image/'):
                                    sanitized_item = {
                                        'type': 'image_url',
                                        'image_url': {'url': '[BASE64_IMAGE_DATA_OMITTED]'}
                                    }
                                    sanitized_content.append(sanitized_item)
                                else:
                                    sanitized_content.append(item)
                            else:
                                sanitized_content.append(item)
                        sanitized_msg = msg.copy()
                        sanitized_msg['content'] = sanitized_content
                        sanitized_messages.append(sanitized_msg)
                    else:
                        sanitized_messages.append(msg)
                print(f"Messages: {sanitized_messages}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    temperature=temperature,
                    tools=openai_tools if len(openai_tools) > 0 else OpenAI_NOT_GIVEN,
                    tool_choice=tool_choice_param,
                    max_tokens=max_tokens,
                )
                if response is None:
                    print("Warning: API call returned None response")
                    if retry == self.max_retries - 1:
                        raise ValueError("API call returned None response after all retries")
                    continue
                print(f"API call successful, received response")
                break
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                if retry == self.max_retries - 1:
                    print(f"Failed Gemini request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying Gemini request: {retry + 1}/{self.max_retries}")
                    # Sleep with jitter to avoid thundering herd
                    time.sleep(5 * random.uniform(0.8, 1.2))

        # Convert response back to Augment format
        augment_messages = []

        # Check if we have a valid response
        if response is None:
            print("Error: No response received from API")
            # Return a simple text response as fallback
            augment_messages.append(TextResult(text="I'm sorry, I couldn't process your request due to an API error. Please try again later."))
            return augment_messages, {"error": "No response from API"}

        # Get the first choice (we only support one response)
        if len(response.choices) > 1:
            print("Warning: Multiple choices returned, using only the first one")

        if len(response.choices) == 0:
            print("Error: No choices in response")
            augment_messages.append(TextResult(text="I'm sorry, I couldn't process your request due to an API error. Please try again later."))
            return augment_messages, {"error": "No choices in response"}

        openai_response_message = response.choices[0].message
        tool_calls = openai_response_message.tool_calls
        content = openai_response_message.content

        # Handle tool calls or text content
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    # Parse the JSON string into a dictionary
                    tool_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse tool arguments: {tool_call.function.arguments}")
                    raise ValueError(f"Invalid JSON in tool arguments: {str(e)}") from e

                augment_messages.append(
                    ToolCall(
                        tool_name=tool_call.function.name,
                        tool_input=tool_input,
                        tool_call_id=tool_call.id,
                    )
                )
        elif content is not None:
            augment_messages.append(TextResult(text=content))
        else:
            raise ValueError("Neither tool_calls nor content present in response")

        # Prepare metadata
        message_metadata = {"raw_response": response}

        # Add usage data if available
        if response.usage is not None:
            try:
                message_metadata["input_tokens"] = response.usage.prompt_tokens
                message_metadata["output_tokens"] = response.usage.completion_tokens
            except AttributeError as e:
                print(f"Warning: Could not access usage data: {str(e)}")
                message_metadata["input_tokens"] = -1
                message_metadata["output_tokens"] = -1
        else:
            print("Warning: No usage data in response")
            message_metadata["input_tokens"] = -1
            message_metadata["output_tokens"] = -1

        return augment_messages, message_metadata


def get_client(client_name: str, **kwargs) -> LLMClient:
    """Get a client for a given client name."""
    if client_name == "anthropic-direct":
        return AnthropicDirectClient(**kwargs)
    elif client_name == "openai-direct":
        return OpenAIDirectClient(**kwargs)
    elif client_name == "gemini-direct":
        return GeminiDirectClient(**kwargs)
    else:
        raise ValueError(f"Unknown client name: {client_name}")
