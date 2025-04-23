"""Tool for indicating task completion."""

from typing import Any, Optional
from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)


class CompleteTool(LLMTool):
    name = "complete"
    """The model should call this tool when it is done with the task."""

    description = "Call this tool when you are confident in your final numeric answer. Your answer must be a number (integer or float) and will be verified for consistency before the task is completed."
    input_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "oneOf": [
                    {"type": "number"},
                    {"type": "string"}
                ],
                "description": "Your final numeric answer to the question. Must be a number (integer or float).",
            },
        },
        "required": ["answer"],
    }

    def __init__(self):
        super().__init__()
        self.answer = None  # Will store the numeric answer

    @property
    def should_stop(self):
        return self.answer is not None

    def reset(self):
        self.answer = None

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        raw_answer = tool_input.get("answer")
        assert raw_answer is not None, "Model returned empty answer"

        # Convert string answers to numeric values
        if isinstance(raw_answer, str):
            try:
                # Try to convert to int first, then float if that fails
                try:
                    numeric_answer = int(raw_answer.strip())
                except ValueError:
                    numeric_answer = float(raw_answer.strip())
                self.answer = numeric_answer
            except ValueError:
                # If conversion fails, return an error
                error_msg = f"The answer must be a number (integer or float). Received: {raw_answer}"
                return ToolImplOutput(
                    tool_output=error_msg,
                    tool_result_message=error_msg
                )
        else:
            # If it's already a number, use it directly
            self.answer = raw_answer

        # Note: The actual completion logic is now handled in the Agent class
        # to allow for answer verification before terminating
        return ToolImplOutput(
            tool_output=f"Received numeric answer: {self.answer}",
            tool_result_message=f"Received numeric answer: {self.answer}"
        )

    def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
        return ""
