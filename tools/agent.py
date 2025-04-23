from copy import deepcopy
from typing import Any, Optional
from pathlib import Path
from tools.bash_tool import create_bash_tool, create_docker_bash_tool
from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)
from utils.llm_client import LLMClient, TextResult
from utils.workspace_manager import WorkspaceManager
from tools.complete_tool import CompleteTool
from prompts.system_prompt import SYSTEM_PROMPT
from tools.str_replace_tool import StrReplaceEditorTool
from tools.sequential_thinking_tool import SequentialThinkingTool
from tools.image_tools import (
    CropTool,
    SelectTool,
    BlackoutTool,
    AddImageTool,
)
from termcolor import colored
from rich.console import Console
import logging


class Agent(LLMTool):
    name = "general_agent"
    description = """\
A general agent that can accomplish tasks and answer questions.

If you are faced with a task that involves more than a few steps, or if the task is complex, or if the instructions are very long,
try breaking down the task into smaller steps and call this tool multiple times.
"""
    input_schema = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The instruction to the agent.",
            },
        },
        "required": ["instruction"],
    }

    def _get_system_prompt(self):
        """Get the system prompt, including any pending messages.

        Returns:
            The system prompt with messages prepended if any
        """

        return SYSTEM_PROMPT.format(
            workspace_root=self.workspace_manager.root,
        )

    def __init__(
        self,
        client: LLMClient,
        workspace_manager: WorkspaceManager,
        console: Console,
        logger_for_agent_logs: logging.Logger,
        max_output_tokens_per_turn: int = 8192,
        max_turns: int = 10,
        use_prompt_budgeting: bool = True,
        ask_user_permission: bool = False,
        docker_container_id: Optional[str] = None,
    ):
        # Track the last image sent to the model
        self.last_image_path = None
        # Track the image paths that have been processed
        self.processed_images = set()
        # Track recent tool calls to detect loops
        self.recent_tool_calls = []
        self.max_recent_calls = 5
        """Initialize the agent.

        Args:
            client: The LLM client to use
            max_output_tokens_per_turn: Maximum tokens per turn
            max_turns: Maximum number of turns
            workspace_manager: Optional workspace manager for taking snapshots
        """
        super().__init__()
        self.client = client
        self.console = console
        self.logger_for_agent_logs = logger_for_agent_logs
        self.max_output_tokens = max_output_tokens_per_turn
        self.max_turns = max_turns
        self.workspace_manager = workspace_manager
        self.interrupted = False
        self.dialog = DialogMessages(
            logger_for_agent_logs=logger_for_agent_logs,
            use_prompt_budgeting=use_prompt_budgeting,
        )

        # Create and store the complete tool
        self.complete_tool = CompleteTool()

        if docker_container_id is not None:
            print(
                colored(
                    f"Enabling docker bash tool with container {docker_container_id}",
                    "blue",
                )
            )
            self.logger_for_agent_logs.info(
                f"Enabling docker bash tool with container {docker_container_id}"
            )
            bash_tool = create_docker_bash_tool(
                container=docker_container_id,
                ask_user_permission=ask_user_permission,
            )
        else:
            bash_tool = create_bash_tool(
                ask_user_permission=ask_user_permission,
            )

        self.tools = [
            bash_tool,
            StrReplaceEditorTool(workspace_manager=workspace_manager),
            SequentialThinkingTool(),
            # Image tools for VQA
            CropTool(workspace_manager=workspace_manager),
            SelectTool(workspace_manager=workspace_manager),
            BlackoutTool(workspace_manager=workspace_manager),
            AddImageTool(workspace_manager=workspace_manager),
            self.complete_tool,
        ]

        # Initialize the image manager for listing images
        from utils.image_manager import ImageManager
        self.image_manager = ImageManager(workspace_manager.root)

        # Initialize a counter for consecutive identical tool calls
        self.consecutive_identical_tool_calls = 0
        self.last_tool_name = None

    def get_image_list(self) -> str:
        """Get a formatted list of all images and views in the workspace.

        Returns:
            A string containing the list of images and views
        """
        images = self.image_manager.list_images()

        if not images:
            return "No images found in the workspace."

        result = "Images in workspace:\n"

        from utils.image_utils import get_image_size

        for img_path in images:
            size = get_image_size(img_path)
            result += f"- {img_path.name} ({size[0]}x{size[1]})\n"

            # Add views for this image
            views = self.image_manager.list_views(img_path)
            if views:
                result += "  Views:\n"
                for view_path in views:
                    view_info = self.image_manager.get_view_info(view_path)
                    result += f"  - {view_path.name} ({view_info['size'][0]}x{view_info['size'][1]}) - Coordinates: {view_info['coordinates']}\n"

        return result

    def run_impl(
        self,
        tool_input: dict[str, Any],
        _dialog_messages: Optional[DialogMessages] = None,  # Unused parameter
    ) -> ToolImplOutput:
        instruction = tool_input["instruction"]

        user_input_delimiter = "-" * 45 + " USER INPUT " + "-" * 45 + "\n" + instruction
        self.logger_for_agent_logs.info(f"\n{user_input_delimiter}\n")

        # print("Agent starting with instruction:", instruction)

        # Get the list of images
        image_list = self.get_image_list()

        # Add instruction to dialog before getting mode
        # Append the image list to the instruction
        full_instruction = f"{instruction}\n\n{image_list}"
        self.dialog.add_user_prompt(full_instruction)
        self.interrupted = False

        remaining_turns = self.max_turns
        while remaining_turns > 0:
            remaining_turns -= 1

            delimiter = "-" * 45 + " NEW TURN " + "-" * 45
            self.logger_for_agent_logs.info(f"\n{delimiter}\n")

            if self.dialog.use_prompt_budgeting:
                current_tok_count = self.dialog.count_tokens()
                self.logger_for_agent_logs.info(
                    f"(Current token count: {current_tok_count})\n"
                )

            # Get tool parameters for available tools
            tool_params = [tool.get_tool_param() for tool in self.tools]

            # Check for duplicate tool names
            tool_names = [param.name for param in tool_params]
            sorted_names = sorted(tool_names)
            for i in range(len(sorted_names) - 1):
                if sorted_names[i] == sorted_names[i + 1]:
                    raise ValueError(f"Tool {sorted_names[i]} is duplicated")

            try:
                model_response, _metadata = self.client.generate(  # Metadata is unused
                    messages=self.dialog.get_messages_for_llm_client(),
                    max_tokens=self.max_output_tokens,
                    tools=tool_params,
                    system_prompt=self._get_system_prompt(),
                )
                self.dialog.add_model_response(model_response)

                # Handle tool calls
                pending_tool_calls = self.dialog.get_pending_tool_calls()

                if len(pending_tool_calls) == 0:
                    # No tools were called, so assume the task is complete
                    self.logger_for_agent_logs.info("[no tools were called]")
                    return ToolImplOutput(
                        tool_output=self.dialog.get_last_model_text_response(),
                        tool_result_message="Task completed",
                    )

                # Log the pending tool calls
                self.logger_for_agent_logs.info(f"Pending tool calls: {[call.tool_name for call in pending_tool_calls]}")

                # Handle multiple tool calls per turn (for Gemini model)
                for tool_call in pending_tool_calls:
                    text_results = [
                        item for item in model_response if isinstance(item, TextResult)
                    ]
                    if len(text_results) > 0:
                        text_result = text_results[0]
                        self.logger_for_agent_logs.info(
                            f"Top-level agent planning next step: {text_result.text}\n",
                        )

                    # Check for repetitive tool calls (loop detection)
                    self.recent_tool_calls.append(tool_call.tool_name)
                    if len(self.recent_tool_calls) > self.max_recent_calls:
                        self.recent_tool_calls.pop(0)  # Remove oldest call

                    # If we've called the same tool multiple times in a row, try a different approach
                    if len(self.recent_tool_calls) >= 3 and \
                       all(call == self.recent_tool_calls[0] for call in self.recent_tool_calls):
                        print(f"Detected a loop of {len(self.recent_tool_calls)} identical tool calls: {tool_call.tool_name}")
                        print("Breaking out of the loop by forcing a different tool call...")

                        # If we're stuck in a select_image loop, try crop_image instead
                        if tool_call.tool_name == "select_image":
                            # Create a crop of the image to break the loop
                            crop_tool = next((t for t in self.tools if t.name == "crop_image"), None)
                            if crop_tool and self.workspace_manager.list_images():
                                first_image = self.workspace_manager.list_images()[0]
                                result = crop_tool.run_impl({
                                    "image_path": str(first_image),
                                    "view_id": "region_1",
                                    "x1": 0,
                                    "y1": 0,
                                    "x2": 1000,  # Use a reasonable default size
                                    "y2": 1000
                                })
                                tool_result = result.tool_output
                                self.dialog.add_tool_call_result(tool_call, tool_result)
                                # Reset the loop detection
                                self.recent_tool_calls = []
                                continue

                        # Reset the loop detection
                        self.recent_tool_calls = []

                    # Special handling for list_images tool which has been removed
                    if tool_call.tool_name == "list_images":
                        # Instead of raising an error, just return the image list
                        print("Handling list_images tool call with direct image list")
                        image_list = self.get_image_list()
                        tool_result = image_list
                        self.dialog.add_tool_call_result(tool_call, tool_result)
                        # Reset the loop detection
                        self.recent_tool_calls = []
                        continue

                    try:
                        tool = next(t for t in self.tools if t.name == tool_call.tool_name)
                    except StopIteration as exc:
                        raise ValueError(
                            f"Tool with name {tool_call.tool_name} not found"
                        ) from exc

                    try:
                        result = tool.run(tool_call.tool_input, deepcopy(self.dialog))

                        tool_input_str = "\n".join(
                            [f" - {k}: {v}" for k, v in tool_call.tool_input.items()]
                        )
                        log_message = f"Calling tool {tool_call.tool_name} with input:\n{tool_input_str}"
                        log_message += f"\nTool output: \n{result}\n\n"
                        self.logger_for_agent_logs.info(log_message)

                        # Handle both ToolResult objects and tuples
                        if isinstance(result, tuple):
                            tool_result, _ = result
                        else:
                            tool_result = result

                        # Handle different tool types and send appropriate images
                        from utils.llm_client import TextPrompt
                        import re
                        from pathlib import Path

                        # Extract image path from tool result if present

                        # For select_image tool
                        if tool_call.tool_name == "select_image":
                            # Extract the image path
                            path_match = re.search(r"Selected image at ([^\n]+)", tool_result)
                            if path_match:
                                image_path = Path(path_match.group(1))
                                self.last_image_path = image_path

                            # Check if we have aux_data with the image URL
                            if hasattr(result, 'aux_data') and result.aux_data and 'image_url' in result.aux_data:
                                image_url = result.aux_data['image_url']
                                # Get the list of images
                                image_list = self.get_image_list()

                                # Create a new user prompt with the image
                                prompt = TextPrompt(text=f"I'm analyzing this image.\n\n{image_list}")
                                prompt.image_url = image_url
                                # Add the image to the dialog
                                self.dialog._message_lists.append([prompt])
                                # Now add the tool result
                                self.dialog.add_tool_call_result(tool_call, tool_result)
                            else:
                                # Just add the regular tool result
                                self.dialog.add_tool_call_result(tool_call, tool_result)

                        # For crop_image tool
                        elif tool_call.tool_name == "crop_image":
                            # Extract the new view path
                            path_match = re.search(r"Created new view at ([^\n]+)", tool_result)
                            if path_match:
                                view_path = Path(path_match.group(1))
                                self.last_image_path = view_path

                                # Load the image and send it to the model
                                try:
                                    from PIL import Image
                                    import base64
                                    from io import BytesIO

                                    # Load and resize the image if needed
                                    img = Image.open(view_path)
                                    max_size = (1500, 1500)
                                    if img.width > max_size[0] or img.height > max_size[1]:
                                        ratio = min(max_size[0] / img.width, max_size[1] / img.height)
                                        new_size = (int(img.width * ratio), int(img.height * ratio))
                                        img = img.resize(new_size, Image.LANCZOS)

                                    # Convert to base64
                                    buffered = BytesIO()
                                    img.save(buffered, format="PNG")
                                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    img_url = f"data:image/png;base64,{img_base64}"
                                    # Log a message without the full base64 string
                                    print(f"Created image URL for cropped view (base64 data omitted)")

                                    # Get the list of images
                                    image_list = self.get_image_list()

                                    # Create a new user prompt with the image
                                    prompt = TextPrompt(text=f"Here's the cropped view I requested. Let me analyze it.\n\n{image_list}")
                                    prompt.image_url = img_url
                                    # Add the image to the dialog
                                    self.dialog._message_lists.append([prompt])
                                except Exception as e:
                                    print(f"Error processing cropped image: {str(e)}")

                            # Add the tool result
                            self.dialog.add_tool_call_result(tool_call, tool_result)

                        # For blackout_image tool
                        elif tool_call.tool_name == "blackout_image":
                            # Extract the blacked out view path
                            path_match = re.search(r"Blacked out (view|image) at ([^\n]+)", tool_result)
                            if path_match:
                                view_path = Path(path_match.group(2))
                                self.last_image_path = view_path

                                # Load the image and send it to the model
                                try:
                                    from PIL import Image
                                    import base64
                                    from io import BytesIO

                                    # Load and resize the image if needed
                                    img = Image.open(view_path)
                                    max_size = (1500, 1500)
                                    if img.width > max_size[0] or img.height > max_size[1]:
                                        ratio = min(max_size[0] / img.width, max_size[1] / img.height)
                                        new_size = (int(img.width * ratio), int(img.height * ratio))
                                        img = img.resize(new_size, Image.LANCZOS)

                                    # Convert to base64
                                    buffered = BytesIO()
                                    img.save(buffered, format="PNG")
                                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    img_url = f"data:image/png;base64,{img_base64}"
                                    # Log a message without the full base64 string
                                    print(f"Created image URL for blacked out view (base64 data omitted)")

                                    # Get the list of images
                                    image_list = self.get_image_list()

                                    # Create a new user prompt with the image
                                    prompt = TextPrompt(text=f"I've blacked out this region. Here's the updated image.\n\n{image_list}")
                                    prompt.image_url = img_url
                                    # Add the image to the dialog
                                    self.dialog._message_lists.append([prompt])
                                except Exception as e:
                                    print(f"Error processing blacked out image: {str(e)}")

                            # Add the tool result
                            self.dialog.add_tool_call_result(tool_call, tool_result)

                        # For all other tools
                        else:
                            # Just add the tool result
                            self.dialog.add_tool_call_result(tool_call, tool_result)

                        if self.complete_tool.should_stop:
                            # Add a fake model response, so the next turn is the user's
                            # turn in case they want to resume
                            self.dialog.add_model_response(
                                [TextResult(text="Completed the task.")]
                            )
                            return ToolImplOutput(
                                tool_output=self.complete_tool.answer,
                                tool_result_message="Task completed",
                            )
                    except KeyboardInterrupt:
                        # Handle interruption during tool execution
                        self.interrupted = True
                        interrupt_message = "Tool execution was interrupted by user."
                        self.dialog.add_tool_call_result(tool_call, interrupt_message)
                        self.dialog.add_model_response(
                            [
                                TextResult(
                                    text="Tool execution interrupted by user. You can resume by providing a new instruction."
                                )
                            ]
                        )
                        return ToolImplOutput(
                            tool_output=interrupt_message,
                            tool_result_message=interrupt_message,
                        )

            except KeyboardInterrupt:
                # Handle interruption during model generation or other operations
                self.interrupted = True
                self.dialog.add_model_response(
                    [
                        TextResult(
                            text="Agent interrupted by user. You can resume by providing a new instruction."
                        )
                    ]
                )
                return ToolImplOutput(
                    tool_output="Agent interrupted by user",
                    tool_result_message="Agent interrupted by user",
                )

        agent_answer = "Agent did not complete after max turns"
        return ToolImplOutput(
            tool_output=agent_answer, tool_result_message=agent_answer
        )

    def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
        return f"Agent started with instruction: {tool_input['instruction']}"

    def run_agent(
        self,
        instruction: str,
        resume: bool = False,
        orientation_instruction: str | None = None,
        initial_image_path: Path | None = None,
    ) -> str:
        """Start a new agent run.

        Args:
            instruction: The instruction to the agent.
            resume: Whether to resume the agent from the previous state,
                continuing the dialog.
            orientation_instruction: Optional orientation instruction.
            initial_image_path: Optional path to an initial image to include with the first message.

        Returns:
            A tuple of (result, message).
        """
        self.complete_tool.reset()
        if resume:
            assert self.dialog.is_user_turn()
        else:
            self.dialog.clear()
            self.interrupted = False
            self.last_image_path = None
            self.processed_images.clear()

        # If an initial image is provided, include it with the first message
        if initial_image_path is not None and not resume:
            try:
                from utils.llm_client import TextPrompt
                from PIL import Image
                import base64
                from io import BytesIO

                # Load and resize the image if needed
                img = Image.open(initial_image_path)
                max_size = (1500, 1500)
                if img.width > max_size[0] or img.height > max_size[1]:
                    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)

                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_url = f"data:image/png;base64,{img_base64}"
                # Log a message without the full base64 string
                print(f"Created image URL for initial image (base64 data omitted)")

                # Get the list of images
                image_list = self.get_image_list()

                # Create a prompt with the image and instruction
                prompt = TextPrompt(text=f"{instruction}\n\n{image_list}")
                prompt.image_url = img_url

                # Add the prompt to the dialog
                self.dialog.add_user_prompt("")
                self.dialog._message_lists[-1] = [prompt]

                # Store the image path
                self.last_image_path = initial_image_path
                self.processed_images.add(str(initial_image_path))

                # Return early since we've already added the instruction with the image
                tool_input = {"instruction": ""}
            except Exception as e:
                print(f"Error including initial image: {str(e)}")
                # Fall back to normal instruction without image
                tool_input = {"instruction": instruction}
        else:
            # Normal instruction without image
            tool_input = {"instruction": instruction}

        if orientation_instruction:
            tool_input["orientation_instruction"] = orientation_instruction

        return self.run(tool_input, self.dialog)

    def clear(self):
        self.dialog.clear()
        self.interrupted = False
        self.last_image_path = None
        self.processed_images.clear()
        self.recent_tool_calls.clear()
