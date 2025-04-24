from copy import deepcopy
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import base64
from PIL import Image
from rich.console import Console

from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)
from utils.llm_client import LLMClient, TextResult, TextPrompt
from utils.workspace_manager import WorkspaceManager
from utils.run_logger import RunLogger
from utils.text_cleaner import clean_tool_result
from tools.complete_tool import CompleteTool
from prompts.instruction import SYSTEM_PROMPT
from tools.sequential_thinking_tool import SequentialThinkingTool
from tools.image_tools import (
    CropTool,
    SwitchImageTool,
    BlackoutTool,
)


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
        log_dir: Optional[str] = "agent_runs",
    ):
        """Initialize the agent.

        Args:
            client: The LLM client to use
            workspace_manager: Workspace manager for file operations
            console: Rich console for output
            logger_for_agent_logs: Logger for agent logs
            max_output_tokens_per_turn: Maximum tokens per turn
            max_turns: Maximum number of turns
            use_prompt_budgeting: Whether to use prompt budgeting
            log_dir: Directory for detailed run logs
        """
        super().__init__()
        self.client = client
        self.console = console
        self.logger_for_agent_logs = logger_for_agent_logs
        self.max_output_tokens = max_output_tokens_per_turn
        self.max_turns = max_turns
        self.workspace_manager = workspace_manager
        self.interrupted = False

        # Initialize the run logger
        self.run_logger = None  # Will be initialized for each run
        self.log_dir = log_dir

        # Track the last image sent to the model
        self.last_image_path = None
        # Track the original image path
        self.original_image_path = None
        # Track the current view coordinates
        self.current_view_coordinates = None
        # Track the image paths that have been processed
        self.processed_images = set()
        self.max_recent_calls = 5

        self.dialog = DialogMessages(
            logger_for_agent_logs=logger_for_agent_logs,
            use_prompt_budgeting=use_prompt_budgeting,
        )

        # Create and store the complete tool
        self.complete_tool = CompleteTool()

        self.tools = [
            SequentialThinkingTool(),
            # Image tools for VQA
            CropTool(workspace_manager=workspace_manager),
            SwitchImageTool(workspace_manager=workspace_manager),
            BlackoutTool(workspace_manager=workspace_manager),
            self.complete_tool,
        ]

        # We're using the workspace manager for image management
        # No need to initialize a separate image manager


    def get_image_list(self, simplified: bool = False) -> str:
        """Get a formatted list of all images and views in the workspace.

        Args:
            simplified: Whether to return a simplified list without paths and coordinates

        Returns:
            A string containing the list of images and views
        """
        images = self.workspace_manager.list_images()

        if not images:
            return "No images found in the workspace."

        if simplified:
            # Simplified version for the VLM - just basic info
            result = "Available images:\n"

            from utils.image_utils import get_image_size

            for img_path in images:
                size = get_image_size(img_path)
                result += f"- {img_path.name} ({size[0]}x{size[1]})\n"

                views = self.workspace_manager.list_views(img_path)
                if views:
                    result += "  Views:\n"
                    for view_path in views:
                        view_info = self.workspace_manager.get_view_info(view_path)
                        # Just include the view name and size, not the full path or coordinates
                        result += f"  - {view_path.name} ({view_info['size'][0]}x{view_info['size'][1]})\n"
        else:
            # Detailed version for logging
            result = "Images in workspace:\n"

            from utils.image_utils import get_image_size

            for img_path in images:
                size = get_image_size(img_path)
                result += f"- {img_path.name} ({size[0]}x{size[1]})\n"

                # Add views for this image
                views = self.workspace_manager.list_views(img_path)
                if views:
                    result += "  Views:\n"
                    for view_path in views:
                        view_info = self.workspace_manager.get_view_info(view_path)
                        result += f"  - {view_path.name} ({view_info['size'][0]}x{view_info['size'][1]}) - Coordinates: {view_info['coordinates']}\n"

        return result

    def run_impl(
        self,
        tool_input: dict[str, Any],
        _dialog_messages: Optional[DialogMessages] = None,  # Unused parameter
    ) -> ToolImplOutput:
        instruction = tool_input["instruction"]

        self.logger_for_agent_logs.info("\n" + "-" * 45 + " USER INPUT " + "-" * 45 + "\n" + instruction + "\n")

        # Get the simplified list of images for the VLM
        image_list = self.get_image_list(simplified=True)

        # Add instruction to dialog before getting mode
        # Append the image list to the instruction
        full_instruction = f"{instruction}\n\n{image_list}"
        self.dialog.add_user_prompt(full_instruction)
        self.interrupted = False

        remaining_turns = self.max_turns
        while remaining_turns > 0:
            remaining_turns -= 1

            self.logger_for_agent_logs.info("\n" + "-" * 45 + " NEW TURN " + "-" * 45 +"\n")

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
                # Get the messages to send to the model
                messages = self.dialog.get_messages_for_llm_client()

                # Log the current view coordinates before calling the model
                self.logger_for_agent_logs.info(f"IMPORTANT: Before calling model - dialog.current_view_coordinates: {self.dialog.current_view_coordinates}")

                # Check if the message contains an image
                for message_list in messages:
                    for message in message_list:
                        if hasattr(message, 'image_url') and message.image_url:
                            self.logger_for_agent_logs.info("IMPORTANT: Message contains an image URL")
                            # We can't log the full URL as it's too large, but we can log that it exists

                # Call the model
                model_response, metadata = self.client.generate(
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    tools=tool_params,
                    system_prompt=self._get_system_prompt(),
                )
                self.dialog.add_model_response(model_response)

                # Log the model response
                if self.run_logger:
                    # Extract text from model response
                    text_results = [item for item in model_response if isinstance(item, TextResult)]
                    if text_results:
                        response_text = text_results[0].text
                        self.run_logger.log_model_response(response_text)

                    # Log the MLLM call with all details
                    images = [self.last_image_path] if self.last_image_path else []
                    self.run_logger.log_mllm_call(
                        messages=messages,
                        model_response=model_response,
                        metadata=metadata,
                        images=images
                    )

                # Handle tool calls
                pending_tool_calls = self.dialog.get_pending_tool_calls()

                # Check for tool calls in code blocks if no explicit tool calls were made
                if len(pending_tool_calls) == 0:
                    # Extract tool calls from code blocks in the model's text response
                    text_results = [item for item in model_response if isinstance(item, TextResult)]
                    if text_results:
                        code_block_tool_calls = self._extract_tool_calls_from_code_blocks(text_results[0].text)
                        if code_block_tool_calls:
                            pending_tool_calls = code_block_tool_calls

                if len(pending_tool_calls) == 0:
                    # No tools were called, so default to blackout tool if the original image is not all black
                    if self.last_image_path:
                        try:
                            from PIL import Image
                            import numpy as np

                            # Check if the image is all black
                            img = Image.open(self.last_image_path)
                            img_array = np.array(img)
                            if np.mean(img_array) < 1.0:  # Image is essentially all black
                                # If image is all black, complete the task
                                self.logger_for_agent_logs.info("[image is all black, completing task]")
                                return ToolImplOutput(
                                    tool_output=self.dialog.get_last_model_text_response(),
                                    tool_result_message="Task completed",
                                )
                            else:
                                # If image is not all black, use blackout tool
                                self.logger_for_agent_logs.info("[no tools were called, defaulting to blackout tool]")
                                blackout_tool = next(t for t in self.tools if t.name == "blackout_image")
                                tool_input = {"image_path": str(self.last_image_path)}
                                result = blackout_tool.run(tool_input, deepcopy(self.dialog))
                                return ToolImplOutput(
                                    tool_output=result,
                                    tool_result_message="Applied blackout tool as default action",
                                )
                        except Exception as e:
                            self.logger_for_agent_logs.info(f"[error checking image: {str(e)}]")

                    # No image or error occurred, so assume the task is complete
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

                        # Log the tool call and result
                        if self.run_logger:
                            self.run_logger.log_tool_call(tool_call.tool_name, tool_call.tool_input)

                        # Handle both ToolResult objects and tuples
                        if isinstance(result, tuple):
                            tool_result, _ = result
                        else:
                            tool_result = result

                        # Handle different tool types and send appropriate images
                        # Extract image path from tool result if present

                        # For switch_image tool
                        if tool_call.tool_name == "switch_image":
                            # Extract the image path
                            path_match = re.search(r"Switched to image at ([^\n]+)", tool_result)
                            if path_match:
                                image_path = Path(path_match.group(1))
                                self.last_image_path = image_path

                            # Check if we have aux_data with the image URL
                            if hasattr(result, 'aux_data') and result.aux_data and 'image_url' in result.aux_data:
                                image_url = result.aux_data['image_url']
                                # Get the simplified list of images for the VLM
                                image_list = self.get_image_list(simplified=True)

                                # Create a new user prompt with the image
                                prompt = TextPrompt(text="I'm analyzing this image.")
                                prompt.image_url = image_url
                                # Add the image to the dialog
                                self.dialog._message_lists.append([prompt])
                                # Clean the tool result for the VLM
                                cleaned_result = clean_tool_result(tool_call.tool_name, tool_result, self.workspace_manager.root)

                                # Now add the cleaned tool result
                                self.dialog.add_tool_call_result(tool_call, cleaned_result)

                                # Log the tool result with the image
                                if self.run_logger and self.last_image_path:
                                    self.run_logger.log_tool_result(tool_result, image_path=self.last_image_path)
                            else:
                                # Clean the tool result for the VLM
                                cleaned_result = clean_tool_result(tool_call.tool_name, tool_result, self.workspace_manager.root)

                                # Just add the cleaned tool result
                                self.dialog.add_tool_call_result(tool_call, cleaned_result)

                                # Log the tool result with the image if available
                                if self.run_logger:
                                    if self.last_image_path:
                                        self.run_logger.log_tool_result(tool_result, image_path=self.last_image_path)
                                    else:
                                        self.run_logger.log_tool_result(tool_result)

                        # For crop_image tool
                        elif tool_call.tool_name == "crop_image":
                            # Extract the new view path
                            path_match = re.search(r"Created new view at ([^\n]+)", tool_result)
                            if path_match:
                                view_path = Path(path_match.group(1))
                                self.last_image_path = view_path

                                # Extract view coordinates from the tool result
                                coords_match = re.search(r"Coordinates: \[(.*?)\]", tool_result)
                                if coords_match:
                                    try:
                                        coords_str = coords_match.group(1)
                                        coords = [int(x.strip()) for x in coords_str.split(",")]
                                        if len(coords) == 4:
                                            # Update the dialog's current view coordinates
                                            self.dialog.current_view_coordinates = coords
                                            print(f"Updated dialog.current_view_coordinates to {coords} from tool result")
                                    except Exception as e:
                                        print(f"Error parsing coordinates from tool result: {str(e)}")

                                # Load the image and send it to the model
                                try:
                                    # Use imports from the top of the file

                                    # Load the original image
                                    if hasattr(self, 'original_image_path') and self.original_image_path:
                                        img = Image.open(self.original_image_path)
                                    else:
                                        # Fall back to the dialog's original image path
                                        img = Image.open(self.dialog.original_image_path)

                                    # Crop according to the current view coordinates
                                    if self.dialog.current_view_coordinates:
                                        x1, y1, x2, y2 = self.dialog.current_view_coordinates
                                        # Crop the image
                                        cropped_img = img.crop((x1, y1, x2, y2))
                                        print(f"Cropped image with coordinates: {self.dialog.current_view_coordinates}")
                                    else:
                                        # Use the full image
                                        cropped_img = img
                                        print("No view coordinates set, using full image")

                                    # Crop to remove black regions
                                    from utils.image_utils import crop_to_remove_black_regions
                                    cropped_img = crop_to_remove_black_regions(cropped_img)

                                    # Resize if needed
                                    max_size = (1500, 1500)
                                    if cropped_img.width > max_size[0] or cropped_img.height > max_size[1]:
                                        ratio = min(max_size[0] / cropped_img.width, max_size[1] / cropped_img.height)
                                        new_size = (int(cropped_img.width * ratio), int(cropped_img.height * ratio))
                                        cropped_img = cropped_img.resize(new_size, Image.LANCZOS)

                                    # Convert to base64
                                    buffered = BytesIO()
                                    cropped_img.save(buffered, format="PNG")
                                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    img_url = f"data:image/png;base64,{img_base64}"
                                    # Log a message without the full base64 string
                                    print("Created image URL for cropped view (base64 data omitted)")

                                    # Get the simplified list of images for the VLM
                                    image_list = self.get_image_list(simplified=True)

                                    # Create a new user prompt with the image
                                    prompt = TextPrompt(text="Here's the cropped view I requested. Let me analyze it.")
                                    prompt.image_url = img_url
                                    # Add the image to the dialog
                                    self.dialog._message_lists.append([prompt])
                                except Exception as e:
                                    print(f"Error processing cropped image: {str(e)}")

                            # Clean the tool result for the VLM
                            cleaned_result = clean_tool_result(tool_call.tool_name, tool_result, self.workspace_manager.root)

                            # Add the cleaned tool result
                            self.dialog.add_tool_call_result(tool_call, cleaned_result)

                            # Log the tool result with the image if available
                            if self.run_logger:
                                if self.last_image_path:
                                    self.run_logger.log_tool_result(tool_result, image_path=self.last_image_path)
                                else:
                                    self.run_logger.log_tool_result(tool_result)

                        # For blackout_image tool
                        elif tool_call.tool_name == "blackout_image":
                            # Extract the blacked out view path
                            path_match = re.search(r"Blacked out (view|image) at ([^\n]+)", tool_result)
                            if path_match:
                                view_path = Path(path_match.group(2))
                                self.last_image_path = view_path

                                # Extract coordinates from the tool result if available
                                coords_match = re.search(r"coordinates \[(.*?)\]", tool_result)
                                if coords_match:
                                    try:
                                        coords_str = coords_match.group(1)
                                        coords = [int(x.strip()) for x in coords_str.split(",")]
                                        if len(coords) == 4:
                                            # Update the dialog's current view coordinates
                                            # Note: For blackout, we don't update the view coordinates
                                            # as we're just marking a region as analyzed
                                            print(f"Blackout applied to region at coordinates {coords}")
                                    except Exception as e:
                                        print(f"Error parsing coordinates from tool result: {str(e)}")

                                # Load the image and send it to the model
                                try:
                                    # Use imports from the top of the file

                                    # Load the original image with blackout applied
                                    img = Image.open(view_path)

                                    # Crop according to the current view coordinates
                                    if self.dialog.current_view_coordinates:
                                        x1, y1, x2, y2 = self.dialog.current_view_coordinates
                                        # Crop the image
                                        cropped_img = img.crop((x1, y1, x2, y2))
                                        print(f"Cropped image with coordinates: {self.dialog.current_view_coordinates}")
                                    else:
                                        # Use the full image
                                        cropped_img = img
                                        print("No view coordinates set, using full image")

                                    # Crop to remove black regions
                                    from utils.image_utils import crop_to_remove_black_regions
                                    cropped_img = crop_to_remove_black_regions(cropped_img)

                                    # Resize if needed
                                    max_size = (1500, 1500)
                                    if cropped_img.width > max_size[0] or cropped_img.height > max_size[1]:
                                        ratio = min(max_size[0] / cropped_img.width, max_size[1] / cropped_img.height)
                                        new_size = (int(cropped_img.width * ratio), int(cropped_img.height * ratio))
                                        cropped_img = cropped_img.resize(new_size, Image.LANCZOS)

                                    # Convert to base64
                                    buffered = BytesIO()
                                    cropped_img.save(buffered, format="PNG")
                                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    img_url = f"data:image/png;base64,{img_base64}"
                                    # Log a message without the full base64 string
                                    print("Created image URL for blacked out view (base64 data omitted)")

                                    # Get the simplified list of images for the VLM
                                    image_list = self.get_image_list(simplified=True)

                                    # Create a new user prompt with the image
                                    prompt = TextPrompt(text="I've blacked out this region. Here's the updated image, cropped to focus on the remaining non-black areas.")
                                    prompt.image_url = img_url
                                    # Add the image to the dialog
                                    self.dialog._message_lists.append([prompt])
                                except Exception as e:
                                    print(f"Error processing blacked out image: {str(e)}")

                            # Clean the tool result for the VLM
                            cleaned_result = clean_tool_result(tool_call.tool_name, tool_result, self.workspace_manager.root)

                            # Add the cleaned tool result
                            self.dialog.add_tool_call_result(tool_call, cleaned_result)

                            # Log the tool result with the image if available
                            if self.run_logger:
                                if self.last_image_path:
                                    self.run_logger.log_tool_result(tool_result, image_path=self.last_image_path)
                                else:
                                    self.run_logger.log_tool_result(tool_result)

                        # For all other tools
                        else:
                            # For other tools, we don't need to clean the result
                            self.dialog.add_tool_call_result(tool_call, tool_result)

                            # Log the tool result
                            if self.run_logger:
                                self.run_logger.log_tool_result(tool_result)

                        # Handle the complete tool specially
                        if tool_call.tool_name == "complete":
                            # Store the answer but don't immediately terminate
                            answer = tool_call.tool_input.get("answer", "")
                            self.logger_for_agent_logs.info(f"Received final answer: {answer}")

                            # Verify the answer consistency
                            if self.verify_answer_consistency(answer):
                                # Answer is consistent, proceed with termination
                                self.logger_for_agent_logs.info("Answer verified as consistent, completing task.")
                                # Add a fake model response, so the next turn is the user's
                                # turn in case they want to resume
                                self.dialog.add_model_response(
                                    [TextResult(text="Completed the task.")]
                                )
                                return ToolImplOutput(
                                    tool_output=answer,
                                    tool_result_message="Task completed with verified answer",
                                )
                            else:
                                # Answer is not consistent, revert the complete tool call
                                self.logger_for_agent_logs.info("Answer not consistent, continuing with analysis.")
                                # Reset the complete tool
                                self.complete_tool.reset()

                                # Create a new set of tools excluding the complete tool
                                tools_without_complete = [tool for tool in self.tools if tool.name != "complete"]

                                # Add a message to the dialog asking for more analysis
                                self.dialog.add_tool_call_result(
                                    tool_call,
                                    "I need you to analyze further before providing a final numeric answer. Remember that your final answer must be a number (integer or float). Please continue your analysis."
                                )

                                # Get messages for the model
                                messages = self.dialog.get_messages_for_llm_client()

                                # Get tool parameters for available tools (excluding complete)
                                tool_params = [tool.get_tool_param() for tool in tools_without_complete]

                                # Call the model again without the complete tool
                                try:
                                    model_response, metadata = self.client.generate(
                                        messages=messages,
                                        max_tokens=self.max_output_tokens,
                                        tools=tool_params,
                                        system_prompt=self._get_system_prompt(),
                                    )
                                    self.dialog.add_model_response(model_response)

                                    # Continue with the next turn
                                    continue
                                except Exception as e:
                                    self.logger_for_agent_logs.info(f"Error generating response after reverting complete tool: {str(e)}")

                        # For other tools that might set should_stop
                        elif self.complete_tool.should_stop:
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
        run_id: Optional[str] = None,
    ) -> str:
        """Start a new agent run.

        Args:
            instruction: The instruction to the agent.
            resume: Whether to resume the agent from the previous state,
                continuing the dialog.
            orientation_instruction: Optional orientation instruction.
            initial_image_path: Optional path to an initial image to include with the first message.
            run_id: Optional run ID for logging.

        Returns:
            A tuple of (result, message).
        """
        # Initialize the run logger for this run
        self.run_logger = RunLogger(base_log_dir=self.log_dir, console=self.console, run_id=run_id)

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
                # Use imports from the top of the file

                # Load the image
                img = Image.open(initial_image_path)

                # Crop to remove black regions
                from utils.image_utils import crop_to_remove_black_regions
                img = crop_to_remove_black_regions(img)

                # Resize if needed
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
                print("Created image URL for initial image (base64 data omitted)")

                # Get the simplified list of images for the VLM
                image_list = self.get_image_list(simplified=True)

                # Create a prompt with the image, instruction, and image list
                prompt = TextPrompt(text=f"{instruction}\n\n{image_list}")
                prompt.image_url = img_url

                # Add the prompt to the dialog and set the initial question
                self.dialog.add_user_prompt(instruction)
                self.dialog._message_lists[-1] = [prompt]

                # Make sure the initial question is set
                if self.dialog.initial_question is None:
                    self.dialog.initial_question = instruction

                # Store the image path
                self.last_image_path = initial_image_path
                self.original_image_path = initial_image_path
                self.processed_images.add(str(initial_image_path))

                # Set the image path in the dialog
                self.dialog.original_image_path = initial_image_path

                # Log the user message with the image
                self.run_logger.log_user_message(instruction, image_path=initial_image_path)

                # Return early since we've already added the instruction with the image
                tool_input = {"instruction": ""}
            except Exception as e:
                print(f"Error including initial image: {str(e)}")
                # Fall back to normal instruction without image
                tool_input = {"instruction": instruction}
                # Log the user message without image
                self.run_logger.log_user_message(instruction)
        else:
            # Normal instruction without image
            tool_input = {"instruction": instruction}
            # Log the user message without image
            self.run_logger.log_user_message(instruction)

        if orientation_instruction:
            tool_input["orientation_instruction"] = orientation_instruction

        result = self.run(tool_input, self.dialog)

        # Finalize the run logger
        if self.run_logger:
            self.run_logger.finalize_run()

        return result

    def verify_answer_consistency(self, answer) -> bool:
        """Verify the consistency of a numeric answer by making additional calls to the model.

        Args:
            answer: The numeric answer to verify (int or float)

        Returns:
            True if the answer is consistent across multiple runs, False otherwise
        """
        self.logger_for_agent_logs.info(f"Verifying numeric answer consistency: {answer}")

        # Create a copy of the dialog for verification
        verification_dialog = deepcopy(self.dialog)

        # Create a verification prompt that includes the original question but excludes image tools
        verification_prompt = "Based on the information provided so far, what is your final numeric answer to the question? Please provide only a number (integer or float)."
        verification_dialog.add_user_prompt(verification_prompt)

        # Create a report answer tool (simplified version of complete tool)
        report_answer_tool = CompleteTool()

        # Track verification answers
        verification_answers = []
        consistent_count = 0
        required_consistent_answers = 2  # Need 2 more consistent answers

        # Make verification calls
        for i in range(3):  # Try up to 3 times to get consistent answers
            try:
                # Get messages for the model
                messages = verification_dialog.get_messages_for_llm_client()

                # Call the model with only the report answer tool
                model_response, _ = self.client.generate(
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    tools=[report_answer_tool.get_tool_param()],
                    system_prompt=self._get_system_prompt(),
                )

                # Add the response to the verification dialog
                verification_dialog.add_model_response(model_response)

                # Check if the model called the report answer tool
                pending_tool_calls = verification_dialog.get_pending_tool_calls()

                if pending_tool_calls and pending_tool_calls[0].tool_name == "complete":
                    verification_answer = pending_tool_calls[0].tool_input.get("answer", "")
                    verification_answers.append(verification_answer)

                    # Log the verification answer
                    self.logger_for_agent_logs.info(f"Verification answer {i+1}: {verification_answer}")

                    # Check if this answer is consistent with the original
                    if self._answers_are_consistent(answer, verification_answer):
                        consistent_count += 1
                        if consistent_count >= required_consistent_answers:
                            self.logger_for_agent_logs.info("Answer verified as consistent!")
                            return True

                    # Add a fake tool result to continue the dialog
                    verification_dialog.add_tool_call_result(
                        pending_tool_calls[0],
                        "Thank you for your answer. Let's verify once more."
                    )
                else:
                    # If the model didn't call the tool, extract text response if any
                    text_results = [item for item in model_response if isinstance(item, TextResult)]
                    if text_results:
                        verification_answer = text_results[0].text
                        verification_answers.append(verification_answer)

                        # Log the verification answer
                        self.logger_for_agent_logs.info(f"Verification answer {i+1} (text): {verification_answer}")

                        # Check if this answer is consistent with the original
                        if self._answers_are_consistent(answer, verification_answer):
                            consistent_count += 1
                            if consistent_count >= required_consistent_answers:
                                self.logger_for_agent_logs.info("Answer verified as consistent!")
                                return True

                    # Add a prompt to continue verification
                    verification_dialog.add_user_prompt("Please provide your final numeric answer using the complete tool. Your answer must be a number (integer or float).")
            except Exception as e:
                self.logger_for_agent_logs.info(f"Error during verification: {str(e)}")

        # If we get here, we didn't get enough consistent answers
        self.logger_for_agent_logs.info("Answer verification failed. Answers were not consistent.")
        return False

    def _answers_are_consistent(self, answer1, answer2) -> bool:
        """Check if two numeric answers are consistent.

        This implementation handles numeric comparisons with appropriate tolerance for floating-point values.

        Args:
            answer1: First answer (numeric or string representation of a number)
            answer2: Second answer (numeric or string representation of a number)

        Returns:
            True if the answers are consistent, False otherwise
        """
        # Convert string answers to numeric values if needed
        try:
            num1 = self._convert_to_numeric(answer1)
            num2 = self._convert_to_numeric(answer2)

            # If both are integers, they should match exactly
            if isinstance(num1, int) and isinstance(num2, int):
                return num1 == num2

            # For floating point values, use a relative tolerance
            # This handles cases where the same calculation might have small floating-point differences
            rel_tol = 1e-9  # Relative tolerance for floating-point comparison
            abs_tol = 1e-9  # Absolute tolerance for values close to zero

            return abs(num1 - num2) <= max(rel_tol * max(abs(num1), abs(num2)), abs_tol)
        except (ValueError, TypeError):
            # If conversion fails, fall back to string comparison
            self.logger_for_agent_logs.info("Numeric comparison failed, falling back to string comparison")
            if isinstance(answer1, str) and isinstance(answer2, str):
                return answer1.strip() == answer2.strip()
            return str(answer1).strip() == str(answer2).strip()

    def _convert_to_numeric(self, value):
        """Convert a value to a numeric type (int or float).

        Args:
            value: The value to convert (can be a string, int, or float)

        Returns:
            The numeric value as int or float

        Raises:
            ValueError: If the value cannot be converted to a number
        """
        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            value = value.strip()
            try:
                # Try to convert to int first
                return int(value)
            except ValueError:
                # If that fails, try float
                return float(value)

        raise ValueError(f"Cannot convert {value} to a numeric value")

    def _extract_tool_calls_from_code_blocks(self, text: str) -> list:
        """Extract tool calls from code blocks in the model's text response.

        Args:
            text: The model's text response

        Returns:
            A list of ToolCallParameters objects
        """
        from utils.common import ToolCallParameters
        import re

        # Find all code blocks with tool_code tag
        code_block_pattern = r"```tool_code\s*(.*?)\s*```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)

        if not code_blocks:
            return []

        tool_calls = []

        for code_block in code_blocks:
            # Extract function name and arguments
            try:
                # Parse the function call
                function_pattern = r"(\w+)\((.*)\)"
                match = re.match(function_pattern, code_block.strip())

                if match:
                    function_name = match.group(1)
                    args_str = match.group(2)

                    # Parse the arguments
                    tool_input = {}

                    # Handle different argument formats
                    if function_name == "crop_image" and "bbox=" in args_str:
                        # Handle bbox parameter for crop_image
                        bbox_match = re.search(r"bbox=\[(.*?)\]", args_str)
                        if bbox_match:
                            bbox_values = [int(x.strip()) for x in bbox_match.group(1).split(",")]
                            if len(bbox_values) == 4:
                                # Keep the bbox parameter as expected by the tool
                                tool_input = {
                                    "bbox": bbox_values
                                }

                                # If there's an image_path parameter, extract it
                                image_path_match = re.search(r"image_path=['\"]?(.*?)['\"]?(?:,|\))", args_str)
                                if image_path_match:
                                    tool_input["image_path"] = image_path_match.group(1)
                    else:
                        # For other functions, try to parse the arguments as a Python dict
                        try:
                            # Add curly braces to make it a valid dict string
                            dict_str = "{" + args_str + "}"
                            # Replace any single quotes with double quotes for JSON compatibility
                            dict_str = dict_str.replace("'", '"')
                            # Parse the dict
                            import json
                            tool_input = json.loads(dict_str)
                        except json.JSONDecodeError:
                            # If that fails, try to parse individual key-value pairs
                            for pair in args_str.split(","):
                                if "=" in pair:
                                    key, value = pair.split("=", 1)
                                    key = key.strip()
                                    value = value.strip()

                                    # Try to convert value to appropriate type
                                    try:
                                        # Remove quotes if present
                                        if (value.startswith('"') and value.endswith('"')) or \
                                           (value.startswith("'") and value.endswith("'")):
                                            value = value[1:-1]
                                        # Try to convert to int or float if appropriate
                                        elif value.isdigit():
                                            value = int(value)
                                        elif value.replace(".", "", 1).isdigit():
                                            value = float(value)
                                    except:
                                        pass

                                    tool_input[key] = value

                    # Create a ToolCallParameters object
                    tool_call = ToolCallParameters(
                        tool_call_id=f"code_block_{len(tool_calls)}",
                        tool_name=function_name,
                        tool_input=tool_input
                    )

                    tool_calls.append(tool_call)
            except Exception as e:
                self.logger_for_agent_logs.info(f"Error parsing code block: {str(e)}")

        return tool_calls

    def clear(self):
        self.dialog.clear()
        self.interrupted = False
        self.last_image_path = None
        self.processed_images.clear()
        self.recent_tool_calls.clear()
