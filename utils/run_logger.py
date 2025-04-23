"""Run Logger for Agent.

This module provides a structured logging system for agent runs, creating a dedicated
folder for each run with human-friendly logs and images.
"""

import base64
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image
from rich.console import Console

class RunLogger:
    """Logger for agent runs that creates a dedicated folder with human-friendly logs and images."""

    def __init__(
        self,
        base_log_dir: Union[str, Path] = "agent_runs",
        console: Optional[Console] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize the run logger.

        Args:
            base_log_dir: Base directory for all run logs
            console: Rich console for output
            run_id: Optional run ID, if not provided a timestamp will be used
        """
        self.base_log_dir = Path(base_log_dir)
        self.console = console or Console()

        # Create a unique run ID if not provided
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create the run directory
        self.run_dir = self.base_log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.conversations_dir = self.run_dir / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(f"agent_run_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)

        # File handler for detailed logs
        log_file = self.run_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Initialize conversation log
        self.conversation_log = []
        self.turn_counter = 0
        self.image_counter = 0

        # Create index.html file
        self._create_index_html()

        self.logger.info(f"Started new agent run: {self.run_id}")
        self.console.print(f"[bold green]Started new agent run:[/] {self.run_id}")
        self.console.print(f"[bold green]Log directory:[/] {self.run_dir}")

    def _create_index_html(self):
        """Create the initial index.html file for the run."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Agent Run: {self.run_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .turn {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .user-message {{
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .model-message {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .tool-call {{
            background-color: #fff8e1;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 3px solid #ffc107;
        }}
        .tool-result {{
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 3px solid #4caf50;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            margin: 10px 0;
        }}
        .image-container {{
            text-align: center;
            margin: 15px 0;
        }}
        .image-container img {{
            max-height: 400px;
        }}
        .metadata {{
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Agent Run: {self.run_id}</h1>
        <p>Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div id="conversation">
            <!-- Conversation turns will be added here -->
        </div>
    </div>
</body>
</html>
"""
        with open(self.run_dir / "index.html", "w") as f:
            f.write(html_content)

    def _update_html(self):
        """Update the index.html file with the current conversation."""
        html_content = ""
        for turn in self.conversation_log:
            turn_html = f'<div class="turn" id="turn-{turn["turn_id"]}">\n'
            turn_html += f'<h2>Turn {turn["turn_id"]}</h2>\n'

            # User message
            if "user_message" in turn:
                turn_html += '<div class="user-message">\n'
                turn_html += '<h3>User Message</h3>\n'

                # Check if there's an image in the user message
                if "image_path" in turn:
                    relative_path = os.path.relpath(turn["image_path"], self.run_dir)
                    turn_html += f'<div class="image-container">\n'
                    turn_html += f'<img src="{relative_path}" alt="User provided image">\n'
                    turn_html += f'<div class="metadata">Image: {os.path.basename(turn["image_path"])}</div>\n'
                    turn_html += '</div>\n'

                # Format the message text
                message_text = turn["user_message"].replace("\n", "<br>")
                turn_html += f'<p>{message_text}</p>\n'
                turn_html += '</div>\n'

            # Model response
            if "model_response" in turn:
                turn_html += '<div class="model-message">\n'
                turn_html += '<h3>Model Response</h3>\n'
                message_text = turn["model_response"].replace("\n", "<br>")
                turn_html += f'<p>{message_text}</p>\n'
                turn_html += '</div>\n'

            # Tool calls
            if "tool_calls" in turn:
                for i, tool_call in enumerate(turn["tool_calls"]):
                    turn_html += '<div class="tool-call">\n'
                    turn_html += f'<h3>Tool Call #{i+1}: {tool_call["tool_name"]}</h3>\n'
                    turn_html += '<pre>\n'
                    turn_html += json.dumps(tool_call["tool_input"], indent=2)
                    turn_html += '\n</pre>\n'
                    turn_html += '</div>\n'

                    # Tool result
                    if "tool_result" in tool_call:
                        turn_html += '<div class="tool-result">\n'
                        turn_html += '<h3>Tool Result</h3>\n'

                        # Check if there's an image in the tool result
                        if "image_path" in tool_call:
                            relative_path = os.path.relpath(tool_call["image_path"], self.run_dir)
                            turn_html += f'<div class="image-container">\n'
                            turn_html += f'<img src="{relative_path}" alt="Tool result image">\n'
                            turn_html += f'<div class="metadata">Image: {os.path.basename(tool_call["image_path"])}</div>\n'
                            turn_html += '</div>\n'

                        result_text = tool_call["tool_result"].replace("\n", "<br>")
                        turn_html += f'<p>{result_text}</p>\n'
                        turn_html += '</div>\n'

            turn_html += '</div>\n'
            html_content += turn_html

        # Update the HTML file
        html_file = self.run_dir / "index.html"
        with open(html_file, "r") as f:
            full_html = f.read()

        # Replace the conversation placeholder
        updated_html = re.sub(
            r'<div id="conversation">.*?</div>',
            f'<div id="conversation">\n{html_content}</div>',
            full_html,
            flags=re.DOTALL
        )

        with open(html_file, "w") as f:
            f.write(updated_html)

    def save_image_from_base64(self, base64_data: str, prefix: str = "image") -> Path:
        """Save an image from base64 data.

        Args:
            base64_data: Base64 encoded image data
            prefix: Prefix for the image filename

        Returns:
            Path to the saved image
        """
        # Extract the actual base64 data if it's a data URL
        if base64_data.startswith("data:image"):
            base64_data = base64_data.split(",", 1)[1]

        # Decode the base64 data
        image_data = base64.b64decode(base64_data)

        # Generate a filename
        self.image_counter += 1
        image_filename = f"{prefix}_{self.image_counter}.png"
        image_path = self.images_dir / image_filename

        # Save the image
        with open(image_path, "wb") as f:
            f.write(image_data)

        self.logger.info(f"Saved image to {image_path}")
        return image_path

    def save_image_from_path(self, source_path: Union[str, Path], prefix: str = "image") -> Path:
        """Save a copy of an image from a path.

        Args:
            source_path: Path to the source image
            prefix: Prefix for the image filename

        Returns:
            Path to the saved image
        """
        source_path = Path(source_path)
        if not source_path.exists():
            self.logger.warning(f"Image not found: {source_path}")
            return None

        # Generate a filename
        self.image_counter += 1
        image_filename = f"{prefix}_{self.image_counter}{source_path.suffix}"
        image_path = self.images_dir / image_filename

        # Copy the image
        shutil.copy2(source_path, image_path)

        self.logger.info(f"Copied image from {source_path} to {image_path}")
        return image_path

    def log_user_message(self, message: str, image_path: Optional[Union[str, Path]] = None, image_base64: Optional[str] = None):
        """Log a user message.

        Args:
            message: The user message
            image_path: Optional path to an image to include
            image_base64: Optional base64 encoded image data
        """
        self.turn_counter += 1
        turn_data = {
            "turn_id": self.turn_counter,
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
        }

        # Save the image if provided
        if image_path:
            saved_image_path = self.save_image_from_path(image_path, f"turn_{self.turn_counter}_user")
            turn_data["image_path"] = saved_image_path
        elif image_base64:
            saved_image_path = self.save_image_from_base64(image_base64, f"turn_{self.turn_counter}_user")
            turn_data["image_path"] = saved_image_path

        self.conversation_log.append(turn_data)
        self._update_html()

        # Log to file
        self.logger.info(f"User message (Turn {self.turn_counter}): {message[:100]}...")
        if image_path or image_base64:
            self.logger.info(f"User message includes an image")

    def log_model_response(self, response: str):
        """Log a model response.

        Args:
            response: The model's response text
        """
        if not self.conversation_log:
            self.logger.warning("Trying to log model response without a preceding user message")
            self.turn_counter += 1
            self.conversation_log.append({
                "turn_id": self.turn_counter,
                "timestamp": datetime.now().isoformat(),
            })

        current_turn = self.conversation_log[-1]
        current_turn["model_response"] = response

        self._update_html()

        # Log to file
        self.logger.info(f"Model response (Turn {current_turn['turn_id']}): {response[:100]}...")

    def log_tool_call(self, tool_name: str, tool_input: Dict[str, Any]):
        """Log a tool call.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters for the tool
        """
        if not self.conversation_log:
            self.logger.warning("Trying to log tool call without a preceding user message")
            self.turn_counter += 1
            self.conversation_log.append({
                "turn_id": self.turn_counter,
                "timestamp": datetime.now().isoformat(),
            })

        current_turn = self.conversation_log[-1]

        if "tool_calls" not in current_turn:
            current_turn["tool_calls"] = []

        tool_call_data = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "timestamp": datetime.now().isoformat(),
        }

        current_turn["tool_calls"].append(tool_call_data)
        self._update_html()

        # Log to file
        self.logger.info(f"Tool call (Turn {current_turn['turn_id']}): {tool_name} - {json.dumps(tool_input)[:100]}...")

    def log_tool_result(self, tool_result: str, image_path: Optional[Union[str, Path]] = None, image_base64: Optional[str] = None):
        """Log a tool result.

        Args:
            tool_result: The result of the tool call
            image_path: Optional path to an image to include
            image_base64: Optional base64 encoded image data
        """
        if not self.conversation_log:
            self.logger.warning("Trying to log tool result without a preceding user message")
            return

        current_turn = self.conversation_log[-1]

        if "tool_calls" not in current_turn or not current_turn["tool_calls"]:
            self.logger.warning("Trying to log tool result without a preceding tool call")
            return

        # Add the result to the most recent tool call
        latest_tool_call = current_turn["tool_calls"][-1]
        latest_tool_call["tool_result"] = tool_result

        # Save the image if provided
        if image_path:
            saved_image_path = self.save_image_from_path(
                image_path,
                f"turn_{current_turn['turn_id']}_tool_{len(current_turn['tool_calls'])}"
            )
            latest_tool_call["image_path"] = saved_image_path
        elif image_base64:
            saved_image_path = self.save_image_from_base64(
                image_base64,
                f"turn_{current_turn['turn_id']}_tool_{len(current_turn['tool_calls'])}"
            )
            latest_tool_call["image_path"] = saved_image_path

        self._update_html()

        # Log to file
        self.logger.info(f"Tool result (Turn {current_turn['turn_id']}): {tool_result[:100]}...")
        if image_path or image_base64:
            self.logger.info(f"Tool result includes an image")

    def _make_json_serializable(self, obj):
        """Convert an object to a JSON serializable format.

        Args:
            obj: The object to convert

        Returns:
            A JSON serializable version of the object
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def save_conversation_json(self):
        """Save the conversation log as a JSON file."""
        json_path = self.conversations_dir / f"conversation_{self.turn_counter}.json"

        # Convert Path objects to strings for JSON serialization
        serializable_log = self._make_json_serializable(self.conversation_log)

        with open(json_path, "w") as f:
            json.dump(serializable_log, f, indent=2)

        self.logger.info(f"Saved conversation log to {json_path}")
        return json_path

    def finalize_run(self, summary: Optional[str] = None):
        """Finalize the run, saving all logs and generating a summary.

        Args:
            summary: Optional summary of the run
        """
        # Save the conversation log
        self.save_conversation_json()

        # Add summary to the HTML if provided
        if summary:
            html_file = self.run_dir / "index.html"
            with open(html_file, "r") as f:
                html_content = f.read()

            # Add summary before the conversation
            # Replace newlines with <br> tags first
            summary_with_breaks = summary.replace('\n', '<br>')
            summary_html = f"""
            <div class="summary">
                <h2>Run Summary</h2>
                <p>{summary_with_breaks}</p>
            </div>
            """

            # Insert after the start time
            html_content = re.sub(
                r'<p>Started at:.*?</p>',
                r'\g<0>\n' + summary_html,
                html_content
            )

            with open(html_file, "w") as f:
                f.write(html_content)

        self.logger.info(f"Finalized run: {self.run_id}")
        self.console.print(f"[bold green]Finalized run:[/] {self.run_id}")
        self.console.print(f"[bold green]Log directory:[/] {self.run_dir}")
        self.console.print(f"[bold green]HTML report:[/] {self.run_dir / 'index.html'}")
