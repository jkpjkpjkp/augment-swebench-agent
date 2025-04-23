#!/usr/bin/env python3
"""
Test script for running the agent with VQA functionality.

This script initializes the agent with the VQA tools and runs it with a sample VQA task.
"""

import os
import shutil
from pathlib import Path
import argparse
import logging

from rich.console import Console
from rich.panel import Panel

from tools.agent import Agent
from utils.workspace_manager import WorkspaceManager
from utils.llm_client import get_client
from prompts.instruction import INSTRUCTION_PROMPT
from PIL import Image

# Create a sample image for testing
def create_test_image(output_path, size=(500, 500)):
    """Create a test image with colored regions."""
    # Create a new RGB image with a white background
    img = Image.new('RGB', size, color=(255, 255, 255))

    # Create a 2x2 grid of colored squares
    width, height = size
    colors = [
        (255, 0, 0),    # Red (top-left)
        (0, 255, 0),    # Green (top-right)
        (0, 0, 255),    # Blue (bottom-left)
        (255, 255, 0),  # Yellow (bottom-right)
    ]

    # Draw the colored squares
    for i, color in enumerate(colors):
        x = (i % 2) * (width // 2)
        y = (i // 2) * (height // 2)
        for px in range(x, x + width // 2):
            for py in range(y, y + height // 2):
                img.putpixel((px, py), color)

    # Save the image
    img.save(output_path)
    print(f"Created test image at {output_path}")
    return output_path

def main():
    """Main function to run the agent with VQA functionality."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the agent with VQA functionality")
    parser.add_argument(
        "--workspace",
        type=str,
        default="./vqa_agent_workspace",
        help="Path to the workspace",
    )
    parser.add_argument(
        "--logs-path",
        type=str,
        default="vqa_agent_logs.txt",
        help="Path to save logs",
    )
    args = parser.parse_args()

    # Set up logging
    if os.path.exists(args.logs_path):
        os.remove(args.logs_path)
    logger = logging.getLogger("vqa_agent_logs")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.logs_path))
    logger.addHandler(logging.StreamHandler())

    # Initialize console
    console = Console()

    # Print welcome message
    console.print(
        Panel(
            "[bold]VQA Agent Test[/bold]\n\n"
            + "Testing the agent with VQA functionality.\n",
            title="[bold blue]VQA Agent Test[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Create a clean workspace
    workspace_path = Path(args.workspace).resolve()
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Create images and views directories
    images_dir = workspace_path / "images"
    views_dir = workspace_path / "views"
    images_dir.mkdir(exist_ok=True)
    views_dir.mkdir(exist_ok=True)

    # Create a test image
    test_image_path = images_dir / "sample.png"
    create_test_image(test_image_path)

    # Initialize workspace manager
    workspace_manager = WorkspaceManager(root=workspace_path)

    # Initialize LLM client (using Gemini-2.5-pro-exp via OpenAI API)
    client = get_client(
        "gemini-direct",
        model_name="gemini-2.5-pro-exp",
    )

    # Initialize agent
    agent = Agent(
        client=client,
        workspace_manager=workspace_manager,
        console=console,
        logger_for_agent_logs=logger,
        max_output_tokens_per_turn=32768,
        max_turns=20,
        ask_user_permission=False,
    )

    # Create a sample VQA task
    vqa_task = """
    Look at the image and identify the colors in each quadrant of the image.
    What color is in the top-left quadrant? What color is in the top-right quadrant?
    What color is in the bottom-left quadrant? What color is in the bottom-right quadrant?
    """

    # Format the instruction using the template
    instruction = INSTRUCTION_PROMPT.format(
        location=workspace_path,
        pr_description=vqa_task,
    )

    # Run the agent
    logger.info(f"Starting agent with instruction:\n{instruction}\n")
    result = agent.run_agent(instruction)

    # Print the result
    console.print(f"\n[bold]Agent Result:[/bold]\n{result}")
    logger.info(f"Agent result: {result}")

    console.print("\n[bold green]VQA Agent test completed![/bold green]")

if __name__ == "__main__":
    main()
