#!/usr/bin/env python3
"""
Test script for running the agent with real VQA data.

This script loads data from a parquet file, extracts images and questions,
and runs the agent to answer questions about the images.
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
import io
import polars as pl
from PIL import Image

from rich.console import Console
from rich.panel import Panel

from tools.agent import Agent
from utils.workspace_manager import WorkspaceManager
from utils.llm_client import get_client
from prompts.instruction import INSTRUCTION_PROMPT
from utils.image_manager import ImageManager

def load_vqa_data(parquet_path, task_id='37_3'):
    """Load VQA data from a parquet file.

    Args:
        parquet_path: Path to the parquet file
        task_id: Task ID to filter by (default: '37_3')

    Returns:
        A dictionary with the task data
    """
    # Load the parquet file
    df = pl.read_parquet(parquet_path)
    print(f"Loaded data with {df.height} rows and columns: {df.columns}")

    # Filter by task_id
    filtered_df = df.filter(pl.col('question_id') == task_id)

    if filtered_df.height == 0:
        raise ValueError(f"Task ID {task_id} not found in the dataset")
    elif filtered_df.height > 1:
        print(f"Warning: Multiple entries found for task ID {task_id}, using the first one")

    # Get the row data
    row = filtered_df.row(0, named=True)

    # Extract images
    images = [Image.open(io.BytesIO(x['bytes'])) for x in row['question_images_decoded']]

    print(f"Found {len(images)} images for task ID {task_id}")

    # Merge images side by side if there are multiple
    if len(images) > 1:
        # Calculate total width and maximum height
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create a new image with the calculated size
        merged_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # Paste each image side by side
        x = 0
        for img in images:
            merged_image.paste(img, (x, 0))
            x += img.width

        image = merged_image
    else:
        image = images[0]

    # Create a dictionary with the task data
    task_data = {
        'task_id': task_id,
        'question': row['question_text'],
        'answer': row['question_answer'],
        'image': image,
    }

    print(f"Question: {task_data['question']}")
    print(f"Answer: {task_data['answer']}")

    return task_data

def main():
    """Main function to run the agent with real VQA data."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the agent with real VQA data")
    parser.add_argument(
        "--workspace",
        type=str,
        default="./vqa_real_data_workspace",
        help="Path to the workspace",
    )
    parser.add_argument(
        "--logs-path",
        type=str,
        default="vqa_real_data_logs.txt",
        help="Path to save logs",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="/home/jkp/Téléchargements/zerobench_subquestions-00000-of-00001.parquet",
        help="Path to the parquet file with VQA data",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="37_3",
        help="Task ID to use (default: 37_3)",
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
            "[bold]VQA Agent Test with Real Data[/bold]\n\n"
            + f"Testing the agent with VQA data from {args.parquet_path}\n",
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

    # Load VQA data
    task_data = load_vqa_data(args.parquet_path, args.task_id)

    # Print task information
    console.print(f"\n[bold]Task ID:[/bold] {task_data['task_id']}")
    console.print(f"[bold]Question:[/bold] {task_data['question']}")
    console.print(f"[bold]Answer:[/bold] {task_data['answer']}")
    console.print(f"[bold]Image Size:[/bold] {task_data['image'].size}")

    # Save the image to a temporary location first
    temp_image_path = workspace_path / f"{task_data['task_id']}_temp.png"
    task_data['image'].save(temp_image_path)
    console.print(f"[bold]Temporary image saved to:[/bold] {temp_image_path}")

    # Use ImageManager to properly register the image
    image_manager = ImageManager(workspace_path)
    image_path = image_manager.add_image(temp_image_path, f"{task_data['task_id']}.png")
    console.print(f"[bold]Image added to workspace at:[/bold] {image_path}")

    # Remove the temporary image
    temp_image_path.unlink()

    # Initialize workspace manager
    workspace_manager = WorkspaceManager(root=workspace_path)

    # Initialize LLM client (using Gemini-2.5-pro-exp via OpenAI API)
    client = get_client(
        "gemini-direct",
        model_name="gemini-2.5-pro-exp-03-25",
    )

    # Initialize agent
    agent = Agent(
        client=client,
        workspace_manager=workspace_manager,
        console=console,
        logger_for_agent_logs=logger,
        max_output_tokens_per_turn=32768,
        max_turns=20,
        log_dir="agent_runs"
    )

    # Format the instruction using the template
    # Replace the literal curly braces example with double braces to escape them during formatting
    instruction_template = INSTRUCTION_PROMPT.replace("{important detail to remember}", "{{important detail to remember}}")
    instruction = instruction_template.format(
        pr_description=task_data['question'],
    )

    # Run the agent with the initial image
    logger.info(f"Starting agent with instruction:\n{instruction}\n")
    result = agent.run_agent(
        instruction=instruction,
        initial_image_path=image_path  # Pass the image path to include with the first message
    )

    # Print the result
    console.print(f"\n[bold]Agent Result:[/bold]\n{result}")
    logger.info(f"Agent result: {result}")

    # Compare with the expected answer
    console.print(f"\n[bold]Expected Answer:[/bold] {task_data['answer']}")

    console.print("\n[bold green]VQA Agent test completed![/bold green]")

if __name__ == "__main__":
    main()
