#!/usr/bin/env python3
"""
Simple test script for VQA functionality with fixed path handling.

This script tests the core image management functionality without relying on external data.
"""

import os
import shutil
from pathlib import Path
from PIL import Image

from utils.workspace_manager import WorkspaceManager
from tools.image_tools import (
    CropTool,
    SwitchImageTool,
    BlackoutTool,
    ListImagesTool,
)

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
    """Main test function."""
    # Create a clean test workspace
    workspace_root = Path("./fixed_vqa_test_workspace")
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(exist_ok=True)

    # Create images and views directories
    images_dir = workspace_root / "images"
    views_dir = workspace_root / "views"
    images_dir.mkdir(exist_ok=True)
    views_dir.mkdir(exist_ok=True)

    print(f"Testing VQA functionality in workspace: {workspace_root}")

    # Create workspace manager
    workspace_manager = WorkspaceManager(workspace_root)

    # Create a test image directly in the images directory
    test_image_path = images_dir / "sample.png"
    create_test_image(test_image_path)

    # Test ListImagesTool
    print("\n--- Testing ListImagesTool ---")
    list_tool = ListImagesTool(workspace_manager)
    result = list_tool.run_impl({"show_views": True})
    print(result.tool_output)

    # Test CropTool
    print("\n--- Testing CropTool ---")
    crop_tool = CropTool(workspace_manager)

    # Create a grid of views (2x2)
    for i in range(2):
        for j in range(2):
            x1 = j * 250
            y1 = i * 250
            x2 = x1 + 250
            y2 = y1 + 250

            result = crop_tool.run_impl({
                "image_path": "images/sample.png",  # Relative path
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "view_id": f"view_{i}_{j}"
            })
            print(result.tool_output)

    # Test ListImagesTool again
    print("\n--- Testing ListImagesTool After Creating Views ---")
    result = list_tool.run_impl({"show_views": True})
    print(result.tool_output)

    # Test SwitchImageTool
    print("\n--- Testing SwitchImageTool ---")
    switch_tool = SwitchImageTool(workspace_manager)
    result = switch_tool.run_impl({
        "image_path": "images/sample.png"  # Relative path
    })
    print(result.tool_output)

    # Test BlackoutTool
    print("\n--- Testing BlackoutTool ---")
    blackout_tool = BlackoutTool(workspace_manager)

    # Use a relative path to the view
    result = blackout_tool.run_impl({
        "image_path": "views/sample__view_0_0__0_0_250_250.png"  # Relative path
    })
    print(result.tool_output)

    # Test ListImagesTool one more time
    print("\n--- Testing ListImagesTool After Blackout ---")
    result = list_tool.run_impl({"show_views": True})
    print(result.tool_output)

    print("\nVQA test completed successfully!")

if __name__ == "__main__":
    main()
