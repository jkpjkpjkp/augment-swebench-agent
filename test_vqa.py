#!/usr/bin/env python3
"""
Test script for VQA functionality.

This script tests the image tools and image management functionality
by creating a simple test image, adding it to the workspace, creating views,
and applying modifications.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from utils.image_manager import ImageManager
from utils.image_utils import load_image, save_image, crop_image, blackout_region
from tools.image_tools import (
    CropTool,
    SelectTool,
    BlackoutTool,
    AddImageTool,
    ListImagesTool,
)
from utils.workspace_manager import WorkspaceManager
from utils.common import ToolImplOutput

def create_test_image(output_path: Path, size=(500, 500)):
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

def test_image_manager(workspace_root: Path):
    """Test the ImageManager class."""
    print("\n--- Testing ImageManager ---")
    
    # Create an ImageManager
    manager = ImageManager(workspace_root)
    
    # Create a test image
    test_image_path = workspace_root / "test_image.png"
    create_test_image(test_image_path)
    
    # Add the image to the workspace
    image_path = manager.add_image(test_image_path, "sample.png")
    print(f"Added image to workspace: {image_path}")
    
    # Create views
    view1_path = manager.create_view(image_path, (0, 0, 250, 250), "top_left")
    print(f"Created view 1: {view1_path}")
    
    view2_path = manager.create_view(image_path, (250, 0, 500, 250), "top_right")
    print(f"Created view 2: {view2_path}")
    
    view3_path = manager.create_view(image_path, (0, 250, 250, 500), "bottom_left")
    print(f"Created view 3: {view3_path}")
    
    view4_path = manager.create_view(image_path, (250, 250, 500, 500), "bottom_right")
    print(f"Created view 4: {view4_path}")
    
    # List all views
    views = manager.list_views(image_path)
    print(f"Views for {image_path.name}: {[v.name for v in views]}")
    
    # Blackout a view
    updated_paths = manager.blackout_view(view1_path)
    print(f"Blacked out view: {view1_path.name}")
    print(f"Updated paths: {[p.name for p in updated_paths]}")
    
    print("ImageManager test completed successfully!")
    return image_path

def test_image_tools(workspace_root: Path, test_image_path: Path):
    """Test the image tools."""
    print("\n--- Testing Image Tools ---")
    
    # Create a workspace manager
    workspace_manager = WorkspaceManager(workspace_root)
    
    # Test AddImageTool
    add_tool = AddImageTool(workspace_manager)
    result = add_tool.run_impl({
        "image_path": str(test_image_path),
        "image_name": "tool_test.png"
    })
    print(f"AddImageTool result: {result.tool_output}")
    
    # Test ListImagesTool
    list_tool = ListImagesTool(workspace_manager)
    result = list_tool.run_impl({"show_views": True})
    print(f"ListImagesTool result: {result.tool_output}")
    
    # Test CropTool
    crop_tool = CropTool(workspace_manager)
    result = crop_tool.run_impl({
        "image_path": "images/tool_test.png",
        "x1": 100,
        "y1": 100,
        "x2": 300,
        "y2": 300,
        "view_id": "center_crop"
    })
    print(f"CropTool result: {result.tool_output}")
    
    # Test SelectTool
    select_tool = SelectTool(workspace_manager)
    result = select_tool.run_impl({
        "image_path": "images/tool_test.png"
    })
    print(f"SelectTool (original) result: {result.tool_output}")
    
    # Find the view path from the crop tool output
    view_path = None
    for line in result.tool_output.split('\n'):
        if line.startswith("Created new view at"):
            view_path = line.split("at ")[1].strip()
            break
    
    if view_path:
        result = select_tool.run_impl({
            "image_path": view_path
        })
        print(f"SelectTool (view) result: {result.tool_output}")
    
    # Test BlackoutTool
    blackout_tool = BlackoutTool(workspace_manager)
    if view_path:
        result = blackout_tool.run_impl({
            "image_path": view_path
        })
        print(f"BlackoutTool result: {result.tool_output}")
    
    print("Image tools test completed successfully!")

def main():
    """Main test function."""
    # Create a test workspace
    workspace_root = Path("./vqa_test_workspace")
    workspace_root.mkdir(exist_ok=True)
    
    print(f"Testing VQA functionality in workspace: {workspace_root}")
    
    # Test ImageManager
    test_image_path = test_image_manager(workspace_root)
    
    # Test image tools
    test_image_tools(workspace_root, test_image_path)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
