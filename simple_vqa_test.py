#!/usr/bin/env python3
"""
Simple test script for VQA functionality.

This script tests the core image management functionality without relying on external data.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

from utils.image_manager import ImageManager

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
    # Create a test workspace
    workspace_root = Path("./simple_vqa_test_workspace")
    workspace_root.mkdir(exist_ok=True)
    
    print(f"Testing VQA functionality in workspace: {workspace_root}")
    
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
    
    print("VQA test completed successfully!")

if __name__ == "__main__":
    main()
