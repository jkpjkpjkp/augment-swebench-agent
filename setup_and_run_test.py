"""
Set up the test environment and run the blackout tool test.
"""

import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image

def setup_test_environment():
    """Set up the test environment with the necessary directories and test image."""
    # Create test workspace directory
    workspace_path = Path("test_workspace")
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.mkdir(exist_ok=True)
    
    # Create images and views directories
    images_dir = workspace_path / "images"
    views_dir = workspace_path / "views"
    images_dir.mkdir(exist_ok=True)
    views_dir.mkdir(exist_ok=True)
    
    # Create a blue test image (800x1000)
    test_image_path = images_dir / "blue_test.png"
    blue_image = Image.new('RGB', (800, 1000), (0, 0, 255))  # Blue image
    blue_image.save(test_image_path)
    
    print(f"Created test environment:")
    print(f"- Workspace: {workspace_path}")
    print(f"- Images directory: {images_dir}")
    print(f"- Views directory: {views_dir}")
    print(f"- Test image: {test_image_path} (800x1000 blue)")
    
    return workspace_path

if __name__ == "__main__":
    # Set up the test environment
    workspace_path = setup_test_environment()
    
    # Run the test
    print("\nRunning test_blackout_tool.py...")
    subprocess.run(["uv", "run", "python", "test_blackout_tool.py"], check=True)
