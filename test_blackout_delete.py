import os
from pathlib import Path
from PIL import Image
import numpy as np

from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager
from tools.image_tools import BlackoutTool

def test_blackout_delete():
    # Create a test workspace
    workspace_path = Path("test_blackout_delete_workspace")
    workspace_path.mkdir(exist_ok=True)
    
    # Create images and views directories
    images_dir = workspace_path / "images"
    views_dir = workspace_path / "views"
    images_dir.mkdir(exist_ok=True)
    views_dir.mkdir(exist_ok=True)
    
    # Create a test image (blue)
    test_image = Image.new('RGB', (100, 100), (0, 0, 255))
    test_image_path = images_dir / "test_image.png"
    test_image.save(test_image_path)
    
    # Initialize workspace manager and image manager
    workspace_manager = WorkspaceManager(workspace_path)
    image_manager = ImageManager(workspace_path)
    
    # Create multiple views of different sizes
    view1_id = "view1"
    view1_coords = (0, 0, 50, 50)  # 50x50 = 2500 pixels
    print(f"Creating view1 with coordinates: {view1_coords}")
    view1_path = image_manager.create_view(test_image_path, view1_id, view1_coords)
    
    view2_id = "view2"
    view2_coords = (0, 0, 30, 30)  # 30x30 = 900 pixels (smallest)
    print(f"Creating view2 with coordinates: {view2_coords}")
    view2_path = image_manager.create_view(test_image_path, view2_id, view2_coords)
    
    view3_id = "view3"
    view3_coords = (0, 0, 40, 40)  # 40x40 = 1600 pixels
    print(f"Creating view3 with coordinates: {view3_coords}")
    view3_path = image_manager.create_view(test_image_path, view3_id, view3_coords)
    
    # Initialize the blackout tool
    blackout_tool = BlackoutTool(workspace_manager)
    blackout_tool.image_manager = image_manager
    
    # List all views
    print("\nViews before blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Black out view1
    print("\nBlacking out view1...")
    result = blackout_tool.run_impl({"image_path": view1_id})
    print(f"Result: {result.tool_output}")
    
    # List all views after blackout
    print("\nViews after blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Check if view1 was deleted
    view1_exists = (views_dir / f"test_image__{view1_id}__0_0_50_50.png").exists()
    print(f"\nView1 exists: {view1_exists} (should be False)")
    
    # Find the smallest view
    smallest_view = image_manager.find_smallest_view(test_image_path)
    if smallest_view:
        print(f"\nSmallest view: {smallest_view.name}")
        # This should be view2 (30x30 = 900 pixels)
    else:
        print("No smallest view found")
    
    # Black out view2 (the smallest view)
    print("\nBlacking out view2 (the smallest view)...")
    result = blackout_tool.run_impl({"image_path": view2_id})
    print(f"Result: {result.tool_output}")
    
    # List all views after blackout
    print("\nViews after blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Check if view2 was deleted
    view2_exists = (views_dir / f"test_image__{view2_id}__0_0_30_30.png").exists()
    print(f"\nView2 exists: {view2_exists} (should be False)")
    
    # Find the smallest view again
    smallest_view = image_manager.find_smallest_view(test_image_path)
    if smallest_view:
        print(f"\nSmallest view: {smallest_view.name}")
        # This should be view3 (40x40 = 1600 pixels)
    else:
        print("No smallest view found")
    
    # Black out view3 (the last view)
    print("\nBlacking out view3 (the last view)...")
    result = blackout_tool.run_impl({"image_path": view3_id})
    print(f"Result: {result.tool_output}")
    
    # List all views after blackout
    print("\nViews after blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Check if view3 was deleted
    view3_exists = (views_dir / f"test_image__{view3_id}__0_0_40_40.png").exists()
    print(f"\nView3 exists: {view3_exists} (should be False)")
    
    # Find the smallest view again
    smallest_view = image_manager.find_smallest_view(test_image_path)
    if smallest_view:
        print(f"\nSmallest view: {smallest_view.name}")
    else:
        print("No smallest view found (expected since all views are deleted)")
    
    # Clean up
    print("\nCleaning up...")
    for file in images_dir.glob("*"):
        file.unlink()
    for file in views_dir.glob("*"):
        file.unlink()
    images_dir.rmdir()
    views_dir.rmdir()
    workspace_path.rmdir()
    print("Done!")

if __name__ == "__main__":
    test_blackout_delete()
