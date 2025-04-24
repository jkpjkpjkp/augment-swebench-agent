from pathlib import Path
from PIL import Image

from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager
from tools.image_tools import BlackoutTool

def test_blackout_non_overlapping():
    # Create a test workspace
    workspace_path = Path("test_blackout_non_overlapping_workspace")
    workspace_path.mkdir(exist_ok=True)
    
    # Create images and views directories
    images_dir = workspace_path / "images"
    views_dir = workspace_path / "views"
    images_dir.mkdir(exist_ok=True)
    views_dir.mkdir(exist_ok=True)
    
    # Create a test image (blue)
    test_image = Image.new('RGB', (200, 200), (0, 0, 255))
    test_image_path = images_dir / "test_image.png"
    test_image.save(test_image_path)
    
    # Initialize workspace manager and image manager
    workspace_manager = WorkspaceManager(workspace_path)
    image_manager = ImageManager(workspace_path)
    
    # Create multiple non-overlapping views
    view1_id = "view1"
    view1_coords = (0, 0, 50, 50)  # Top-left
    print(f"Creating view1 with coordinates: {view1_coords}")
    image_manager.create_view(test_image_path, view1_id, view1_coords)
    
    view2_id = "view2"
    view2_coords = (150, 0, 200, 50)  # Top-right
    print(f"Creating view2 with coordinates: {view2_coords}")
    image_manager.create_view(test_image_path, view2_id, view2_coords)
    
    view3_id = "view3"
    view3_coords = (0, 150, 50, 200)  # Bottom-left
    print(f"Creating view3 with coordinates: {view3_coords}")
    image_manager.create_view(test_image_path, view3_id, view3_coords)
    
    view4_id = "view4"
    view4_coords = (150, 150, 200, 200)  # Bottom-right
    print(f"Creating view4 with coordinates: {view4_coords}")
    image_manager.create_view(test_image_path, view4_id, view4_coords)
    
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
        # All remaining views are the same size
    else:
        print("No smallest view found")
    
    # Black out view2
    print("\nBlacking out view2...")
    result = blackout_tool.run_impl({"image_path": view2_id})
    print(f"Result: {result.tool_output}")
    
    # List all views after blackout
    print("\nViews after blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Check if view2 was deleted
    view2_exists = (views_dir / f"test_image__{view2_id}__150_0_200_50.png").exists()
    print(f"\nView2 exists: {view2_exists} (should be False)")
    
    # Black out view3
    print("\nBlacking out view3...")
    result = blackout_tool.run_impl({"image_path": view3_id})
    print(f"Result: {result.tool_output}")
    
    # List all views after blackout
    print("\nViews after blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Check if view3 was deleted
    view3_exists = (views_dir / f"test_image__{view3_id}__0_150_50_200.png").exists()
    print(f"\nView3 exists: {view3_exists} (should be False)")
    
    # Black out view4
    print("\nBlacking out view4...")
    result = blackout_tool.run_impl({"image_path": view4_id})
    print(f"Result: {result.tool_output}")
    
    # List all views after blackout
    print("\nViews after blackout:")
    for view_path in image_manager.list_views(test_image_path):
        print(f"- {view_path.name}")
    
    # Check if view4 was deleted
    view4_exists = (views_dir / f"test_image__{view4_id}__150_150_200_200.png").exists()
    print(f"\nView4 exists: {view4_exists} (should be False)")
    
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
    test_blackout_non_overlapping()
