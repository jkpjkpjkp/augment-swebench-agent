"""
Test the blackout tool functionality.

This test verifies that when a view is blacked out:
1. The view itself becomes black
2. The corresponding region in the original image becomes black
3. Any overlapping regions in other views also become black
"""

import shutil
from pathlib import Path
import unittest
from PIL import Image
import numpy as np

from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager
from tools.image_tools import BlackoutTool


class TestBlackoutTool(unittest.TestCase):
    """Test the blackout tool functionality."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary workspace
        self.workspace_path = Path("test_workspace")
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self.workspace_path.mkdir(exist_ok=True)
        
        # Create images and views directories
        (self.workspace_path / "images").mkdir(exist_ok=True)
        (self.workspace_path / "views").mkdir(exist_ok=True)
        
        # Create workspace manager and image manager
        self.workspace_manager = WorkspaceManager(self.workspace_path)
        self.image_manager = ImageManager(self.workspace_path)
        
        # Create tools
        self.blackout_tool = BlackoutTool(self.workspace_manager)
        
        # Create a blue test image directly in the test_workspace/images directory
        self.test_image_path = self.workspace_path / "images" / "blue_test.png"
        
        # Create the image if it doesn't exist
        if not self.test_image_path.exists():
            blue_image = Image.new('RGB', (800, 1000), (0, 0, 255))  # Blue image
            blue_image.save(self.test_image_path)
            print(f"Created new test image at: {self.test_image_path}")
        else:
            print(f"Using existing test image at: {self.test_image_path}")
        
        # Get the image dimensions
        img = Image.open(self.test_image_path)
        self.img_width, self.img_height = img.size
        print(f"Test image dimensions: {self.img_width}x{self.img_height}")
        
        # Force reload of images to ensure it's in the registry
        self.image_manager._load_existing_images()
        
        # Verify the image is in the registry
        found = False
        for img_path in self.image_manager.image_views.keys():
            if img_path.name == self.test_image_path.name:
                found = True
                self.test_image_path = img_path  # Use the registered path
                break
        
        if not found:
            # Manually add to registry if not found
            self.image_manager.image_views[self.test_image_path] = {}
            print(f"Manually added image to registry: {self.test_image_path}")

    def tearDown(self):
        """Clean up after the test."""
        # Don't delete the workspace so we can inspect it after the test
        # if self.workspace_path.exists():
        #     shutil.rmtree(self.workspace_path)
        pass

    def test_blackout_view(self):
        """Test that blackout properly affects all related views."""
        # Calculate view dimensions based on image size
        half_width = self.img_width // 2
        half_height = self.img_height // 2
        
        # Create overlapping views of the blue image using the ImageManager directly
        # View 1: Top-left quadrant
        view1 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, half_width, half_height),
            "top_left"
        )
        print(f"Created view 1: {view1}")
        
        # View 2: Top-half - overlaps with view 1
        view2 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, self.img_width, half_height),
            "top_half"
        )
        print(f"Created view 2: {view2}")
        
        # View 3: Left-half - overlaps with view 1
        view3 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, half_width, self.img_height),
            "left_half"
        )
        print(f"Created view 3: {view3}")
        
        # View 4: Bottom-right quadrant - doesn't overlap with view 1
        view4 = self.image_manager.create_view(
            self.test_image_path,
            (half_width, half_height, self.img_width, self.img_height),
            "bottom_right"
        )
        print(f"Created view 4: {view4}")
        
        # List all views to verify they were created
        views = self.image_manager.list_views(self.test_image_path)
        print(f"Created views: {[v.name for v in views]}")
        
        # Now blackout the top-left view
        updated_paths = self.image_manager.blackout_view(view1)
        print(f"Blackout result: {updated_paths}")
        
        # Verify that the top-left view is now black
        top_left_img = Image.open(view1)
        top_left_pixels = np.array(top_left_img)
        self.assertTrue(np.all(top_left_pixels == 0), "Top-left view should be completely black")
        
        # Find the view paths
        top_half_view_path = view2
        left_half_view_path = view3
        bottom_right_view_path = view4
        
        # Calculate half dimensions
        half_width = self.img_width // 2
        half_height = self.img_height // 2
        
        # Verify that the top-half view has its left half black
        top_half_img = Image.open(top_half_view_path)
        top_half_pixels = np.array(top_half_img)
        # Left half should be black
        self.assertTrue(np.all(top_half_pixels[:, :half_width, :] == 0), "Left half of top-half view should be black")
        # Right half should still be blue
        self.assertTrue(np.all(top_half_pixels[:, half_width:, 2] == 255), "Right half of top-half view should still be blue")
        
        # Verify that the left-half view has its top half black
        left_half_img = Image.open(left_half_view_path)
        left_half_pixels = np.array(left_half_img)
        # Top half should be black
        self.assertTrue(np.all(left_half_pixels[:half_height, :, :] == 0), "Top half of left-half view should be black")
        # Bottom half should still be blue
        self.assertTrue(np.all(left_half_pixels[half_height:, :, 2] == 255), "Bottom half of left-half view should still be blue")
        
        # Verify that the bottom-right view is still completely blue (no overlap)
        bottom_right_img = Image.open(bottom_right_view_path)
        bottom_right_pixels = np.array(bottom_right_img)
        self.assertTrue(np.all(bottom_right_pixels[:, :, 2] == 255), "Bottom-right view should still be completely blue")
        
        # Verify that the original image has the top-left quadrant black
        original_img = Image.open(self.test_image_path)
        original_pixels = np.array(original_img)
        # Top-left quadrant should be black
        self.assertTrue(np.all(original_pixels[:half_height, :half_width, :] == 0), "Top-left quadrant of original image should be black")
        # Rest should still be blue
        self.assertTrue(np.all(original_pixels[half_height:, :, 2] == 255), "Bottom half of original image should still be blue")
        self.assertTrue(np.all(original_pixels[:half_height, half_width:, 2] == 255), "Top-right quadrant of original image should still be blue")

    def test_blackout_with_relative_paths(self):
        """Test that blackout works with relative paths."""
        # Create a view using 1/10 of the image dimensions
        view_width = self.img_width // 10
        view_height = self.img_height // 10
        test_view = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, view_width, view_height),
            "test_view"
        )
        print(f"Created test view: {test_view}")
        
        # Try to blackout using different path formats
        # 1. Just the view ID
        updated_paths = self.image_manager.blackout_view(test_view)
        print(f"Blackout result with view ID: {updated_paths}")
        
        # Verify the view is black
        view_img = Image.open(test_view)
        view_pixels = np.array(view_img)
        self.assertTrue(np.all(view_pixels == 0), "View should be completely black")

    def test_blackout_with_agent_style_paths(self):
        """Test blackout with paths in the style the agent would use."""
        # Create a view using 1/8 of the image dimensions
        view_width = self.img_width // 8
        view_height = self.img_height // 8
        agent_view = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, view_width, view_height),
            "agent_view"
        )
        print(f"Created agent view: {agent_view}")
        
        # Try to blackout using the agent-style path
        agent_view_path = Path(f"views/{agent_view.name}")
        updated_paths = self.image_manager.blackout_view(agent_view_path)
        print(f"Blackout result with agent-style path: {updated_paths}")
        
        # Verify the view is black
        view_img = Image.open(agent_view)
        view_pixels = np.array(view_img)
        self.assertTrue(np.all(view_pixels == 0), "View should be completely black")


if __name__ == "__main__":
    unittest.main()
