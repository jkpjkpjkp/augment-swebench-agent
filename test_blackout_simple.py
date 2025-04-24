"""
Simple test for the blackout tool functionality.

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
from tools.image_tools import CropTool, BlackoutTool

class TestBlackoutSimple(unittest.TestCase):
    """Simple test for the blackout tool functionality."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary workspace
        self.workspace_path = Path("test_workspace")
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self.workspace_path.mkdir(exist_ok=True)

        # Create images and views directories
        self.images_dir = self.workspace_path / "images"
        self.views_dir = self.workspace_path / "views"
        self.images_dir.mkdir(exist_ok=True)
        self.views_dir.mkdir(exist_ok=True)

        # Create a blue test image (800x1000)
        self.test_image_path = self.images_dir / "blue_test.png"
        blue_image = Image.new('RGB', (800, 1000), (0, 0, 255))  # Blue image
        blue_image.save(self.test_image_path)
        print(f"Created test image at: {self.test_image_path}")

        # Create workspace manager and image manager
        self.workspace_manager = WorkspaceManager(self.workspace_path)
        self.image_manager = ImageManager(self.workspace_path)

        # Register the image with the image manager
        self.image_manager._load_existing_images()

        # Create tools
        self.crop_tool = CropTool(self.workspace_manager)
        self.blackout_tool = BlackoutTool(self.workspace_manager)

    def tearDown(self):
        """Clean up after the test."""
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    def test_blackout_overlapping_views(self):
        """Test that blackout properly affects all related views."""
        # Create overlapping views of the blue image
        # View 1: Top-left quadrant (0, 0, 400, 500)
        view1 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, 400, 500),
            "top_left"
        )
        print(f"Created view 1: {view1}")

        # View 2: Top-half (0, 0, 800, 500) - overlaps with view 1
        view2 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, 800, 500),
            "top_half"
        )
        print(f"Created view 2: {view2}")

        # View 3: Left-half (0, 0, 400, 1000) - overlaps with view 1
        view3 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, 400, 1000),
            "left_half"
        )
        print(f"Created view 3: {view3}")

        # View 4: Bottom-right quadrant (400, 500, 800, 1000) - doesn't overlap with view 1
        view4 = self.image_manager.create_view(
            self.test_image_path,
            (400, 500, 800, 1000),
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

        # Verify that the top-half view has its left half black
        top_half_img = Image.open(view2)
        top_half_pixels = np.array(top_half_img)
        # Left half should be black
        self.assertTrue(np.all(top_half_pixels[:, :400, :] == 0), "Left half of top-half view should be black")
        # Right half should still be blue
        self.assertTrue(np.all(top_half_pixels[:, 400:, 2] == 255), "Right half of top-half view should still be blue")

        # Verify that the left-half view has its top half black
        left_half_img = Image.open(view3)
        left_half_pixels = np.array(left_half_img)
        # Top half should be black
        self.assertTrue(np.all(left_half_pixels[:500, :, :] == 0), "Top half of left-half view should be black")
        # Bottom half should still be blue
        self.assertTrue(np.all(left_half_pixels[500:, :, 2] == 255), "Bottom half of left-half view should still be blue")

        # Verify that the bottom-right view is still completely blue (no overlap)
        bottom_right_img = Image.open(view4)
        bottom_right_pixels = np.array(bottom_right_img)
        self.assertTrue(np.all(bottom_right_pixels[:, :, 2] == 255), "Bottom-right view should still be completely blue")

        # Verify that the original image has the top-left quadrant black
        original_img = Image.open(self.test_image_path)
        original_pixels = np.array(original_img)
        # Top-left quadrant should be black
        self.assertTrue(np.all(original_pixels[:500, :400, :] == 0), "Top-left quadrant of original image should be black")
        # Rest should still be blue
        self.assertTrue(np.all(original_pixels[500:, :, 2] == 255), "Bottom half of original image should still be blue")
        self.assertTrue(np.all(original_pixels[:500, 400:, 2] == 255), "Top-right quadrant of original image should still be blue")


if __name__ == "__main__":
    unittest.main()
