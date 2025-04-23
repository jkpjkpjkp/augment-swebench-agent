"""
Test the blackout tool's behavior with overlapping views.

This test specifically verifies that when a view is blacked out:
1. The corresponding pixels in overlapping views also become black
2. Only the overlapping regions are affected
"""

import shutil
from pathlib import Path
import unittest
from PIL import Image
import numpy as np

from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager


class TestBlackoutOverlap(unittest.TestCase):
    """Test the blackout tool's behavior with overlapping views."""

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
        
        # Create a blue test image (800x1000)
        self.test_image_path = self.workspace_path / "images" / "blue_test.png"
        blue_image = Image.new('RGB', (800, 1000), (0, 0, 255))  # Blue image
        blue_image.save(self.test_image_path)
        print(f"Created test image at: {self.test_image_path}")
        
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
        pass

    def test_overlapping_views_blackout(self):
        """Test that blackout properly affects overlapping views."""
        # Create two overlapping views:
        # View 1: Top-left 2/3 (0, 0, 533, 667)
        # View 2: Bottom-right 2/3 (267, 333, 800, 1000)
        # The overlap is the region (267, 333, 533, 667)
        
        view1 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, 533, 667),
            "top_left_two_thirds"
        )
        print(f"Created view 1 (top-left 2/3): {view1}")
        
        view2 = self.image_manager.create_view(
            self.test_image_path,
            (267, 333, 800, 1000),
            "bottom_right_two_thirds"
        )
        print(f"Created view 2 (bottom-right 2/3): {view2}")
        
        # Verify both views are blue before blackout
        view1_img_before = Image.open(view1)
        view1_pixels_before = np.array(view1_img_before)
        self.assertTrue(np.all(view1_pixels_before[:, :, 2] == 255), "View 1 should be completely blue before blackout")
        
        view2_img_before = Image.open(view2)
        view2_pixels_before = np.array(view2_img_before)
        self.assertTrue(np.all(view2_pixels_before[:, :, 2] == 255), "View 2 should be completely blue before blackout")
        
        # Now blackout view 1
        updated_paths = self.image_manager.blackout_view(view1)
        print(f"Blackout result: {updated_paths}")
        
        # Verify view 1 is completely black
        view1_img_after = Image.open(view1)
        view1_pixels_after = np.array(view1_img_after)
        self.assertTrue(np.all(view1_pixels_after == 0), "View 1 should be completely black after blackout")
        
        # Verify view 2 has the overlapping region black
        view2_img_after = Image.open(view2)
        view2_pixels_after = np.array(view2_img_after)
        
        # Calculate the overlap region in view2's coordinates
        # View1 is (0, 0, 533, 667) and View2 is (267, 333, 800, 1000)
        # Overlap in original image coordinates is (267, 333, 533, 667)
        # In view2's coordinates, this is (0, 0, 266, 334)
        overlap_width = 533 - 267
        overlap_height = 667 - 333
        
        # Check that the overlapping region is black
        self.assertTrue(np.all(view2_pixels_after[:overlap_height, :overlap_width, :] == 0), 
                        "Overlapping region in view 2 should be black")
        
        # Check that the non-overlapping regions are still blue
        # Right side of view2 (non-overlapping)
        self.assertTrue(np.all(view2_pixels_after[:, overlap_width:, 2] == 255), 
                        "Right side of view 2 (non-overlapping) should still be blue")
        
        # Bottom side of view2 (non-overlapping)
        self.assertTrue(np.all(view2_pixels_after[overlap_height:, :, 2] == 255), 
                        "Bottom side of view 2 (non-overlapping) should still be blue")
        
        # Verify the original image has the top-left 2/3 black
        original_img = Image.open(self.test_image_path)
        original_pixels = np.array(original_img)
        
        # Top-left 2/3 should be black
        self.assertTrue(np.all(original_pixels[:667, :533, :] == 0), 
                        "Top-left 2/3 of original image should be black")
        
        # Rest should still be blue
        self.assertTrue(np.all(original_pixels[667:, :, 2] == 255), 
                        "Bottom 1/3 of original image should still be blue")
        self.assertTrue(np.all(original_pixels[:667, 533:, 2] == 255), 
                        "Right 1/3 of original image should still be blue")

    def test_precise_pixel_overlap(self):
        """Test that blackout affects precisely the overlapping pixels."""
        # Create a more complex overlap pattern with three views
        # View 1: Left half with a diagonal pattern (0, 0, 400, 1000)
        # View 2: Top half with a horizontal pattern (0, 0, 800, 500)
        # View 3: Center region (200, 200, 600, 800)
        
        # Create the views
        view1 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, 400, 1000),
            "left_half"
        )
        print(f"Created view 1 (left half): {view1}")
        
        view2 = self.image_manager.create_view(
            self.test_image_path,
            (0, 0, 800, 500),
            "top_half"
        )
        print(f"Created view 2 (top half): {view2}")
        
        view3 = self.image_manager.create_view(
            self.test_image_path,
            (200, 200, 600, 800),
            "center"
        )
        print(f"Created view 3 (center): {view3}")
        
        # Now blackout view 3 (the center view)
        updated_paths = self.image_manager.blackout_view(view3)
        print(f"Blackout result: {updated_paths}")
        
        # Verify view 3 is completely black
        view3_img_after = Image.open(view3)
        view3_pixels_after = np.array(view3_img_after)
        self.assertTrue(np.all(view3_pixels_after == 0), "View 3 should be completely black after blackout")
        
        # Verify view 1 has the overlapping region black
        view1_img_after = Image.open(view1)
        view1_pixels_after = np.array(view1_img_after)
        
        # Calculate the overlap region in view1's coordinates
        # View1 is (0, 0, 400, 1000) and View3 is (200, 200, 600, 800)
        # Overlap in original image coordinates is (200, 200, 400, 800)
        # In view1's coordinates, this is (200, 200, 400, 800)
        
        # Check that the overlapping region is black
        self.assertTrue(np.all(view1_pixels_after[200:800, 200:400, :] == 0), 
                        "Overlapping region in view 1 should be black")
        
        # Check that the non-overlapping regions are still blue
        # Top part of view1 (non-overlapping)
        self.assertTrue(np.all(view1_pixels_after[:200, :, 2] == 255), 
                        "Top part of view 1 (non-overlapping) should still be blue")
        
        # Bottom part of view1 (non-overlapping)
        self.assertTrue(np.all(view1_pixels_after[800:, :, 2] == 255), 
                        "Bottom part of view 1 (non-overlapping) should still be blue")
        
        # Left part of view1 (non-overlapping)
        self.assertTrue(np.all(view1_pixels_after[200:800, :200, 2] == 255), 
                        "Left part of view 1 (non-overlapping) should still be blue")
        
        # Verify view 2 has the overlapping region black
        view2_img_after = Image.open(view2)
        view2_pixels_after = np.array(view2_img_after)
        
        # Calculate the overlap region in view2's coordinates
        # View2 is (0, 0, 800, 500) and View3 is (200, 200, 600, 800)
        # Overlap in original image coordinates is (200, 200, 600, 500)
        # In view2's coordinates, this is (200, 200, 600, 500)
        
        # Check that the overlapping region is black
        self.assertTrue(np.all(view2_pixels_after[200:500, 200:600, :] == 0), 
                        "Overlapping region in view 2 should be black")
        
        # Check that the non-overlapping regions are still blue
        # Top part of view2 (non-overlapping)
        self.assertTrue(np.all(view2_pixels_after[:200, :, 2] == 255), 
                        "Top part of view 2 (non-overlapping) should still be blue")
        
        # Left part of view2 (non-overlapping)
        self.assertTrue(np.all(view2_pixels_after[200:500, :200, 2] == 255), 
                        "Left part of view 2 (non-overlapping) should still be blue")
        
        # Right part of view2 (non-overlapping)
        self.assertTrue(np.all(view2_pixels_after[200:500, 600:, 2] == 255), 
                        "Right part of view 2 (non-overlapping) should still be blue")


if __name__ == "__main__":
    unittest.main()
