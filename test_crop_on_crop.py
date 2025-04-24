#!/usr/bin/env python3
"""
Test script for crop-on-crop functionality.

This script tests the functionality of creating crops on top of other crops,
ensuring that coordinates are properly adjusted relative to the original image.
"""

import shutil
from pathlib import Path
import unittest
from PIL import Image
import numpy as np

from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager
from tools.image_tools import CropTool, SwitchImageTool


class TestCropOnCrop(unittest.TestCase):
    """Test the crop-on-crop functionality."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary workspace
        self.workspace_path = Path("test_crop_workspace")
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
        self.crop_tool = CropTool(self.workspace_manager)
        self.switch_tool = SwitchImageTool(self.workspace_manager)

        # Reset class attributes to ensure clean state between tests
        CropTool.last_selected_image = None
        CropTool.current_view_size = None
        CropTool.current_view_coords = None

        # Create a test image with colored regions
        self.test_image_path = self.workspace_path / "images" / "test_grid.png"
        self.create_test_image(self.test_image_path, size=(800, 800))

        # Get the image dimensions
        with Image.open(self.test_image_path) as img:
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

        # Make sure the crop tool and switch tool use the same image manager instance
        self.crop_tool.image_manager = self.image_manager
        self.switch_tool.image_manager = self.image_manager

    def tearDown(self):
        """Clean up after the test."""
        # Don't delete the workspace so we can inspect it after the test
        # if self.workspace_path.exists():
        #     shutil.rmtree(self.workspace_path)
        pass

    def create_test_image(self, output_path, size=(500, 500)):
        """Create a test image with colored regions."""
        # Create a new RGB image with a white background
        img = Image.new('RGB', size, color=(255, 255, 255))

        # Create a 4x4 grid of colored squares
        width, height = size
        cell_width = width // 4
        cell_height = height // 4

        # Define colors for the grid
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Navy
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (192, 192, 192),# Silver
            (128, 128, 128),# Gray
            (255, 165, 0),  # Orange
            (128, 0, 255)   # Violet
        ]

        # Draw the colored squares
        color_index = 0
        for row in range(4):
            for col in range(4):
                color = colors[color_index % len(colors)]
                color_index += 1

                x_start = col * cell_width
                y_start = row * cell_height

                for px in range(x_start, x_start + cell_width):
                    for py in range(y_start, y_start + cell_height):
                        img.putpixel((px, py), color)

        # Save the image
        img.save(output_path)
        print(f"Created test image at {output_path}")
        return output_path

    def test_crop_on_original(self):
        """Test creating a crop on the original image."""
        # First, switch to the original image
        result = self.switch_tool.run_impl({
            "image_path": str(self.test_image_path)
        })
        print(f"Switch result: {result.tool_output}")

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, self.test_image_path,
                         "CropTool.last_selected_image should be set to the original image path")

        # Create a crop of the top-left quadrant
        crop1_result = self.crop_tool.run_impl({
            "bbox": [0, 0, 500, 500],
            "view_id": "crop1"
        })
        print(f"Crop1 result: {crop1_result.tool_output}")

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the created view
        crop1_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "crop1" in view_path.name:
                crop1_path = view_path
                break

        self.assertIsNotNone(crop1_path, "Failed to find crop1 view")

        # Verify the crop dimensions
        with Image.open(crop1_path) as crop1_img:
            self.assertEqual(crop1_img.size, (400, 400), "Crop1 should be 400x400 pixels")

            # Verify the crop contains the correct colors from the original image
            # The top-left quadrant should contain the first 4 colors in a 2x2 grid
            crop1_pixels = np.array(crop1_img)

            # Check the top-left cell (should be red)
            self.assertTrue(np.all(crop1_pixels[100, 100] == [255, 0, 0]),
                            f"Top-left cell should be red, got {crop1_pixels[100, 100]}")

            # Check the top-right cell (should be green)
            self.assertTrue(np.all(crop1_pixels[100, 300] == [0, 255, 0]),
                            f"Top-right cell should be green, got {crop1_pixels[100, 300]}")

            # Check the bottom-left cell (should be blue)
            self.assertTrue(np.all(crop1_pixels[300, 100] == [0, 0, 255]),
                            f"Bottom-left cell should be blue, got {crop1_pixels[300, 100]}")

            # Check the bottom-right cell (should be yellow)
            self.assertTrue(np.all(crop1_pixels[300, 300] == [255, 255, 0]),
                            f"Bottom-right cell should be yellow, got {crop1_pixels[300, 300]}")

    def test_crop_on_crop(self):
        """Test creating a crop on top of another crop."""
        # First, switch to the original image
        result = self.switch_tool.run_impl({
            "image_path": str(self.test_image_path)
        })
        print(f"Switch result: {result.tool_output}")

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, self.test_image_path,
                         "CropTool.last_selected_image should be set to the original image path")

        # Create a crop of the top-left quadrant
        crop1_result = self.crop_tool.run_impl({
            "bbox": [0, 0, 500, 500],
            "view_id": "crop1"
        })
        print(f"Crop1 result: {crop1_result.tool_output}")

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the created view
        crop1_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "crop1" in view_path.name:
                crop1_path = view_path
                break

        self.assertIsNotNone(crop1_path, "Failed to find crop1 view")

        # Now switch to the first crop
        result = self.switch_tool.run_impl({
            "image_path": str(crop1_path)
        })
        print(f"Switch to crop1 result: {result.tool_output}")

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, crop1_path,
                         "CropTool.last_selected_image should be set to the crop1 path")

        # Create a crop of the bottom-right quadrant of crop1
        # This should correspond to the center of the original image's top-left quadrant
        crop2_result = self.crop_tool.run_impl({
            "bbox": [500, 500, 1000, 1000],  # Bottom-right quadrant in normalized coordinates
            "view_id": "crop2"
        })
        print(f"Crop2 result: {crop2_result.tool_output}")

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the second crop
        crop2_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "crop2" in view_path.name:
                crop2_path = view_path
                break

        self.assertIsNotNone(crop2_path, "Failed to find crop2 view")

        # Verify the crop dimensions
        with Image.open(crop2_path) as crop2_img:
            self.assertEqual(crop2_img.size, (200, 200), "Crop2 should be 200x200 pixels")

            # Verify the crop contains the correct colors from the original image
            # This should be the bottom-right cell of the top-left quadrant (yellow)
            crop2_pixels = np.array(crop2_img)

            # The entire crop2 should be yellow (from the bottom-right of the top-left quadrant)
            self.assertTrue(np.all(crop2_pixels[100, 100] == [255, 255, 0]),
                            f"Crop2 should be yellow, got {crop2_pixels[100, 100]}")

    def test_multiple_nested_crops(self):
        """Test creating multiple levels of nested crops."""
        # First, switch to the original image
        self.switch_tool.run_impl({
            "image_path": str(self.test_image_path)
        })

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, self.test_image_path,
                         "CropTool.last_selected_image should be set to the original image path")

        # Create a crop of the top-half of the image
        self.crop_tool.run_impl({
            "bbox": [0, 0, 1000, 500],
            "view_id": "top_half"
        })

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the created view
        crop1_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "top_half" in view_path.name:
                crop1_path = view_path
                break

        self.assertIsNotNone(crop1_path, "Failed to find top_half view")

        # Switch to the first crop
        self.switch_tool.run_impl({
            "image_path": str(crop1_path)
        })

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, crop1_path,
                         "CropTool.last_selected_image should be set to the crop1 path")

        # Create a crop of the right-half of the top-half
        self.crop_tool.run_impl({
            "bbox": [500, 0, 1000, 1000],
            "view_id": "top_right"
        })

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the second crop
        crop2_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "top_right" in view_path.name:
                crop2_path = view_path
                break

        self.assertIsNotNone(crop2_path, "Failed to find top_right view")

        # Switch to the second crop
        self.switch_tool.run_impl({
            "image_path": str(crop2_path)
        })

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, crop2_path,
                         "CropTool.last_selected_image should be set to the crop2 path")

        # Create a crop of the bottom-left of the top-right
        self.crop_tool.run_impl({
            "bbox": [0, 500, 500, 1000],
            "view_id": "nested_crop"
        })

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the third crop
        crop3_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "nested_crop" in view_path.name:
                crop3_path = view_path
                break

        self.assertIsNotNone(crop3_path, "Failed to find nested_crop view")

        # Verify the dimensions of all crops
        with Image.open(crop1_path) as crop1_img, \
             Image.open(crop2_path) as crop2_img, \
             Image.open(crop3_path) as crop3_img, \
             Image.open(self.test_image_path) as original_img:

            self.assertEqual(crop1_img.size, (800, 400), "Top half should be 800x400 pixels")
            self.assertEqual(crop2_img.size, (400, 400), "Top right should be 400x400 pixels")
            self.assertEqual(crop3_img.size, (200, 200), "Nested crop should be 200x200 pixels")

            # Verify the colors in the final nested crop
            # This should be the bottom-left of the top-right quadrant
            crop3_pixels = np.array(crop3_img)

            # The entire crop3 should have the color from that specific region
            # This would be the color at position (600, 200) in the original image
            original_pixels = np.array(original_img)
            expected_color = original_pixels[200, 600]

            self.assertTrue(np.all(crop3_pixels[100, 100] == expected_color),
                            f"Nested crop should have color {expected_color}, got {crop3_pixels[100, 100]}")

    def test_crop_with_current_view_tracking(self):
        """Test that the CropTool correctly tracks and uses the current view."""
        # First, switch to the original image
        self.switch_tool.run_impl({
            "image_path": str(self.test_image_path)
        })

        # Verify the last_selected_image is set correctly
        self.assertEqual(CropTool.last_selected_image, self.test_image_path,
                         "CropTool.last_selected_image should be set to the original image path")

        # Create a crop of the top-left quadrant
        self.crop_tool.run_impl({
            "bbox": [0, 0, 500, 500],
            "view_id": "crop1"
        })

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the created view
        crop1_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "crop1" in view_path.name:
                crop1_path = view_path
                break

        self.assertIsNotNone(crop1_path, "Failed to find crop1 view")

        # Switch to the first crop
        self.switch_tool.run_impl({
            "image_path": str(crop1_path)
        })

        # Verify that the switch tool correctly sets the last_selected_image
        self.assertEqual(CropTool.last_selected_image, crop1_path,
                         "CropTool.last_selected_image should be set to the crop1 path")

        # Create a crop of the bottom-right quadrant of crop1
        self.crop_tool.run_impl({
            "bbox": [500, 500, 1000, 1000],
            "view_id": "crop2"
        })

        # Force reload of images to ensure the view is in the registry
        self.image_manager._load_existing_images()

        # Get the path to the second crop
        crop2_path = None
        for view_path in self.image_manager.list_views(self.test_image_path):
            if "crop2" in view_path.name:
                crop2_path = view_path
                break

        self.assertIsNotNone(crop2_path, "Failed to find crop2 view")

        # Verify that the crop was created with the correct coordinates
        # The coordinates in the view name should be relative to the original image
        # For a 800x800 image, the crop1 is 0,0,400,400
        # The crop2 should be at 200,200,400,400 in the original image coordinates
        self.assertIn("200_200_400_400", crop2_path.name,
                      f"Crop2 should have coordinates 200,200,400,400 in the original image, got {crop2_path.name}")


if __name__ == "__main__":
    unittest.main()
