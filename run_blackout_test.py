"""
Run the blackout tool test after creating a blue test image.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image

def create_blue_test_image(width=800, height=1000):
    """Create a pure blue test image.
    
    Args:
        width: Width of the image in pixels
        height: Height of the image in pixels
        
    Returns:
        Path to the created image
    """
    # Create the images directory if it doesn't exist
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    # Create a blue test image
    image_path = images_dir / "blue_test.png"
    blue_image = Image.new('RGB', (width, height), (0, 0, 255))  # Blue image
    blue_image.save(image_path)
    
    print(f"Created blue test image at {image_path} with dimensions {width}x{height}")
    return image_path

if __name__ == "__main__":
    # Create the blue test image
    create_blue_test_image()
    
    # Import and run the test
    from test_blackout_tool import TestBlackoutTool
    
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBlackoutTool)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())
