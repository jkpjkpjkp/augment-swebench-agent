"""Utility functions for image processing in VQA tasks."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

def crop_image(
    image: Union[Image.Image, Path],
    coordinates: Tuple[int, int, int, int]
) -> Image.Image:
    """Crop an image using the given coordinates.

    Args:
        image: Image or path to image
        coordinates: Crop coordinates (x1, y1, x2, y2)

    Returns:
        Cropped image
    """
    if isinstance(image, Path):
        image = Image.open(image)

    return image.crop(coordinates)

def blackout_region(
    image: Union[Image.Image, Path],
    coordinates: Optional[Tuple[int, int, int, int]] = None
) -> Image.Image:
    """Blackout a region in an image.

    Args:
        image: Image or path to image
        coordinates: Optional region to blackout (x1, y1, x2, y2)
                    If None, blackout the entire image

    Returns:
        Image with the region blacked out
    """
    if isinstance(image, Path):
        image = Image.open(image)

    # Create a copy of the image
    result = image.copy()

    # Get image dimensions
    width, height = image.size

    # If no coordinates provided, blackout the entire image
    if coordinates is None:
        coordinates = (0, 0, width, height)

    # Create a drawing context
    draw = ImageDraw.Draw(result)

    # Draw a black rectangle over the region
    draw.rectangle(coordinates, fill=(0, 0, 0))

    return result

def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get the dimensions of an image.

    Args:
        image_path: Path to the image

    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size

def crop_to_remove_black_regions(image: Union[Image.Image, Path], padding: int = 10) -> Image.Image:
    """Crop an image to remove black regions around the edges.

    This function analyzes the image and finds the smallest bounding box that contains
    all non-black pixels, then crops the image to that bounding box plus padding.

    Args:
        image: Image or path to image
        padding: Number of pixels to add around the non-black region (default: 10)

    Returns:
        Cropped image with black regions removed
    """
    if isinstance(image, Path):
        image = Image.open(image)

    # Convert to numpy array for faster processing
    img_array = np.array(image)

    # Check if the image is entirely black
    if np.all(img_array == 0):
        return image  # Return the original image if it's all black

    # Find non-black pixels (any channel > 0)
    non_black = np.any(img_array > 0, axis=2)

    # Find the bounding box of non-black pixels
    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)

    if not np.any(rows) or not np.any(cols):
        return image  # Return the original image if no non-black pixels found

    # Find the boundaries
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add padding
    height, width = img_array.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width - 1, x_max + padding)
    y_max = min(height - 1, y_max + padding)

    # Crop the image
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))
