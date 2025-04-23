"""Utility functions for image processing in VQA tasks."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

def load_image(image_path: Path) -> Image.Image:
    """Load an image from a file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image
    """
    return Image.open(image_path)

def save_image(image: Image.Image, output_path: Path) -> Path:
    """Save an image to a file.
    
    Args:
        image: Image to save
        output_path: Path where to save the image
        
    Returns:
        Path to the saved image
    """
    image.save(output_path)
    return output_path

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
        image = load_image(image)
    
    x1, y1, x2, y2 = coordinates
    return image.crop((x1, y1, x2, y2))

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
        image = load_image(image)
    
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

def highlight_region(
    image: Union[Image.Image, Path],
    coordinates: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 2
) -> Image.Image:
    """Highlight a region in an image by drawing a colored rectangle.
    
    Args:
        image: Image or path to image
        coordinates: Region to highlight (x1, y1, x2, y2)
        color: RGB color for the highlight
        width: Width of the highlight border
        
    Returns:
        Image with the highlighted region
    """
    if isinstance(image, Path):
        image = load_image(image)
    
    # Create a copy of the image
    result = image.copy()
    
    # Create a drawing context
    draw = ImageDraw.Draw(result)
    
    # Draw a rectangle around the region
    x1, y1, x2, y2 = coordinates
    for i in range(width):
        draw.rectangle((x1+i, y1+i, x2-i, y2-i), outline=color)
    
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
