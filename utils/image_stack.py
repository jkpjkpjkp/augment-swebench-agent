"""
Image Stack for tracking images sent to the model.

This module provides a stack-based mechanism for tracking which images
have been sent to the model and ensuring that the appropriate image is
sent after each tool operation.
"""

from pathlib import Path
from typing import List, Optional, Dict
import base64
from io import BytesIO
from PIL import Image

from utils.llm_client import TextPrompt


class ImageStack:
    """A stack for tracking images sent to the model."""

    def __init__(self):
        """Initialize an empty image stack."""
        self.stack: List[Path] = []
        self.image_cache: Dict[str, str] = {}  # Cache of image paths to base64 URLs

    def push(self, image_path: Path) -> None:
        """Push an image path onto the stack.
        
        Args:
            image_path: Path to the image
        """
        self.stack.append(image_path)

    def pop(self) -> Optional[Path]:
        """Pop the top image path from the stack.
        
        Returns:
            The path to the top image, or None if the stack is empty
        """
        if not self.stack:
            return None
        return self.stack.pop()

    def peek(self) -> Optional[Path]:
        """Peek at the top image path on the stack without removing it.
        
        Returns:
            The path to the top image, or None if the stack is empty
        """
        if not self.stack:
            return None
        return self.stack[-1]

    def clear(self) -> None:
        """Clear the image stack."""
        self.stack.clear()
        self.image_cache.clear()

    def get_image_url(self, image_path: Path) -> str:
        """Get the base64 URL for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Base64 URL for the image
        """
        # Check if we've already cached this image
        path_str = str(image_path)
        if path_str in self.image_cache:
            return self.image_cache[path_str]
        
        # Load the image and convert to base64
        try:
            img = Image.open(image_path)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_base64}"
            
            # Cache the URL
            self.image_cache[path_str] = img_url
            
            return img_url
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return ""

    def create_image_prompt(self, image_path: Path, text: str) -> TextPrompt:
        """Create a TextPrompt with an image.
        
        Args:
            image_path: Path to the image
            text: Text to include in the prompt
            
        Returns:
            TextPrompt with the image
        """
        prompt = TextPrompt(text=text)
        prompt.image_url = self.get_image_url(image_path)
        return prompt
