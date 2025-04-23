"""Image management utilities for VQA tasks.

This module provides functionality for managing images and their views (crops)
in the workspace. It ensures that changes to one view are reflected in all other
views and the original image.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

class ImageView:
    """Represents a view (crop) of an original image."""

    def __init__(
        self,
        view_id: str,
        original_image_path: Path,
        coordinates: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        view_path: Path
    ):
        """Initialize an image view.

        Args:
            view_id: Unique identifier for this view
            original_image_path: Path to the original image
            coordinates: Crop coordinates (x1, y1, x2, y2)
            view_path: Path where this view is saved
        """
        self.view_id = view_id
        self.original_image_path = original_image_path
        self.coordinates = coordinates
        self.view_path = view_path

    def get_region(self) -> Tuple[int, int, int, int]:
        """Get the region coordinates of this view in the original image."""
        return self.coordinates


class ImageManager:
    """Manages images and their views for VQA tasks."""

    def __init__(self, workspace_root: Path):
        """Initialize the image manager.

        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = workspace_root
        self.images_dir = workspace_root / "images"
        self.views_dir = workspace_root / "views"

        # Ensure directories exist
        self.images_dir.mkdir(exist_ok=True)
        self.views_dir.mkdir(exist_ok=True)

        # Track original images and their views
        # {original_image_path: {view_id: ImageView}}
        self.image_views: Dict[Path, Dict[str, ImageView]] = {}

        # Load existing images and views
        self._load_existing_images()

    def _load_existing_images(self):
        """Load existing images and views from the workspace."""
        # Load original images
        for img_path in self.images_dir.glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.image_views[img_path] = {}

        # Load views
        for view_path in self.views_dir.glob("*"):
            if view_path.is_file() and view_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Parse view metadata from filename
                # Format: original_image_name__viewid__x1_y1_x2_y2.ext
                try:
                    parts = view_path.stem.split('__')
                    if len(parts) >= 3:
                        original_name = parts[0]
                        view_id = parts[1]
                        coords_str = parts[2]

                        # Parse coordinates
                        coords = tuple(map(int, coords_str.split('_')))
                        if len(coords) == 4:
                            original_path = self.images_dir / f"{original_name}{view_path.suffix}"
                            if original_path in self.image_views:
                                view = ImageView(
                                    view_id=view_id,
                                    original_image_path=original_path,
                                    coordinates=coords,
                                    view_path=view_path
                                )
                                self.image_views[original_path][view_id] = view
                except Exception as e:
                    print(f"Error loading view {view_path}: {e}")

    def add_image(self, image_path: Path, image_name: Optional[str] = None) -> Path:
        """Add a new image to the workspace.

        Args:
            image_path: Path to the image to add
            image_name: Optional name for the image (defaults to original filename)

        Returns:
            Path to the image in the workspace
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Use original filename if no name provided
        if image_name is None:
            image_name = image_path.name

        # Ensure image has proper extension
        if not any(image_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            image_name = f"{image_name}.png"

        # Copy image to workspace
        dest_path = self.images_dir / image_name

        # Load and save the image (ensures it's a valid image)
        img = Image.open(image_path)
        img.save(dest_path)

        # Register the image
        self.image_views[dest_path] = {}

        return dest_path

    def create_view(
        self,
        image_path: Path,
        coordinates: Tuple[int, int, int, int],
        view_id: Optional[str] = None
    ) -> Path:
        """Create a new view (crop) of an image.

        Args:
            image_path: Path to the original image
            coordinates: Crop coordinates (x1, y1, x2, y2)
            view_id: Optional identifier for the view

        Returns:
            Path to the created view
        """
        # Resolve image path
        if not image_path.is_absolute():
            image_path = self.images_dir / image_path

        # Check if the path exists
        if not image_path.exists():
            # Try to find the image by name in our images directory
            possible_path = self.images_dir / image_path.name
            if possible_path.exists():
                image_path = possible_path
            else:
                raise ValueError(f"Image not found: {image_path}")

        # Find the image in our registry by name
        found_path = None
        for img_path in self.image_views.keys():
            try:
                if img_path.name == image_path.name:
                    found_path = img_path
                    break
            except Exception:
                continue

        if found_path is None:
            raise ValueError(f"Image not found in workspace registry: {image_path}")

        # Use the found path for the rest of the function
        image_path = found_path

        # Generate view ID if not provided
        if view_id is None:
            view_id = f"view_{len(self.image_views[image_path]) + 1}"

        # Validate coordinates
        x1, y1, x2, y2 = coordinates
        img = Image.open(image_path)
        width, height = img.size

        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid coordinates {coordinates} for image of size {width}x{height}")

        # Create the view
        view = img.crop(coordinates)

        # Generate view filename: original_name__viewid__x1_y1_x2_y2.ext
        original_name = image_path.stem
        coords_str = f"{x1}_{y1}_{x2}_{y2}"
        view_filename = f"{original_name}__{view_id}__{coords_str}{image_path.suffix}"
        view_path = self.views_dir / view_filename

        # Save the view
        view.save(view_path)

        # Register the view
        view_obj = ImageView(
            view_id=view_id,
            original_image_path=image_path,
            coordinates=coordinates,
            view_path=view_path
        )
        self.image_views[image_path][view_id] = view_obj

        return view_path

    def update_view(
        self,
        view_path: Path,
        modified_image: Image.Image
    ) -> List[Path]:
        """Update a view and propagate changes to the original image and other views.

        Args:
            view_path: Path to the view to update
            modified_image: Modified image data

        Returns:
            List of paths to all updated images (original + views)
        """
        # Find the view
        view_obj = None
        original_path = None

        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == view_path:
                    view_obj = v
                    original_path = img_path
                    break
            if view_obj:
                break

        if not view_obj or not original_path:
            raise ValueError(f"View not found: {view_path}")

        # Get view coordinates
        x1, y1, x2, y2 = view_obj.coordinates

        # Load original image
        original_img = Image.open(original_path)

        # Ensure modified image has the same size as the view
        expected_size = (x2 - x1, y2 - y1)
        if modified_image.size != expected_size:
            modified_image = modified_image.resize(expected_size)

        # Update original image
        original_img.paste(modified_image, (x1, y1))
        original_img.save(original_path)

        # Update all views
        updated_paths = [original_path]
        for v_id, v in self.image_views[original_path].items():
            if v.view_path != view_path:  # Skip the view we're updating
                vx1, vy1, vx2, vy2 = v.coordinates

                # Check if this view overlaps with the modified view
                if (x1 < vx2 and x2 > vx1 and y1 < vy2 and y2 > vy1):
                    # There's overlap, update this view
                    view_img = original_img.crop((vx1, vy1, vx2, vy2))
                    view_img.save(v.view_path)
                    updated_paths.append(v.view_path)

        # Save the modified view itself
        modified_image.save(view_path)
        updated_paths.append(view_path)

        return updated_paths

    def blackout_view(self, view_path: Path) -> List[Path]:
        """Blackout a view, marking it as analyzed.

        Args:
            view_path: Path to the view to blackout

        Returns:
            List of paths to all updated images (original + views)
        """
        # Load the view
        view_img = Image.open(view_path)

        # Create a black image of the same size
        black_img = Image.new('RGB', view_img.size, (0, 0, 0))

        # Update the view with the black image
        return self.update_view(view_path, black_img)

    def list_images(self) -> List[Path]:
        """List all original images in the workspace.

        Returns:
            List of paths to original images
        """
        return list(self.image_views.keys())

    def list_views(self, image_path: Optional[Path] = None) -> List[Path]:
        """List all views in the workspace.

        Args:
            image_path: Optional path to an original image to filter views

        Returns:
            List of paths to views
        """
        if image_path:
            if not image_path.is_absolute():
                image_path = self.images_dir / image_path

            # Check if the path exists
            if not image_path.exists():
                # Try to find the image by name in our images directory
                possible_path = self.images_dir / image_path.name
                if possible_path.exists():
                    image_path = possible_path
                else:
                    return []

            # Find the image in our registry by name
            found_path = None
            for img_path in self.image_views.keys():
                try:
                    if img_path.name == image_path.name:
                        found_path = img_path
                        break
                except Exception:
                    continue

            if found_path is None:
                return []

            return [v.view_path for v in self.image_views[found_path].values()]
        else:
            views = []
            for img_views in self.image_views.values():
                views.extend(v.view_path for v in img_views.values())
            return views

    def get_view_info(self, view_path: Path) -> Dict:
        """Get information about a view.

        Args:
            view_path: Path to the view

        Returns:
            Dictionary with view information
        """
        # Find the view
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == view_path:
                    img = Image.open(view_path)
                    return {
                        "view_id": v.view_id,
                        "original_image": str(v.original_image_path),
                        "coordinates": v.coordinates,
                        "size": img.size,
                        "path": str(view_path)
                    }

        raise ValueError(f"View not found: {view_path}")
