"""Image management utilities for VQA tasks.

This module provides functionality for managing images and their views (crops)
in the workspace. It ensures that changes to one view are reflected in all other
views and the original image.
"""

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
        # Clear existing registry
        self.image_views = {}

        # Load original images
        for img_path in self.images_dir.glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"Registering image: {img_path}")
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
        view_id: Optional[str] = None,
        coordinates: Tuple[int, int, int, int] = None
    ) -> Path:
        """Create a new view (crop) of an image.

        Args:
            image_path: Path to the original image
            view_id: Optional identifier for the view
            coordinates: Crop coordinates (x1, y1, x2, y2)

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
            # If not found in registry, add it
            print(f"Image not found in registry, adding: {image_path}")
            self.image_views[image_path] = {}
            found_path = image_path

        # Use the found path for the rest of the function
        image_path = found_path

        # Generate view ID if not provided
        if view_id is None:
            view_id = f"view_{len(self.image_views[image_path]) + 1}"

        # Validate coordinates
        if coordinates is None:
            raise ValueError("Coordinates must be provided")

        if len(coordinates) != 4:
            raise ValueError(f"Invalid coordinates: {coordinates}. Must be a tuple of (x1, y1, x2, y2)")

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
        # Normalize the path
        if not view_path.is_absolute():
            view_path = self.views_dir / view_path.name

        # Find the view
        view_obj = None
        original_path = None

        # First try to find by path
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                # Compare by name to handle path differences
                if v.view_path.name == view_path.name:
                    view_obj = v
                    original_path = img_path
                    view_path = v.view_path  # Use the registered path
                    break
            if view_obj:
                break

        # If not found, try to find by view ID
        if not view_obj:
            view_id = str(view_path)
            for img_path, views in self.image_views.items():
                if view_id in views:
                    view_obj = views[view_id]
                    original_path = img_path
                    view_path = view_obj.view_path
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

    def is_fully_black(self, image_path: Path) -> bool:
        """Check if an image is fully black.

        Args:
            image_path: Path to the image to check

        Returns:
            True if the image is fully black, False otherwise
        """
        img = Image.open(image_path)
        img_array = np.array(img)
        return np.all(img_array == 0)

    def find_smallest_view(self, original_image_path: Path) -> Optional[Path]:
        """Find the smallest view (least pixels) for an image.

        Args:
            original_image_path: Path to the original image

        Returns:
            Path to the smallest view, or None if there are no views
        """
        if original_image_path not in self.image_views:
            return None

        views = self.image_views[original_image_path]
        if not views:
            return None

        # Find the view with the smallest area (width * height)
        smallest_view = None
        smallest_area = float('inf')

        for view_id, view in views.items():
            # Skip fully black views
            if self.is_fully_black(view.view_path):
                continue

            # Calculate area
            width = view.coordinates[2] - view.coordinates[0]
            height = view.coordinates[3] - view.coordinates[1]
            area = width * height

            if area < smallest_area:
                smallest_area = area
                smallest_view = view.view_path

        return smallest_view

    def delete_fully_black_views(self, original_image_path: Path) -> List[Path]:
        """Delete all fully black views for an image.

        Args:
            original_image_path: Path to the original image

        Returns:
            List of paths to deleted views
        """
        if original_image_path not in self.image_views:
            return []

        deleted_views = []
        views_to_delete = []

        # Find all fully black views
        for view_id, view in self.image_views[original_image_path].items():
            if self.is_fully_black(view.view_path):
                views_to_delete.append((view_id, view.view_path))

        # Delete the views
        for view_id, view_path in views_to_delete:
            # Remove from registry
            del self.image_views[original_image_path][view_id]
            # Delete the file
            view_path.unlink(missing_ok=True)
            deleted_views.append(view_path)

        return deleted_views

    def blackout_view(self, view_path: Path) -> List[Path]:
        """Blackout a view, marking it as analyzed.

        Args:
            view_path: Path to the view to blackout

        Returns:
            List of paths to all updated images (original + views)
        """
        # Normalize the path
        if not view_path.is_absolute():
            view_path = self.views_dir / view_path.name

        # Check if the file exists
        if not view_path.exists():
            # Try to find the view by name
            for img_path, views in self.image_views.items():
                for v_id, v in views.items():
                    if v.view_path.name == view_path.name:
                        view_path = v.view_path
                        break
                if view_path.exists():
                    break

            # If still not found, try to find by view ID
            if not view_path.exists():
                view_id = str(view_path)
                for img_path, views in self.image_views.items():
                    if view_id in views:
                        view_path = views[view_id].view_path
                        break

        if not view_path.exists():
            raise ValueError(f"View not found: {view_path}")

        # Find the original image for this view
        original_image_path = None
        view_id_to_delete = None
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == view_path:
                    original_image_path = img_path
                    view_id_to_delete = v_id
                    break
            if original_image_path:
                break

        # Load the view
        view_img = Image.open(view_path)

        # Create a black image of the same size
        black_img = Image.new('RGB', view_img.size, (0, 0, 0))

        # Update the view with the black image
        updated_paths = self.update_view(view_path, black_img)

        # If we found the original image, delete the blacked out view and find the smallest view
        if original_image_path and view_id_to_delete:
            # Delete the view we just blacked out
            if view_id_to_delete in self.image_views[original_image_path]:
                view_to_delete = self.image_views[original_image_path][view_id_to_delete].view_path
                del self.image_views[original_image_path][view_id_to_delete]
                view_to_delete.unlink(missing_ok=True)
                print(f"Deleted blacked out view: {view_to_delete.name}")

            # Delete any other fully black views
            deleted_views = self.delete_fully_black_views(original_image_path)
            if deleted_views:
                print(f"Deleted {len(deleted_views)} additional fully black views")

            # Find the smallest view
            smallest_view = self.find_smallest_view(original_image_path)
            if smallest_view:
                print(f"Smallest remaining view: {smallest_view.name}")
            else:
                print("No remaining views after blackout")

        return updated_paths

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
                # If not found in registry, add it
                print(f"Image not found in registry for list_views, adding: {image_path}")
                self.image_views[image_path] = {}
                found_path = image_path

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
        # Find the view by path
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

        # Try to find by view ID
        view_id = str(view_path)
        for img_path, views in self.image_views.items():
            if view_id in views:
                v = views[view_id]
                img = Image.open(v.view_path)
                return {
                    "view_id": v.view_id,
                    "original_image": str(v.original_image_path),
                    "coordinates": v.coordinates,
                    "size": img.size,
                    "path": str(v.view_path)
                }

        raise ValueError(f"View not found: {view_path}")
