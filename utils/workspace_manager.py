"""Workspace management utilities for VQA tasks.

This module provides functionality for managing the workspace, including images and their views (crops).
Instead of creating physical files for each view, we now store only the coordinates of the view in the original image.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
import base64
from io import BytesIO


class ImageView:
    """Represents a view (crop) of an original image."""

    def __init__(
        self,
        view_id: str,
        original_image_path: Path,
        coordinates: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
    ):
        """Initialize an image view.

        Args:
            view_id: Unique identifier for this view
            original_image_path: Path to the original image
            coordinates: Crop coordinates (x1, y1, x2, y2)
        """
        self.view_id = view_id
        self.original_image_path = original_image_path
        self.coordinates = coordinates
        # Generate a virtual path for compatibility with existing code
        self.view_path = self._generate_virtual_path()

    def _generate_virtual_path(self) -> Path:
        """Generate a virtual path for this view.

        This is used for compatibility with existing code that expects a view_path.
        The path is not an actual file on disk, but a virtual path that uniquely
        identifies this view.

        Returns:
            A virtual path for this view
        """
        x1, y1, x2, y2 = self.coordinates
        original_name = self.original_image_path.stem
        coords_str = f"{x1}_{y1}_{x2}_{y2}"
        view_filename = f"{original_name}__{self.view_id}__{coords_str}{self.original_image_path.suffix}"
        # Use the views directory from the original image path
        views_dir = self.original_image_path.parent.parent / "views"
        return views_dir / view_filename

    def get_region(self) -> Tuple[int, int, int, int]:
        """Get the region coordinates of this view in the original image."""
        return self.coordinates

    def get_cropped_image(self) -> Image.Image:
        """Get the cropped image for this view.

        Returns:
            The cropped image
        """
        # Load the original image
        img = Image.open(self.original_image_path)
        # Crop to the view coordinates
        return img.crop(self.coordinates)

    def get_base64_url(self) -> str:
        """Get the base64 URL for this view.

        Returns:
            Base64 URL for the view
        """
        # Get the cropped image
        img = self.get_cropped_image()
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"


class WorkspaceManager:
    """Manages the workspace, including images and their views for VQA tasks."""

    def __init__(self, root: Path, container_workspace: Optional[Path] = None):
        """Initialize the workspace manager.

        Args:
            root: Root directory of the workspace
            container_workspace: Optional container workspace path
        """
        self.root = root
        self.container_workspace = container_workspace

        # Image management
        self.images_dir = root / "images"
        self.views_dir = root / "views"

        # Ensure directories exist
        self.images_dir.mkdir(exist_ok=True)
        self.views_dir.mkdir(exist_ok=True)

        # Track original images and their views
        # {original_image_path: {view_id: ImageView}}
        self.image_views: Dict[Path, Dict[str, ImageView]] = {}

        # Load existing images
        self._load_existing_images()

    def workspace_path(self, path: Path | str) -> Path:
        """Given a path, possibly in a container workspace, return the absolute local path."""
        path = Path(path)
        if not path.is_absolute():
            return self.root / path
        if self.container_workspace and path.is_relative_to(self.container_workspace):
            return self.root / path.relative_to(self.container_workspace)
        return path

    def container_path(self, path: Path | str) -> Path:
        """Given a path, possibly in the local workspace, return the absolute container path.
        If there is no container workspace, return the absolute local path.
        """
        path = Path(path)
        if not path.is_absolute():
            if self.container_workspace:
                return self.container_workspace / path
            else:
                return self.root / path
        if self.container_workspace and path.is_relative_to(self.root):
            return self.container_workspace / path.relative_to(self.root)
        return path

    def _load_existing_images(self):
        """Load existing images from the workspace.

        Note: We no longer load views from disk since they are now stored as coordinates only.
        """
        # Clear existing registry
        self.image_views = {}

        # Load original images
        for img_path in self.images_dir.glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"Registering image: {img_path}")
                self.image_views[img_path] = {}

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

    def list_images(self) -> List[Path]:
        """List all original images in the workspace.

        Returns:
            List of paths to original images
        """
        return list(self.image_views.keys())

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
            Path to the virtual view (not an actual file)
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

        # Create the view object (no physical file is created)
        view_obj = ImageView(
            view_id=view_id,
            original_image_path=image_path,
            coordinates=coordinates
        )

        # Register the view
        self.image_views[image_path][view_id] = view_obj

        # Return the virtual path
        return view_obj.view_path

    def blackout_view(self, view_path: Path) -> List[Path]:
        """Blackout a view, marking it as analyzed.

        Args:
            view_path: Path to the view to blackout

        Returns:
            List of paths to all updated images (original + views)
        """
        # Find the view in our registry
        view_obj = None
        original_image_path = None
        view_id_to_delete = None

        # Try to find by path or name
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == view_path or v.view_path.name == view_path.name:
                    view_obj = v
                    original_image_path = img_path
                    view_id_to_delete = v_id
                    break
            if view_obj:
                break

        # If still not found, try to find by view ID
        if not view_obj:
            view_id = str(view_path)
            for img_path, views in self.image_views.items():
                if view_id in views:
                    view_obj = views[view_id]
                    original_image_path = img_path
                    view_id_to_delete = view_id
                    break

        if not view_obj or not original_image_path:
            raise ValueError(f"View not found: {view_path}")

        # Get the coordinates of the view
        x1, y1, x2, y2 = view_obj.coordinates

        # Load the original image
        original_img = Image.open(original_image_path)

        # Create a black rectangle in the original image at the view coordinates
        draw = ImageDraw.Draw(original_img)
        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))

        # Save the modified original image
        original_img.save(original_image_path)

        # Remove the view from our registry
        if view_id_to_delete in self.image_views[original_image_path]:
            del self.image_views[original_image_path][view_id_to_delete]
            print(f"Removed blacked out view: {view_obj.view_path.name}")

        # Return the path to the updated original image
        return [original_image_path]

    def get_view_info(self, view_path: Path) -> Dict:
        """Get information about a view.

        Args:
            view_path: Path to the view (virtual path)

        Returns:
            Dictionary with view information
        """
        # Find the view by path or name
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == view_path or v.view_path.name == view_path.name:
                    # Get the cropped image size
                    x1, y1, x2, y2 = v.coordinates
                    size = (x2 - x1, y2 - y1)

                    return {
                        "view_id": v.view_id,
                        "original_image": str(v.original_image_path),
                        "coordinates": v.coordinates,
                        "size": size,
                        "path": str(v.view_path)
                    }

        # Try to find by view ID
        view_id = str(view_path)
        for img_path, views in self.image_views.items():
            if view_id in views:
                v = views[view_id]
                # Get the cropped image size
                x1, y1, x2, y2 = v.coordinates
                size = (x2 - x1, y2 - y1)

                return {
                    "view_id": v.view_id,
                    "original_image": str(v.original_image_path),
                    "coordinates": v.coordinates,
                    "size": size,
                    "path": str(v.view_path)
                }

        raise ValueError(f"View not found: {view_path}")

    def is_view_registered(self, view_path: Path) -> bool:
        """Check if a view is registered in the workspace manager.

        Args:
            view_path: Path to the view to check

        Returns:
            True if the view is registered, False otherwise
        """
        # Normalize the path
        if not isinstance(view_path, Path):
            view_path = Path(view_path)

        if not view_path.is_absolute():
            view_path = self.views_dir / view_path.name

        # Check if the view is registered
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == view_path or v.view_path.name == view_path.name:
                    return True
        return False

    def register_view(self, original_path: str, coordinates: list, view_id: Optional[str] = None) -> Path:
        """Register a view with the workspace manager.

        Args:
            original_path: Path to the original image
            coordinates: Crop coordinates (x1, y1, x2, y2)
            view_id: Optional identifier for the view

        Returns:
            Path to the virtual view
        """
        # Convert path to Path object if it's not already
        if not isinstance(original_path, Path):
            original_path = Path(original_path)

        # Make sure the original path is absolute
        if not original_path.is_absolute():
            original_path = self.images_dir / original_path.name

        # Generate view ID if not provided
        if view_id is None:
            view_id = f"view_{len(self.image_views.get(original_path, {})) + 1}"

        # Make sure coordinates is a tuple
        if isinstance(coordinates, list):
            coordinates = tuple(coordinates)

        # Create the view object
        view_obj = ImageView(
            view_id=view_id,
            original_image_path=original_path,
            coordinates=coordinates
        )

        # Register the original image if not already registered
        if original_path not in self.image_views:
            self.image_views[original_path] = {}

        # Register the view
        self.image_views[original_path][view_id] = view_obj

        return view_obj.view_path

    def is_view(self, path: Path) -> bool:
        """Check if a path is a view (not an original image).

        Args:
            path: Path to check

        Returns:
            True if the path is a view, False otherwise
        """
        # Normalize the path
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_absolute():
            # Check if it's in the views directory
            if (self.views_dir / path.name).exists():
                path = self.views_dir / path.name
            # Check if it's in the images directory
            elif (self.images_dir / path.name).exists():
                path = self.images_dir / path.name

        # Check if the path is in the views directory
        if path.parent == self.views_dir:
            return True

        # Check if the path is registered as a view
        for img_path, views in self.image_views.items():
            for v_id, v in views.items():
                if v.view_path == path or v.view_path.name == path.name:
                    return True

        return False

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
