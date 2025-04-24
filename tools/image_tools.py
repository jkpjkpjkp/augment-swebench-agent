"""Image tools for VQA tasks.

This module provides tools for working with images in VQA tasks:
- CropTool: Create a new view by cropping with 4 coordinates
- SelectTool: Select the entire window/view
- BlackoutTool: Black out a selected region (marking it as analyzed)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import base64
from io import BytesIO


from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)
from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager
from utils.image_utils import (
    get_image_size,
)
from PIL import Image


class CropTool(LLMTool):
    """Tool for creating a new view by cropping an image."""

    name = "crop_image"
    description = """Create a new view by cropping the currently displayed image.

This tool allows you to create a new view (crop) of the currently displayed image by specifying a bounding box.

IMPORTANT: The bounding box is specified as a list of 4 integers [x1, y1, x2, y2], where:
- (x1, y1) is the top-left corner of the crop region
- (x2, y2) is the bottom-right corner of the crop region

The coordinates are NORMALIZED to a range of [0, 1000], where:
- (0, 0) is the top-left corner of the image
- (1000, 1000) is the bottom-right corner of the image

Example: To crop the top-left quarter of the image, use bbox=[0, 0, 500, 500]
"""
    input_schema = {
        "type": "object",
        "properties": {
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Bounding box coordinates [x1, y1, x2, y2] in normalized range [0, 1000]. Format is [left, top, right, bottom].",
            },
            "view_id": {
                "type": "string",
                "description": "Optional identifier for the new view. If not provided, a default ID will be generated.",
            },
        },
        "required": ["bbox"],
    }

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the crop tool.

        Args:
            workspace_manager: Workspace manager for resolving paths
        """
        super().__init__()
        self.workspace_manager = workspace_manager
        self.image_manager = ImageManager(workspace_manager.root)

        # Initialize class attributes if they don't exist
        if not hasattr(CropTool, "last_selected_image"):
            CropTool.last_selected_image = None
        if not hasattr(CropTool, "current_view_size"):
            CropTool.current_view_size = None
        if not hasattr(CropTool, "current_view_coords"):
            CropTool.current_view_coords = None

    def run_impl(
        self,
        tool_input: Dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        """Implement the crop tool.

        Args:
            tool_input: Dictionary containing the tool input parameters
            dialog_messages: Optional dialog messages for context

        Returns:
            ToolImplOutput containing the result of the operation
        """
        # Extract the bounding box coordinates
        bbox = tool_input["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return ToolImplOutput(
                tool_output="Error: bbox must be a list of 4 integers [x1, y1, x2, y2]",
                tool_result_message="Error: bbox must be a list of 4 integers [x1, y1, x2, y2]",
            )

        # Extract the coordinates
        x1, y1, x2, y2 = bbox
        view_id = tool_input.get("view_id")

        # Use the last selected image (currently displayed image)
        if not hasattr(CropTool, "last_selected_image") or CropTool.last_selected_image is None:
            # Try to find the last selected image from the image manager
            images = self.image_manager.list_images()
            if not images:
                return ToolImplOutput(
                    tool_output="Error: No image has been selected yet. Use switch_image first.",
                    tool_result_message="Error: No image has been selected yet. Use switch_image first.",
                )
            # Use the first image as a fallback
            image_path = images[0]
        else:
            image_path = CropTool.last_selected_image

        # Debug output
        print(f"CropTool: Using image: {image_path}, Bbox: {bbox}")

        try:
            # Check if the image exists
            if not image_path.exists():
                return ToolImplOutput(
                    tool_output=f"Error: Image not found at {image_path}",
                    tool_result_message=f"Error: Image not found at {image_path}",
                )

            # Get the dimensions of the CURRENT VIEW being displayed, not the original image
            if hasattr(CropTool, "current_view_size") and CropTool.current_view_size is not None:
                img_width, img_height = CropTool.current_view_size
            else:
                # If no current view is set, fall back to original image dimensions
                from utils.image_utils import get_image_size
                img_width, img_height = get_image_size(image_path)

            # Convert normalized coordinates [0, 1000] to pixel coordinates
            # These coordinates are now relative to the current view
            pixel_x1 = int(x1 * img_width / 1000)
            pixel_y1 = int(y1 * img_height / 1000)
            pixel_x2 = int(x2 * img_width / 1000)
            pixel_y2 = int(y2 * img_height / 1000)

            # If this is a view (not original image), adjust coordinates relative to parent view
            if hasattr(CropTool, "current_view_coords") and CropTool.current_view_coords is not None:
                parent_x1, parent_y1, _, _ = CropTool.current_view_coords
                # Adjust coordinates relative to parent view's position
                pixel_x1 += parent_x1
                pixel_y1 += parent_y1
                pixel_x2 += parent_x1
                pixel_y2 += parent_y1

            # Ensure coordinates are within image bounds
            pixel_x1 = max(0, min(pixel_x1, img_width - 1))
            pixel_y1 = max(0, min(pixel_y1, img_height - 1))
            pixel_x2 = max(pixel_x1 + 1, min(pixel_x2, img_width))
            pixel_y2 = max(pixel_y1 + 1, min(pixel_y2, img_height))

            # Create the view with pixel coordinates
            coordinates = (pixel_x1, pixel_y1, pixel_x2, pixel_y2)

            # Check if this is a view or an original image
            is_view = image_path.parent == self.image_manager.views_dir

            if is_view:
                # If it's a view, we need to find the original image and adjust coordinates
                for orig_path, views in self.image_manager.image_views.items():
                    for v_id, v in views.items():
                        if v.view_path == image_path:
                            # Found the view
                            view_x1, view_y1, view_x2, view_y2 = v.coordinates

                            # CRITICAL FIX: The coordinates are already relative to the current view
                            # We need to adjust them to be relative to the original image
                            # by adding the current view's coordinates
                            adjusted_coords = (
                                view_x1 + pixel_x1,  # Add current view's x1 offset
                                view_y1 + pixel_y1,  # Add current view's y1 offset
                                view_x1 + pixel_x2,  # Add current view's x1 offset (not x2)
                                view_y1 + pixel_y2,  # Add current view's y1 offset (not y2)
                            )

                            # Log the adjustment for debugging
                            print(f"CropTool: Adjusting coordinates from {coordinates} to {adjusted_coords} relative to original image")

                            view_path = self.image_manager.create_view(
                                orig_path, view_id, adjusted_coords
                            )
                            break
                    else:
                        continue
                    break
                else:
                    return ToolImplOutput(
                        tool_output=f"Error: Could not find view information for {image_path}",
                        tool_result_message=f"Error: Could not find view information for {image_path}",
                    )
            else:
                # It's an original image - no adjustment needed
                view_path = self.image_manager.create_view(image_path, view_id, coordinates)

            # Get information about the created view
            view_info = self.image_manager.get_view_info(view_path)

            return ToolImplOutput(
                tool_output=f"Created new view at {view_path}\n"
                           f"View ID: {view_info['view_id']}\n"
                           f"Original image: {view_info['original_image']}\n"
                           f"Coordinates: {view_info['coordinates']}\n"
                           f"Size: {view_info['size'][0]}x{view_info['size'][1]}",
                tool_result_message=f"Created new view at {view_path}",
            )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error creating view: {str(e)}",
                tool_result_message=f"Error creating view: {str(e)}",
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        bbox = tool_input.get("bbox", [0, 0, 0, 0])
        return f"Creating a new view with bounding box {bbox}"


class SwitchImageTool(LLMTool):
    """Tool for switching to a different image or view."""

    name = "switch_image"
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image or view to select.",
            },
        },
        "required": ["image_path"],
    }

    @property
    def description(self) -> str:
        """Dynamically generate the tool description with the list of available images."""
        # Get the list of all available images
        all_images = self.image_manager.list_images() if hasattr(self, 'image_manager') else []
        all_views = []

        if all_images:
            for img in all_images:
                if hasattr(self, 'image_manager'):
                    views = self.image_manager.list_views(img)
                    all_views.extend(views)

        # Create a formatted list of all images
        image_list = "Available images:\n"
        for img in all_images:
            image_list += f"- {img.name}\n"

        # Add views
        if all_views:
            image_list += "\nAvailable views:\n"
            for view in all_views:
                image_list += f"- {view.name}\n"

        # Create the description with the image list
        desc = f"""View a different crop.

{image_list}

This tool allows you to switch between different images and views. It returns information about the selected image, including its size and path, and displays the image for analysis.
After switching to an image, you can create views (crops) of specific regions using the crop_image tool."""
        return desc

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the switch image tool.

        Args:
            workspace_manager: Workspace manager for resolving paths
        """
        super().__init__()
        self.workspace_manager = workspace_manager
        self.image_manager = ImageManager(workspace_manager.root)

        # Store the last selected image path for use by other tools
        # This is a class variable shared across all instances
        if not hasattr(CropTool, "last_selected_image"):
            CropTool.last_selected_image = None
        if not hasattr(BlackoutTool, "last_selected_image"):
            BlackoutTool.last_selected_image = None

    def run_impl(
        self,
        tool_input: Dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        """Implement the switch image tool.

        Args:
            tool_input: Dictionary containing the tool input parameters
            dialog_messages: Optional dialog messages for context

        Returns:
            ToolImplOutput containing the result of the operation
        """
        image_path = Path(tool_input["image_path"])

        # Resolve the image path
        original_path = image_path

        # If it's already a Path object, convert to string first
        if isinstance(image_path, Path):
            image_path = str(image_path)

        image_path = Path(image_path)

        # Check if this is a view ID rather than a path
        path_str = str(image_path)
        view_found = False

        # Try to find a view with this ID
        for img_path, views in self.image_manager.image_views.items():
            if path_str in views:
                image_path = views[path_str].view_path
                view_found = True
                break

        if not view_found and not image_path.is_absolute():
            # Try different possible paths
            possible_paths = [
                image_path,  # As is
                self.workspace_manager.root / image_path,  # Relative to workspace root
                self.workspace_manager.root / "images" / image_path.name,  # In images directory
                self.workspace_manager.root / "views" / image_path.name,  # In views directory
            ]

            # Try each path
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break

        # Debug output
        print(f"SelectTool: Original path: {original_path}, Resolved path: {image_path}")

        try:
            # Check if the image exists
            if not image_path.exists():
                return ToolImplOutput(
                    tool_output=f"Error: Image not found at {image_path}",
                    tool_result_message=f"Error: Image not found at {image_path}",
                )

            # Get image information
            size = get_image_size(image_path)

            # Check if this is a view
            is_view = image_path.parent == self.image_manager.views_dir

            if is_view:
                # Try to get view information
                try:
                    view_info = self.image_manager.get_view_info(image_path)
                    return ToolImplOutput(
                        tool_output=f"Switched to view at {image_path}\n"
                                   f"View ID: {view_info['view_id']}\n"
                                   f"Original image: {view_info['original_image']}\n"
                                   f"Coordinates: {view_info['coordinates']}\n"
                                   f"Size: {size[0]}x{size[1]}",
                        tool_result_message=f"Switched to view at {image_path}",
                    )
                except ValueError:
                    # Not a registered view, just return basic info
                    pass

            # Return basic image information and encode the image as base64
            try:
                # Load the image and convert to base64
                img = Image.open(image_path)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_url = f"data:image/png;base64,{img_base64}"
                # Log a sanitized version without the base64 data
                print("Created image URL for selected image (base64 data omitted)")

                # Add the image to the dialog
                # Note: We're not adding the image to the dialog messages directly
                # because the DialogMessages class doesn't have an add_user_message method
                # Instead, we'll return the image URL in the tool output
                # and let the agent handle it

                # Create a message about the image
                f"Switched to image at {image_path}\nSize: {size[0]}x{size[1]}\nImage URL: {img_url}"

                # Store the image URL in a separate variable for the agent to use
                # but don't include it in the tool output to avoid logging it

                # Store the selected image path for use by other tools
                CropTool.last_selected_image = image_path
                BlackoutTool.last_selected_image = image_path

                # Set the current view size for the CropTool
                CropTool.current_view_size = size

                # If this is a view, set the current view coordinates
                if is_view:
                    try:
                        view_info = self.image_manager.get_view_info(image_path)
                        CropTool.current_view_coords = view_info['coordinates']
                    except Exception as e:
                        print(f"Warning: Could not set current view coordinates: {e}")
                else:
                    # If this is an original image, reset the current view coordinates
                    if hasattr(CropTool, "current_view_coords"):
                        delattr(CropTool, "current_view_coords")

                # Get the list of all available images for display
                all_images = self.image_manager.list_images()
                all_views = []
                for img in all_images:
                    views = self.image_manager.list_views(img)
                    all_views.extend(views)

                # Create a formatted list of all images, highlighting the current one
                image_list = "Available images:\n"
                for img in all_images:
                    if img == image_path:
                        image_list += f"→ {img.name} (SELECTED)\n"
                    else:
                        image_list += f"- {img.name}\n"

                # Add views
                if all_views:
                    image_list += "\nAvailable views:\n"
                    for view in all_views:
                        if view == image_path:
                            image_list += f"→ {view.name} (SELECTED)\n"
                        else:
                            image_list += f"- {view.name}\n"

                return ToolImplOutput(
                    tool_output=f"Switched to image at {image_path}\n"
                               f"Size: {size[0]}x{size[1]}\n\n"
                               f"{image_list}",
                    tool_result_message=f"Switched to image: {image_path.name}",
                    aux_data={"image_url": img_url}  # Store the actual URL in aux_data
                )
            except Exception as e:
                print(f"Error encoding image: {str(e)}")
                return ToolImplOutput(
                    tool_output=f"Switched to image at {image_path}\n"
                               f"Size: {size[0]}x{size[1]}",
                    tool_result_message=f"Switched to image at {image_path}",
                )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error switching to image: {str(e)}",
                tool_result_message=f"Error switching to image: {str(e)}",
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        return f"Switching to image at {tool_input['image_path']}"


class BlackoutTool(LLMTool):
    """Tool for blacking out an image or view."""

    name = "blackout_image"
    description = """Black out an image or view, marking it as analyzed.

This tool allows you to black out an entire image or view, indicating that
you have finished analyzing it. The blackout is applied to the original image
and all other views that overlap with the blacked-out region.

After blackout, the view is automatically deleted along with any other fully black views.
The smallest remaining view (with the least number of pixels) is identified for further analysis.
"""
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image or view to black out.",
            },
            "x1": {
                "type": "integer",
                "description": "Optional X coordinate of the top-left corner of the region to black out.",
            },
            "y1": {
                "type": "integer",
                "description": "Optional Y coordinate of the top-left corner of the region to black out.",
            },
            "x2": {
                "type": "integer",
                "description": "Optional X coordinate of the bottom-right corner of the region to black out.",
            },
            "y2": {
                "type": "integer",
                "description": "Optional Y coordinate of the bottom-right corner of the region to black out.",
            },
        },
        "required": ["image_path"],
    }

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the blackout tool.

        Args:
            workspace_manager: Workspace manager for resolving paths
        """
        super().__init__()
        self.workspace_manager = workspace_manager
        self.image_manager = ImageManager(workspace_manager.root)

    def run_impl(
        self,
        tool_input: Dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        """Implement the blackout tool.

        Args:
            tool_input: Dictionary containing the tool input parameters
            dialog_messages: Optional dialog messages for context

        Returns:
            ToolImplOutput containing the result of the operation
        """
        image_path = tool_input["image_path"]

        # Ensure path is relative to workspace
        if isinstance(image_path, str) and not image_path.startswith('/'):
            image_path = str(self.image_manager.views_dir / image_path)

        # Check if image exists
        if not Path(image_path).exists():
            return ToolImplOutput(
                tool_output=f"Error blacking out image: Image not found: {image_path}",
                tool_result_message="Failed to black out image - not found"
            )

        try:
            # Register the view if it's not already registered
            if isinstance(image_path, str) and image_path.endswith('.png'):
                view_name = Path(image_path).name
                original_image = view_name.split('__')[0] + '.png'
                original_path = str(self.image_manager.images_dir / original_image)

                # Parse coordinates from view name
                coords_str = view_name.split('__')[-1].replace('.png', '')
                coords = [int(x) for x in coords_str.split('_')]

                # Register view if not already registered
                if not self.image_manager.is_view_registered(image_path):
                    self.image_manager.register_view(
                        original_path,
                        image_path,
                        coords
                    )

            # Proceed with blackout
            # Convert to Path object if it's a string
            path_obj = Path(image_path) if isinstance(image_path, str) else image_path
            if self.image_manager.is_view(path_obj):
                updated_paths = self.image_manager.blackout_view(path_obj)
                return ToolImplOutput(
                    tool_output=f"Blacked out view at {path_obj}\n"
                               f"Updated {len(updated_paths)} images/views",
                    tool_result_message=f"Blacked out view at {path_obj}"
                )
            else:
                # Handle original image blackout
                img = Image.open(path_obj)
                black_img = Image.new('RGB', img.size, (0, 0, 0))
                black_img.save(path_obj)

                # Update all related views
                for view_id, view in self.image_manager.image_views.get(str(path_obj), {}).items():
                    black_view = Image.new('RGB', (
                        view.coordinates[2] - view.coordinates[0],
                        view.coordinates[3] - view.coordinates[1]
                    ), (0, 0, 0))
                    black_view.save(view.view_path)

                return ToolImplOutput(
                    tool_output=f"Blacked out image at {path_obj}",
                    tool_result_message=f"Blacked out image at {path_obj}"
                )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error blacking out image: {str(e)}",
                tool_result_message="Failed to black out image"
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        return f"Blacking out image at {tool_input['image_path']}"