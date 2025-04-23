"""Image tools for VQA tasks.

This module provides tools for working with images in VQA tasks:
- CropTool: Create a new view by cropping with 4 coordinates
- SelectTool: Select the entire window/view
- BlackoutTool: Black out a selected region (marking it as analyzed)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import base64
from io import BytesIO

from utils.llm_client import TextPrompt

from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)
from utils.workspace_manager import WorkspaceManager
from utils.image_manager import ImageManager
from utils.image_utils import (
    load_image,
    save_image,
    crop_image,
    blackout_region,
    highlight_region,
    get_image_size,
)
from PIL import Image, ImageDraw


class CropTool(LLMTool):
    """Tool for creating a new view by cropping an image."""

    name = "crop_image"
    description = """Create a new view of an image by cropping it with 4 coordinates.

This tool allows you to create a new view (crop) of an image by specifying the coordinates
of the region to crop. The coordinates are specified as (x1, y1, x2, y2), where:
- (x1, y1) is the top-left corner of the crop region
- (x2, y2) is the bottom-right corner of the crop region

The coordinates are in pixels, with (0, 0) being the top-left corner of the image.
"""
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image to crop. Can be an original image or a view.",
            },
            "x1": {
                "type": "integer",
                "description": "X-coordinate of the top-left corner of the crop region.",
            },
            "y1": {
                "type": "integer",
                "description": "Y-coordinate of the top-left corner of the crop region.",
            },
            "x2": {
                "type": "integer",
                "description": "X-coordinate of the bottom-right corner of the crop region.",
            },
            "y2": {
                "type": "integer",
                "description": "Y-coordinate of the bottom-right corner of the crop region.",
            },
            "view_id": {
                "type": "string",
                "description": "Optional identifier for the new view. If not provided, a default ID will be generated.",
            },
        },
        "required": ["image_path", "x1", "y1", "x2", "y2"],
    }

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the crop tool.

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
        """Implement the crop tool.

        Args:
            tool_input: Dictionary containing the tool input parameters
            dialog_messages: Optional dialog messages for context

        Returns:
            ToolImplOutput containing the result of the operation
        """
        image_path = Path(tool_input["image_path"])
        x1 = tool_input["x1"]
        y1 = tool_input["y1"]
        x2 = tool_input["x2"]
        y2 = tool_input["y2"]
        view_id = tool_input.get("view_id")

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
        print(f"CropTool: Original path: {original_path}, Resolved path: {image_path}")

        try:
            # Check if the image exists
            if not image_path.exists():
                return ToolImplOutput(
                    tool_output=f"Error: Image not found at {image_path}",
                    tool_result_message=f"Error: Image not found at {image_path}",
                )

            # Create the view
            coordinates = (x1, y1, x2, y2)

            # Check if this is a view or an original image
            is_view = image_path.parent == self.image_manager.views_dir

            if is_view:
                # If it's a view, we need to find the original image and adjust coordinates
                for orig_path, views in self.image_manager.image_views.items():
                    for v_id, v in views.items():
                        if v.view_path == image_path:
                            # Found the view
                            vx1, vy1, vx2, vy2 = v.coordinates
                            # Adjust coordinates relative to the original image
                            adjusted_coords = (
                                vx1 + x1,
                                vy1 + y1,
                                vx1 + x2,
                                vy1 + y2,
                            )
                            view_path = self.image_manager.create_view(
                                orig_path, adjusted_coords, view_id
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
                # It's an original image
                view_path = self.image_manager.create_view(image_path, coordinates, view_id)

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
        return f"Creating a new view of {tool_input['image_path']} with coordinates ({tool_input['x1']}, {tool_input['y1']}, {tool_input['x2']}, {tool_input['y2']})"


class SelectTool(LLMTool):
    """Tool for selecting an entire image or view."""

    name = "select_image"
    description = """Select an entire image or view.

This tool allows you to select an entire image or view for further processing.
It returns information about the selected image, including its size and path.
"""
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

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the select tool.

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
        """Implement the select tool.

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
                        tool_output=f"Selected view at {image_path}\n"
                                   f"View ID: {view_info['view_id']}\n"
                                   f"Original image: {view_info['original_image']}\n"
                                   f"Coordinates: {view_info['coordinates']}\n"
                                   f"Size: {size[0]}x{size[1]}",
                        tool_result_message=f"Selected view at {image_path}",
                    )
                except ValueError:
                    # Not a registered view, just return basic info
                    pass

            # Return basic image information and encode the image as base64
            try:
                # Load the image and convert to base64
                img = load_image(image_path)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_url = f"data:image/png;base64,{img_base64}"
                # Log a sanitized version without the base64 data
                print(f"Created image URL for selected image (base64 data omitted)")

                # Add the image to the dialog
                # Note: We're not adding the image to the dialog messages directly
                # because the DialogMessages class doesn't have an add_user_message method
                # Instead, we'll return the image URL in the tool output
                # and let the agent handle it

                # Create a message about the image
                image_message = f"Selected image at {image_path}\nSize: {size[0]}x{size[1]}\nImage URL: {img_url}"

                # Store the image URL in a separate variable for the agent to use
                # but don't include it in the tool output to avoid logging it
                return ToolImplOutput(
                    tool_output=f"Selected image at {image_path}\n"
                               f"Size: {size[0]}x{size[1]}\n"
                               f"Image URL: [BASE64_IMAGE_DATA_OMITTED]",
                    tool_result_message=f"Selected image at {image_path}",
                    aux_data={"image_url": img_url}  # Store the actual URL in aux_data
                )
            except Exception as e:
                print(f"Error encoding image: {str(e)}")
                return ToolImplOutput(
                    tool_output=f"Selected image at {image_path}\n"
                               f"Size: {size[0]}x{size[1]}",
                    tool_result_message=f"Selected image at {image_path}",
                )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error selecting image: {str(e)}",
                tool_result_message=f"Error selecting image: {str(e)}",
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        return f"Selecting image at {tool_input['image_path']}"


class BlackoutTool(LLMTool):
    """Tool for blacking out an image or view."""

    name = "blackout_image"
    description = """Black out an image or view, marking it as analyzed.

This tool allows you to black out an entire image or view, indicating that
you have finished analyzing it. The blackout is applied to the original image
and all other views that overlap with the blacked-out region.
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
        print(f"BlackoutTool: Original path: {original_path}, Resolved path: {image_path}")

        try:
            # Check if the image exists
            if not image_path.exists():
                return ToolImplOutput(
                    tool_output=f"Error: Image not found at {image_path}",
                    tool_result_message=f"Error: Image not found at {image_path}",
                )

            # Check if coordinates are provided for partial blackout
            x1 = tool_input.get("x1")
            y1 = tool_input.get("y1")
            x2 = tool_input.get("x2")
            y2 = tool_input.get("y2")

            # If coordinates are provided, black out just that region
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                # Load the image
                img = load_image(image_path)

                # Create a black rectangle for the specified region
                draw = ImageDraw.Draw(img)
                draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0))

                # Save the image
                save_image(img, image_path)

                # Check if this is a view
                is_view = image_path.parent == self.image_manager.views_dir

                if is_view:
                    # Find the original image and update it
                    for img_path, views in self.image_manager.image_views.items():
                        for v_id, v in views.items():
                            if v.view_path == image_path:
                                # Found the view
                                # Adjust coordinates relative to the original image
                                orig_x1 = v.coordinates[0] + x1
                                orig_y1 = v.coordinates[1] + y1
                                orig_x2 = v.coordinates[0] + x2
                                orig_y2 = v.coordinates[1] + y2

                                # Load the original image
                                orig_img = load_image(img_path)

                                # Create a black rectangle for the specified region
                                draw = ImageDraw.Draw(orig_img)
                                draw.rectangle([(orig_x1, orig_y1), (orig_x2, orig_y2)], fill=(0, 0, 0))

                                # Save the original image
                                save_image(orig_img, img_path)

                                # Update all other views that overlap with this region
                                for other_v_id, other_v in views.items():
                                    if other_v.view_path != image_path:
                                        # Check if this view overlaps with the blacked out region
                                        other_x1, other_y1, other_x2, other_y2 = other_v.coordinates

                                        # Check for overlap
                                        if (orig_x1 < other_x2 and orig_x2 > other_x1 and
                                            orig_y1 < other_y2 and orig_y2 > other_y1):
                                            # There's overlap, update this view
                                            other_img = load_image(other_v.view_path)

                                            # Calculate the overlapping region in this view's coordinates
                                            overlap_x1 = max(0, orig_x1 - other_x1)
                                            overlap_y1 = max(0, orig_y1 - other_y1)
                                            overlap_x2 = min(other_x2 - other_x1, orig_x2 - other_x1)
                                            overlap_y2 = min(other_y2 - other_y1, orig_y2 - other_y1)

                                            # Create a black rectangle for the overlapping region
                                            draw = ImageDraw.Draw(other_img)
                                            draw.rectangle([(overlap_x1, overlap_y1), (overlap_x2, overlap_y2)], fill=(0, 0, 0))

                                            # Save the view
                                            save_image(other_img, other_v.view_path)
                                break

                    return ToolImplOutput(
                        tool_output=f"Blacked out region ({x1}, {y1}, {x2}, {y2}) in view at {image_path}",
                        tool_result_message=f"Blacked out region in view at {image_path}",
                    )
                else:
                    # It's an original image, update all views that overlap with this region
                    for view_id, view in self.image_manager.image_views.get(image_path, {}).items():
                        # Check if this view overlaps with the blacked out region
                        view_x1, view_y1, view_x2, view_y2 = view.coordinates

                        # Check for overlap
                        if (x1 < view_x2 and x2 > view_x1 and y1 < view_y2 and y2 > view_y1):
                            # There's overlap, update this view
                            view_img = load_image(view.view_path)

                            # Calculate the overlapping region in this view's coordinates
                            overlap_x1 = max(0, x1 - view_x1)
                            overlap_y1 = max(0, y1 - view_y1)
                            overlap_x2 = min(view_x2 - view_x1, x2 - view_x1)
                            overlap_y2 = min(view_y2 - view_y1, y2 - view_y1)

                            # Create a black rectangle for the overlapping region
                            draw = ImageDraw.Draw(view_img)
                            draw.rectangle([(overlap_x1, overlap_y1), (overlap_x2, overlap_y2)], fill=(0, 0, 0))

                            # Save the view
                            save_image(view_img, view.view_path)

                    return ToolImplOutput(
                        tool_output=f"Blacked out region ({x1}, {y1}, {x2}, {y2}) in image at {image_path}",
                        tool_result_message=f"Blacked out region in image at {image_path}",
                    )
            else:
                # No coordinates provided, black out the entire image/view
                # Check if this is a view
                is_view = image_path.parent == self.image_manager.views_dir

                if is_view:
                    # Black out the view
                    updated_paths = self.image_manager.blackout_view(image_path)

                    return ToolImplOutput(
                        tool_output=f"Blacked out view at {image_path}\n"
                                   f"Updated {len(updated_paths)} images/views",
                        tool_result_message=f"Blacked out view at {image_path}",
                    )
                else:
                    # It's an original image, create a black image of the same size
                    img = load_image(image_path)
                    black_img = Image.new('RGB', img.size, (0, 0, 0))
                    save_image(black_img, image_path)

                    # Update all views of this image
                    for view_id, view in self.image_manager.image_views.get(image_path, {}).items():
                        black_view = Image.new('RGB', (
                            view.coordinates[2] - view.coordinates[0],
                            view.coordinates[3] - view.coordinates[1]
                        ), (0, 0, 0))
                        save_image(black_view, view.view_path)

                    return ToolImplOutput(
                        tool_output=f"Blacked out image at {image_path} and all its views",
                        tool_result_message=f"Blacked out image at {image_path}",
                    )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error blacking out image: {str(e)}",
                tool_result_message=f"Error blacking out image: {str(e)}",
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        return f"Blacking out image at {tool_input['image_path']}"


class AddImageTool(LLMTool):
    """Tool for adding a new image to the workspace."""

    name = "add_image"
    description = """Add a new image to the workspace.

This tool allows you to add a new image to the workspace for VQA analysis.
The image must be accessible from the local filesystem.
"""
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image to add.",
            },
            "image_name": {
                "type": "string",
                "description": "Optional name for the image. If not provided, the original filename will be used.",
            },
        },
        "required": ["image_path"],
    }

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the add image tool.

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
        """Implement the add image tool.

        Args:
            tool_input: Dictionary containing the tool input parameters
            dialog_messages: Optional dialog messages for context

        Returns:
            ToolImplOutput containing the result of the operation
        """
        image_path = Path(tool_input["image_path"])
        image_name = tool_input.get("image_name")

        # Resolve the image path
        original_path = image_path
        if not image_path.is_absolute():
            image_path = self.workspace_manager.workspace_path(image_path)

        # Debug output
        print(f"AddImageTool: Original path: {original_path}, Resolved path: {image_path}")

        try:
            # Check if the image exists
            if not image_path.exists():
                return ToolImplOutput(
                    tool_output=f"Error: Image not found at {image_path}",
                    tool_result_message=f"Error: Image not found at {image_path}",
                )

            # Add the image to the workspace
            dest_path = self.image_manager.add_image(image_path, image_name)

            # Get image size
            size = get_image_size(dest_path)

            return ToolImplOutput(
                tool_output=f"Added image to workspace at {dest_path}\n"
                           f"Size: {size[0]}x{size[1]}",
                tool_result_message=f"Added image to workspace at {dest_path}",
            )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error adding image: {str(e)}",
                tool_result_message=f"Error adding image: {str(e)}",
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        return f"Adding image from {tool_input['image_path']} to workspace"


class ListImagesTool(LLMTool):
    """Tool for listing images and views in the workspace."""

    name = "list_images"
    description = """List all images and views in the workspace.

This tool allows you to list all original images and their views in the workspace.
"""
    input_schema = {
        "type": "object",
        "properties": {
            "show_views": {
                "type": "boolean",
                "description": "Whether to show views in addition to original images.",
                "default": True,
            },
        },
        "required": [],
    }

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize the list images tool.

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
        """Implement the list images tool.

        Args:
            tool_input: Dictionary containing the tool input parameters
            dialog_messages: Optional dialog messages for context

        Returns:
            ToolImplOutput containing the result of the operation
        """
        show_views = tool_input.get("show_views", True)

        try:
            # List original images
            images = self.image_manager.list_images()

            if not images:
                return ToolImplOutput(
                    tool_output="No images found in the workspace.",
                    tool_result_message="No images found in the workspace.",
                )

            result = "Images in workspace:\n"

            for img_path in images:
                size = get_image_size(img_path)
                result += f"- {img_path.name} ({size[0]}x{size[1]})\n"

                if show_views:
                    views = self.image_manager.list_views(img_path)
                    if views:
                        result += "  Views:\n"
                        for view_path in views:
                            view_info = self.image_manager.get_view_info(view_path)
                            result += f"  - {view_path.name} ({view_info['size'][0]}x{view_info['size'][1]}) - Coordinates: {view_info['coordinates']}\n"

            return ToolImplOutput(
                tool_output=result,
                tool_result_message=f"Listed {len(images)} images in workspace",
            )

        except Exception as e:
            return ToolImplOutput(
                tool_output=f"Error listing images: {str(e)}",
                tool_result_message=f"Error listing images: {str(e)}",
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Get a message to display when the tool starts.

        Args:
            tool_input: Dictionary containing the tool input parameters

        Returns:
            A message describing the operation
        """
        return "Listing images in workspace"
