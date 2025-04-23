"""Utility functions for cleaning text before sending to the VLM."""

import re
from pathlib import Path
from typing import Optional

def clean_tool_result(tool_name: str, tool_result: str, workspace_root: Optional[Path] = None) -> str:
    """Clean tool result text to remove unnecessary information.
    
    Args:
        tool_name: Name of the tool that produced the result
        tool_result: Raw tool result text
        workspace_root: Optional workspace root path to remove from paths
        
    Returns:
        Cleaned tool result text
    """
    # Remove absolute paths if workspace_root is provided
    if workspace_root:
        workspace_str = str(workspace_root)
        tool_result = tool_result.replace(workspace_str, "")
    
    # Clean up based on tool type
    if tool_name == "select_image":
        # Extract just the image name from "Selected image at /path/to/image.png"
        match = re.search(r"Selected image at .*?([^/\\]+\.\w+)", tool_result)
        if match:
            image_name = match.group(1)
            return f"Selected image: {image_name}"
        
        # If no match, just return a generic message
        return "Selected an image for analysis."
    
    elif tool_name == "crop_image":
        # Extract just the view name from "Created new view at /path/to/view.png"
        match = re.search(r"Created new view at .*?([^/\\]+\.\w+)", tool_result)
        if match:
            view_name = match.group(1)
            return f"Created cropped view: {view_name}"
        
        # If no match, just return a generic message
        return "Created a cropped view for analysis."
    
    elif tool_name == "blackout_image":
        # Extract just the view name from "Blacked out view at /path/to/view.png"
        match = re.search(r"Blacked out (?:view|image) at .*?([^/\\]+\.\w+)", tool_result)
        if match:
            view_name = match.group(1)
            return f"Blacked out view: {view_name}"
        
        # If no match, just return a generic message
        return "Blacked out a region of the image."
    
    # For other tools, just return the original result
    return tool_result
