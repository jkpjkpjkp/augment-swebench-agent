"""Test script for the text cleaner utility."""

from pathlib import Path
from utils.text_cleaner import clean_tool_result

def test_text_cleaner():
    """Test the text cleaner utility."""
    # Test workspace root
    workspace_root = Path("/mnt/home/jkp/hack/augment-swebench-agent/vqa_real_data_workspace")
    
    # Test select_image tool
    select_result = "Selected image at /mnt/home/jkp/hack/augment-swebench-agent/vqa_real_data_workspace/images/37_3.png\nSize: 800x600"
    cleaned_select = clean_tool_result("select_image", select_result, workspace_root)
    print(f"Original select_image result: {select_result}")
    print(f"Cleaned select_image result: {cleaned_select}")
    print()
    
    # Test crop_image tool
    crop_result = "Created new view at /mnt/home/jkp/hack/augment-swebench-agent/vqa_real_data_workspace/views/37_3__region_1__10_20_100_200.png\nCoordinates: (10, 20, 100, 200)"
    cleaned_crop = clean_tool_result("crop_image", crop_result, workspace_root)
    print(f"Original crop_image result: {crop_result}")
    print(f"Cleaned crop_image result: {cleaned_crop}")
    print()
    
    # Test blackout_image tool
    blackout_result = "Blacked out view at /mnt/home/jkp/hack/augment-swebench-agent/vqa_real_data_workspace/views/37_3__region_1__10_20_100_200.png\nCoordinates: (10, 20, 100, 200)"
    cleaned_blackout = clean_tool_result("blackout_image", blackout_result, workspace_root)
    print(f"Original blackout_image result: {blackout_result}")
    print(f"Cleaned blackout_image result: {cleaned_blackout}")
    print()
    
    # Test other tools
    other_result = "This is a result from another tool that doesn't need cleaning."
    cleaned_other = clean_tool_result("other_tool", other_result, workspace_root)
    print(f"Original other_tool result: {other_result}")
    print(f"Cleaned other_tool result: {cleaned_other}")

if __name__ == "__main__":
    test_text_cleaner()
