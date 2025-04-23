"""Test script for the improved logging system."""

from pathlib import Path
from PIL import Image
import shutil

from utils.run_logger import RunLogger

def test_run_logger():
    """Test the RunLogger class."""
    # Create a test directory
    test_dir = Path("test_run_logs")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Initialize the run logger
    run_logger = RunLogger(base_log_dir=test_dir, run_id="test_run")

    # Create a test image
    img_dir = test_dir / "test_run" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    test_image = Image.new('RGB', (100, 100), (255, 0, 0))
    test_image_path = img_dir / "test_image.png"
    test_image.save(test_image_path)

    # Log a user message with an image
    run_logger.log_user_message("This is a test message with an image", image_path=test_image_path)

    # Log a model response
    run_logger.log_model_response("I see a red square in the image.")

    # Log a tool call
    run_logger.log_tool_call("test_tool", {"param1": "value1", "param2": 42})

    # Log a tool result with an image
    run_logger.log_tool_result("Tool executed successfully", image_path=test_image_path)

    # Log another user message
    run_logger.log_user_message("Another test message")

    # Log another model response
    run_logger.log_model_response("I understand your request.")

    # Finalize the run
    run_logger.finalize_run("Test run completed successfully")

    # Check if the HTML report was created
    html_report = test_dir / "test_run" / "index.html"
    if html_report.exists():
        print(f"HTML report created at: {html_report}")
        # Open the HTML report in the default browser
        import webbrowser
        # Convert to absolute path
        abs_path = html_report.absolute()
        print(f"Absolute path: {abs_path}")
        webbrowser.open(abs_path.as_uri())
    else:
        print("HTML report not created")

    # Check if the conversation JSON was created
    json_file = test_dir / "test_run" / "conversations" / "conversation_2.json"
    if json_file.exists():
        print(f"Conversation JSON created at: {json_file}")
    else:
        print("Conversation JSON not created")

    return html_report

if __name__ == "__main__":
    html_report = test_run_logger()
    print(f"Test completed. HTML report: {html_report}")
