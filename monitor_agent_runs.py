#!/usr/bin/env python3
"""
Monitor Agent Runs

This script monitors the latest agent runs and checks for any bugs related to image handling.
It analyzes the run logs and reports any issues found.
"""

import json
import logging
from pathlib import Path
import argparse
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitor_agent_runs.log')
    ]
)
logger = logging.getLogger('monitor_agent_runs')

def get_latest_run_dir(base_dir="agent_runs", n=1):
    """Get the N latest run directories.
    
    Args:
        base_dir: Base directory containing run directories
        n: Number of latest runs to return
        
    Returns:
        List of paths to the N latest run directories
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.error(f"Base directory {base_dir} does not exist")
        return []
    
    # Get all run directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    # Sort by creation time (newest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Return the N latest
    return run_dirs[:n]

def parse_run_log(log_path):
    """Parse a run log file.
    
    Args:
        log_path: Path to the run log file
        
    Returns:
        Dictionary containing parsed log information
    """
    if not log_path.exists():
        logger.error(f"Log file {log_path} does not exist")
        return {}
    
    log_info = {
        "user_messages": [],
        "model_responses": [],
        "tool_calls": [],
        "tool_results": [],
        "images": [],
        "errors": []
    }
    
    with open(log_path, "r") as f:
        for line in f:
            # Extract user messages
            if "User message" in line:
                log_info["user_messages"].append(line.strip())
            
            # Extract model responses
            elif "Model response" in line:
                log_info["model_responses"].append(line.strip())
            
            # Extract tool calls
            elif "Tool call" in line:
                log_info["tool_calls"].append(line.strip())
            
            # Extract tool results
            elif "Tool result" in line:
                log_info["tool_results"].append(line.strip())
            
            # Extract image information
            elif "image" in line.lower() and ("saved" in line.lower() or "copied" in line.lower()):
                log_info["images"].append(line.strip())
            
            # Extract errors
            elif "error" in line.lower() or "exception" in line.lower() or "warning" in line.lower():
                log_info["errors"].append(line.strip())
    
    return log_info

def check_for_image_bugs(run_dir):
    """Check for bugs related to image handling.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary containing bug information
    """
    bugs = {
        "missing_images": [],
        "duplicate_images": [],
        "blackout_issues": [],
        "crop_issues": [],
        "other_issues": []
    }
    
    # Check run log
    log_path = run_dir / "run.log"
    log_info = parse_run_log(log_path)
    
    # Check for errors
    for error in log_info["errors"]:
        if "image" in error.lower():
            if "crop" in error.lower():
                bugs["crop_issues"].append(error)
            elif "black" in error.lower():
                bugs["blackout_issues"].append(error)
            else:
                bugs["other_issues"].append(error)
    
    # Check for tool calls related to images
    image_tool_calls = []
    for tool_call in log_info["tool_calls"]:
        if "crop_image" in tool_call or "blackout_image" in tool_call or "switch_image" in tool_call:
            image_tool_calls.append(tool_call)
    
    # Check MLLM calls for image-related issues
    mllm_calls_dir = run_dir / "mllm_calls"
    if mllm_calls_dir.exists():
        for call_dir in mllm_calls_dir.iterdir():
            if call_dir.is_dir():
                call_data_path = call_dir / "call_data.json"
                if call_data_path.exists():
                    try:
                        with open(call_data_path, "r") as f:
                            call_data = json.load(f)
                        
                        # Check if images are properly referenced
                        if "images" in call_data:
                            for img_path in call_data["images"]:
                                if not Path(img_path).exists():
                                    bugs["missing_images"].append(f"Missing image referenced in MLLM call: {img_path}")
                        
                        # Check messages for image URLs
                        if "messages" in call_data:
                            for message_list in call_data["messages"]:
                                for message in message_list:
                                    if isinstance(message, dict) and "image_url" in message:
                                        # Check if it's a base64 image
                                        if message["image_url"] and message["image_url"].startswith("data:image"):
                                            # This is good - we're using base64 images
                                            pass
                                        else:
                                            bugs["other_issues"].append(f"Non-base64 image URL found: {message['image_url'][:30]}...")
                    except Exception as e:
                        bugs["other_issues"].append(f"Error parsing MLLM call data: {str(e)}")
    
    # Check for image files
    images_dir = run_dir / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        
        # Check if we have the expected number of images
        expected_images = len(log_info["images"])
        actual_images = len(image_files)
        
        if actual_images < expected_images:
            bugs["missing_images"].append(f"Expected {expected_images} images, but found {actual_images}")
        
        # Check for duplicate images (same content)
        # This is a simple check based on file size, could be improved
        image_sizes = {}
        for img_file in image_files:
            size = img_file.stat().st_size
            if size in image_sizes:
                image_sizes[size].append(img_file)
            else:
                image_sizes[size] = [img_file]
        
        for size, files in image_sizes.items():
            if len(files) > 1:
                bugs["duplicate_images"].append(f"Possible duplicate images (same size {size}): {[f.name for f in files]}")
    
    return bugs

def monitor_runs(interval=60, max_runs=5):
    """Monitor agent runs continuously.
    
    Args:
        interval: Interval in seconds between checks
        max_runs: Maximum number of runs to check each time
    """
    logger.info(f"Starting agent run monitoring (interval: {interval}s, max_runs: {max_runs})")
    
    # Keep track of already checked runs
    checked_runs = set()
    
    while True:
        try:
            # Get the latest runs
            latest_runs = get_latest_run_dir(n=max_runs)
            
            # Check each run that hasn't been checked yet
            for run_dir in latest_runs:
                if run_dir.name in checked_runs:
                    continue
                
                logger.info(f"Checking run: {run_dir.name}")
                
                # Check for bugs
                bugs = check_for_image_bugs(run_dir)
                
                # Report any bugs found
                has_bugs = any(len(bug_list) > 0 for bug_list in bugs.values())
                if has_bugs:
                    logger.warning(f"Found bugs in run {run_dir.name}:")
                    for bug_type, bug_list in bugs.items():
                        if bug_list:
                            logger.warning(f"  {bug_type}:")
                            for bug in bug_list:
                                logger.warning(f"    - {bug}")
                else:
                    logger.info(f"No bugs found in run {run_dir.name}")
                
                # Mark as checked
                checked_runs.add(run_dir.name)
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error during monitoring: {str(e)}")
            time.sleep(interval)

def check_single_run(run_id=None):
    """Check a single run for bugs.
    
    Args:
        run_id: ID of the run to check, or None to check the latest run
    """
    if run_id:
        run_dir = Path("agent_runs") / run_id
        if not run_dir.exists():
            logger.error(f"Run directory {run_dir} does not exist")
            return
    else:
        latest_runs = get_latest_run_dir(n=1)
        if not latest_runs:
            logger.error("No runs found")
            return
        run_dir = latest_runs[0]
    
    logger.info(f"Checking run: {run_dir.name}")
    
    # Check for bugs
    bugs = check_for_image_bugs(run_dir)
    
    # Report any bugs found
    has_bugs = any(len(bug_list) > 0 for bug_list in bugs.values())
    if has_bugs:
        logger.warning(f"Found bugs in run {run_dir.name}:")
        for bug_type, bug_list in bugs.items():
            if bug_list:
                logger.warning(f"  {bug_type}:")
                for bug in bug_list:
                    logger.warning(f"    - {bug}")
    else:
        logger.info(f"No bugs found in run {run_dir.name}")
    
    return bugs

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor agent runs for bugs")
    parser.add_argument("--monitor", action="store_true", help="Monitor runs continuously")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds between checks")
    parser.add_argument("--max-runs", type=int, default=5, help="Maximum number of runs to check each time")
    parser.add_argument("--run-id", type=str, help="Check a specific run by ID")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_runs(interval=args.interval, max_runs=args.max_runs)
    else:
        check_single_run(run_id=args.run_id)

if __name__ == "__main__":
    main()
