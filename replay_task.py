#!/usr/bin/env python3
"""
Replay a previously executed Navi task from a history file.
This is useful for debugging and demonstration purposes.
"""

import argparse
import json
import os
import time
import asyncio
from typing import Dict, List

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from navi.utils import format_action_for_display

# Configure argparse
parser = argparse.ArgumentParser(description="Replay a Navi task from history")
parser.add_argument("task_id", type=str, help="Task ID to replay")
parser.add_argument("--history-dir", type=str, default="history", 
                   help="Directory containing history files")
parser.add_argument("--delay", type=float, default=2.0,
                   help="Delay between steps in seconds")
parser.add_argument("--headless", action="store_true",
                   help="Run in headless mode (no browser UI)")
parser.add_argument("--screenshots", action="store_true",
                   help="Show screenshots instead of live browser")


async def replay_task(task_id: str, history_dir: str = "history", 
                     delay: float = 2.0, headless: bool = False, 
                     screenshots_mode: bool = False) -> None:
    """
    Replay a saved task by following the same actions that were taken.
    
    Args:
        task_id: The ID of the task to replay
        history_dir: Directory containing history files
        delay: Delay between steps in seconds
        headless: Whether to run in headless mode
        screenshots_mode: Whether to show screenshots instead of live browser
    """
    # Load the history file
    history_path = os.path.join(history_dir, task_id, "history.json")
    screenshots_dir = os.path.join(history_dir, task_id, "screenshots")
    
    if not os.path.exists(history_path):
        print(f"Error: History file not found at {history_path}")
        return
    
    with open(history_path, "r") as f:
        history_data = json.load(f)
    
    history = history_data.get("history", [])
    
    if not history:
        print("Error: No history data found in the history file")
        return
    
    print(f"Replaying task: {history[0].get('task', 'Unknown task')}")
    print(f"Steps to replay: {len(history)}")
    
    if screenshots_mode:
        await replay_with_screenshots(history, screenshots_dir, delay)
    else:
        await replay_with_browser(history, delay, headless)


async def replay_with_browser(history: List[Dict], delay: float, headless: bool) -> None:
    """Replay the task using a live browser."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()
        
        # Start with the first URL from the history
        start_entry = history[0]
        if "url" in start_entry:
            await page.goto(start_entry["url"])
            print(f"Step 0: Starting at {start_entry['url']}")
        
        # Wait for initial page load
        await asyncio.sleep(delay)
        
        # Replay each step
        for i, entry in enumerate(history):
            if i == 0:  # Skip the first entry (already handled)
                continue
                
            if entry.get("type") == "decision" and "action" in entry:
                action = entry["action"]
                action_type = action.get("action_type", "UNKNOWN")
                display_action = format_action_for_display(action)
                
                print(f"Step {i}: {display_action}")
                
                if "thinking" in action:
                    thinking = action["thinking"]
                    print(f"  Reasoning: {thinking[:100]}..." if len(thinking) > 100 else f"  Reasoning: {thinking}")
                
                if action_type == "CLICK" and "action_params" in action:
                    params = action["action_params"]
                    if "coordinates" in params:
                        x, y = params["coordinates"]
                        await page.mouse.click(x, y)
                    elif "selector" in params:
                        selector = params["selector"]
                        try:
                            await page.click(selector)
                        except:
                            print(f"  Failed to click element with selector: {selector}")
                
                elif action_type == "TYPE" and "action_params" in action:
                    params = action["action_params"]
                    if "text" in params:
                        text = params["text"]
                        if "coordinates" in params:
                            x, y = params["coordinates"]
                            await page.mouse.click(x, y)
                            await page.keyboard.type(text)
                        elif "selector" in params:
                            selector = params["selector"]
                            try:
                                await page.fill(selector, text)
                            except:
                                print(f"  Failed to type into element with selector: {selector}")
                
                elif action_type == "NAVIGATE" and "action_params" in action:
                    params = action["action_params"]
                    if "url" in params:
                        url = params["url"]
                        await page.goto(url)
                
                elif action_type == "SCROLL" and "action_params" in action:
                    params = action["action_params"]
                    direction = params.get("direction", "down")
                    amount = params.get("amount", 300)
                    if direction.lower() == "up":
                        await page.evaluate(f"window.scrollBy(0, -{amount})")
                    else:
                        await page.evaluate(f"window.scrollBy(0, {amount})")
                
                elif action_type == "COMPLETE":
                    if "result" in action:
                        print(f"  Result: {action['result']}")
            
            # Wait between steps
            await asyncio.sleep(delay)
        
        print("\nReplay completed!")
        print("Press Enter to close the browser...")
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        await page.close()
        await context.close()
        await browser.close()


async def replay_with_screenshots(history: List[Dict], screenshots_dir: str, delay: float) -> None:
    """Replay the task by showing the screenshots."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: OpenCV is required for screenshot mode. Install it with 'pip install opencv-python'")
        return
    
    # Start with the first screenshot
    for i, entry in enumerate(history):
        if entry.get("type") == "decision" and "step" in entry:
            step = entry["step"]
            screenshot_path = os.path.join(screenshots_dir, f"step_{step}.png")
            
            if os.path.exists(screenshot_path):
                # Show the screenshot
                img = cv2.imread(screenshot_path)
                if img is not None:
                    # Resize if too large
                    height, width = img.shape[:2]
                    if height > 900:
                        scale = 900 / height
                        img = cv2.resize(img, (int(width * scale), int(height * scale)))
                    
                    # Add step information
                    cv2.putText(img, f"Step {step}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Display the image
                    window_name = f"Navi Replay - Step {step}"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, img)
                    
                    # Show the action information
                    if "action" in entry:
                        action = entry["action"]
                        display_action = format_action_for_display(action)
                        print(f"Step {step}: {display_action}")
                        
                        if "thinking" in action:
                            thinking = action["thinking"]
                            print(f"  Reasoning: {thinking[:100]}..." if len(thinking) > 100 else f"  Reasoning: {thinking}")
                    
                    # Wait for key press or delay
                    key = cv2.waitKey(int(delay * 1000))
                    if key == 27:  # ESC key
                        break
                    
                    # Close the window after displaying
                    cv2.destroyWindow(window_name)
    
    print("\nReplay completed!")
    cv2.destroyAllWindows()


def main():
    args = parser.parse_args()
    asyncio.run(replay_task(
        args.task_id, 
        history_dir=args.history_dir, 
        delay=args.delay,
        headless=args.headless,
        screenshots_mode=args.screenshots
    ))


if __name__ == "__main__":
    main() 