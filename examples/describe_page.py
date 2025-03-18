#!/usr/bin/env python3
"""
Example script showing how to use Navi to simply capture a screenshot
of a webpage and have the LLM describe what it sees.
"""

import os
import sys
import asyncio
import argparse
from urllib.parse import urlparse

# Add the parent directory to the path so we can import navi
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navi.agent import NaviAgent
import config

async def describe_webpage(url, save_screenshot=False, model_name=None, stream=False):
    """Capture a screenshot of a webpage and have the LLM describe it."""
    # Validate URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        
    print(f"Navigating to {url}...")
    
    # Set up the agent
    agent = NaviAgent(
        headless=True,  # Run headless for this simple task
        model_name=model_name or config.MODEL_NAME,
        save_history=save_screenshot,
        stream_output=stream
    )
    
    try:
        # Initialize browser
        await agent.setup()
        
        # Navigate to the URL
        await agent.page.goto(url)
        await agent.page.wait_for_load_state("networkidle")
        
        print("Capturing screenshot...")
        
        # Get a description of the current page
        if stream:
            print("\nGenerating description: ", end="", flush=True)
            description = await agent.describe_current_page()
            print()  # Add a newline after streaming output
        else:
            description = await agent.describe_current_page()
        
        # Save screenshot if requested
        if save_screenshot:
            # Create a unique task ID
            agent.task_id = agent.task_id or f"page_desc_{int(asyncio.get_event_loop().time())}"
            
            # Get screenshot and save it
            screenshot_base64 = await agent.capture_screenshot()
            
            # Create a filename from the URL
            parsed_url = urlparse(url)
            filename = f"{parsed_url.netloc.replace('.', '_')}.png"
            
            # Create output directory
            output_dir = os.path.join("examples", "screenshots")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the screenshot
            from navi.utils import save_screenshot
            await save_screenshot(
                agent.task_id,
                1,  # Step 1
                screenshot_base64,
                output_dir
            )
            
            print(f"Screenshot saved to: {os.path.join(output_dir, agent.task_id, 'screenshots', 'step_1.png')}")
        
        # Print the description if not streaming
        if not stream:
            print("\n--- Page Description ---\n")
            print(description)
        
    finally:
        # Clean up resources
        await agent.teardown()

async def main():
    """Run the page description example."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Capture screenshot and describe webpage")
    parser.add_argument("url", help="URL of the webpage to describe")
    parser.add_argument("--save", action="store_true", help="Save the screenshot to a file")
    parser.add_argument("--model", help="Ollama model to use", default=None)
    parser.add_argument("--stream", action="store_true", help="Stream LLM output in real-time")
    args = parser.parse_args()
    
    # Call the describe function
    await describe_webpage(args.url, args.save, args.model, args.stream)

if __name__ == "__main__":
    # Create output directory if needed
    os.makedirs("examples/screenshots", exist_ok=True)
    
    # Run the script
    asyncio.run(main()) 