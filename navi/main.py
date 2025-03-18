#!/usr/bin/env python3
"""
Navi - A web navigation AI agent that uses Ollama for decision making.

This module provides the main entry point and CLI for Navi.
"""

import asyncio
import argparse
import datetime
import os
import pathlib
import logging
from typing import Dict

from navi.agent import NaviAgent
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("navi")

async def run_task(args):
    """Run a navigation task using Navi."""
    # Override config settings with command line arguments if provided
    headless = args.headless if args.headless else config.HEADLESS
    model_name = args.model
    task_description = args.task
    starting_url = args.start_url
    save_history = not args.no_history
    history_dir = args.history_dir
    stream_output = args.stream
    use_element_detection = not args.no_element_detection
    max_elements = args.max_elements
    save_raw_screenshots = args.save_raw_screenshots if hasattr(args, 'save_raw_screenshots') else False
    
    logger.info(f"Starting Navi with task: {task_description}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Headless mode: {headless}")
    logger.info(f"Starting URL: {starting_url}")
    logger.info(f"Save history: {save_history}")
    logger.info(f"Stream output: {stream_output}")
    logger.info(f"Element detection: {use_element_detection}")
    if use_element_detection:
        logger.info(f"Max elements: {max_elements}")
    if save_raw_screenshots:
        logger.info("Raw screenshot saving enabled")
    
    # Create history directory if needed
    if save_history:
        pathlib.Path(history_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize agent
    agent = NaviAgent(
        headless=headless,
        screenshot_width=args.screenshot_width,
        browser_height=args.browser_height,
        ollama_base_url=args.ollama_url,
        model_name=model_name,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        user_agent=args.user_agent,
        llm_temperature=args.temperature,
        llm_timeout=args.timeout,
        save_history=save_history,
        history_dir=history_dir,
        stream_output=stream_output,
        use_element_detection=use_element_detection,
        max_elements=max_elements
    )
    
    try:
        # Set up browser
        await agent.setup()
        
        # Run the task
        result = await agent.run_task(
            task_description, 
            starting_url,
            save_raw_screenshots=save_raw_screenshots
        )
        
        # Display results
        print("\n--- Task Results ---")
        print(f"Success: {result['success']}")
        print(f"Steps taken: {result['steps_taken']}")
        print(f"Final URL: {result['final_url']}")
        print(f"Error count: {result['error_count']}")
        
        if result['success']:
            print("\nTask successfully completed!")
            if 'result' in result['history'][-1]:
                print(f"Result: {result['history'][-1]['result']}")
        else:
            print("\nTask not completed.")
            if 'reason' in result:
                print(f"Reason: {result['reason']}")
                
        if save_history:
            print(f"\nHistory saved to: {os.path.join(history_dir, result['task_id'])}")
        
        return result
    
    finally:
        # Clean up resources
        await agent.teardown()

def main():
    """Main entry point for the Navi CLI."""
    parser = argparse.ArgumentParser(description="Navi - Web Navigation AI Agent")
    
    # Core arguments
    parser.add_argument("--task", type=str, help="Task description in natural language", 
                        default="Search for 'latest AI research papers' on Google and find a recent paper about LLMs.")
    parser.add_argument("--start-url", type=str, help="Starting URL for the task", 
                        default=config.DEFAULT_START_URL)
    
    # Browser settings
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no browser UI)")
    parser.add_argument("--screenshot-width", type=int, default=config.SCREENSHOT_WIDTH, 
                        help="Width of screenshots to capture")
    parser.add_argument("--browser-height", type=int, default=config.BROWSER_HEIGHT,
                        help="Height of the browser window")
    parser.add_argument("--user-agent", type=str, default=config.USER_AGENT,
                       help="User agent string to use for the browser")
    
    # LLM settings  
    parser.add_argument("--model", type=str, help="Ollama model to use", default=config.MODEL_NAME)
    parser.add_argument("--stream", action="store_true", help="Stream LLM output in real-time")
    parser.add_argument("--temperature", type=float, default=config.LLM_TEMPERATURE,
                       help="Temperature setting for LLM generation")
    parser.add_argument("--timeout", type=float, default=config.LLM_TIMEOUT,
                      help="Timeout in seconds for LLM API calls")
    parser.add_argument("--ollama-url", type=str, default=config.OLLAMA_BASE_URL,
                      help="Base URL for Ollama API")
    
    # Element detection settings
    parser.add_argument("--no-element-detection", action="store_true", help="Disable automatic element detection")
    parser.add_argument("--max-elements", type=int, default=20, help="Maximum number of elements to detect")
    
    # Task execution settings
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS,
                      help="Maximum number of steps to take before giving up")
    parser.add_argument("--max-errors", type=int, default=config.MAX_ERRORS,
                       help="Maximum number of consecutive errors before giving up")
    
    # History settings
    parser.add_argument("--no-history", action="store_true", help="Disable saving task history")
    parser.add_argument("--history-dir", type=str, help="Directory to save history files", default="history")
    
    # Debug settings
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save-raw-screenshots", action="store_true", 
                       help="Save raw screenshots to disk for debugging purposes")
    
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    if args.debug:
        # Set root logger to DEBUG level
        logging.getLogger().setLevel(logging.DEBUG)
        # Set Navi loggers to DEBUG level
        for logger_name in ["navi", "navi.agent", "navi.browser", "navi.llm", "navi.utils"]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
        
        logger.debug("Debug logging enabled")
    
    # Run the task
    asyncio.run(run_task(args))

if __name__ == "__main__":
    main() 