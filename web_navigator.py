#!/usr/bin/env python3
"""
WebNavigator - An AI agent for autonomous web navigation using a local LLM.

This module implements a full solution for browser-based task automation
through screenshots and LLM decision making.
"""

import asyncio
import base64
import json
import logging
import os
import time
import argparse
import datetime
import pathlib
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import httpx
from PIL import Image
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, ElementHandle

# Import configuration
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebNavigator")

class WebNavigator:
    """Main WebNavigator agent class that coordinates browser control, screenshots, and LLM decisions."""
    
    def __init__(self, 
                 headless: bool = config.HEADLESS, 
                 screenshot_width: int = config.SCREENSHOT_WIDTH, 
                 browser_height: int = config.BROWSER_HEIGHT,
                 ollama_base_url: str = config.OLLAMA_BASE_URL,
                 model_name: str = config.MODEL_NAME,
                 max_steps: int = config.MAX_STEPS,
                 max_errors: int = config.MAX_ERRORS,
                 user_agent: str = config.USER_AGENT,
                 llm_temperature: float = config.LLM_TEMPERATURE,
                 llm_timeout: float = config.LLM_TIMEOUT,
                 save_history: bool = True,
                 history_dir: str = "history"):
        """
        Initialize the WebNavigator agent.
        
        Args:
            headless: Whether to run the browser in headless mode
            screenshot_width: Width of screenshots to capture
            browser_height: Height of the browser window
            ollama_base_url: Base URL for Ollama API
            model_name: Name of the LLM model to use
            max_steps: Maximum number of steps to take before giving up
            max_errors: Maximum number of consecutive errors before giving up
            user_agent: User agent string to use for the browser
            llm_temperature: Temperature setting for LLM generation
            llm_timeout: Timeout in seconds for LLM API calls
            save_history: Whether to save task execution history to a file
            history_dir: Directory to save history files
        """
        self.headless = headless
        self.screenshot_width = screenshot_width
        self.browser_height = browser_height
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.user_agent = user_agent
        self.llm_temperature = llm_temperature
        self.llm_timeout = llm_timeout
        self.save_history = save_history
        self.history_dir = history_dir
        
        # Create history directory if it doesn't exist
        if self.save_history:
            pathlib.Path(self.history_dir).mkdir(exist_ok=True, parents=True)
        
        # State tracking
        self.browser = None
        self.context = None
        self.page = None
        self.history = []
        self.task_complete = False
        self.error_count = 0
        self.thinking_step = 0
        self.task_id = None
        
    async def __aenter__(self):
        """Set up the browser environment when used as context manager."""
        await self.setup()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        await self.teardown()
    
    async def setup(self) -> None:
        """Initialize the browser and context."""
        logger.info("Setting up WebNavigator environment")
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": self.screenshot_width, "height": self.browser_height},
            user_agent=self.user_agent
        )
        self.page = await self.context.new_page()
        # Enable JS dialogs auto-dismissal
        self.page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
        logger.info("Browser environment ready")
    
    async def teardown(self) -> None:
        """Close the browser and clean up resources."""
        logger.info("Cleaning up WebNavigator environment")
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        logger.info("Browser environment closed")
    
    async def run_task(self, task_description: str, starting_url: str = config.DEFAULT_START_URL) -> Dict:
        """
        Execute a complete task described in natural language.
        
        Args:
            task_description: Natural language description of the task to perform
            starting_url: URL to start the navigation from
            
        Returns:
            Dict with task results including success status and execution trace
        """
        if not self.browser:
            await self.setup()
            
        # Create a unique task ID based on timestamp
        self.task_id = f"task_{int(time.time())}"
            
        self.history = []
        self.task_complete = False
        self.error_count = 0
        self.thinking_step = 0
        
        # Initialize task and navigate to starting point
        logger.info(f"Starting task: {task_description}")
        self.history.append({
            "step": 0,
            "type": "task_start",
            "task": task_description,
            "url": starting_url,
            "timestamp": time.time()
        })
        
        await self.page.goto(starting_url)
        await self.page.wait_for_load_state("networkidle")
        
        # Main agent loop
        current_step = 1
        
        while not self.task_complete and current_step <= self.max_steps:
            logger.info(f"Step {current_step}: Processing current page")
            
            try:
                # Take screenshot
                screenshot_base64 = await self.capture_screenshot()
                current_url = self.page.url
                
                # Get current page title and visible text
                title = await self.page.title()
                
                # Process with LLM to decide next action
                action = await self.get_llm_decision(
                    task_description, 
                    screenshot_base64, 
                    current_url,
                    title,
                    current_step
                )
                
                # Record the action in history
                history_entry = {
                    "step": current_step,
                    "type": "decision",
                    "url": current_url,
                    "title": title,
                    "action": action,
                    "timestamp": time.time()
                }
                self.history.append(history_entry)
                
                # Save a screenshot for the history if enabled
                if self.save_history:
                    await self.save_step_screenshot(current_step, screenshot_base64)
                
                # Check if task is complete
                if action.get("task_complete", False):
                    self.task_complete = True
                    logger.info("Task completed successfully")
                    self.history.append({
                        "step": current_step + 1,
                        "type": "task_complete",
                        "result": action.get("result", "Task completed successfully"),
                        "timestamp": time.time()
                    })
                    break
                
                # Execute the action
                success = await self.execute_action(action)
                if not success:
                    logger.warning(f"Failed to execute action: {action}")
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        logger.error("Too many consecutive errors, aborting task")
                        break
                else:
                    self.error_count = 0  # Reset consecutive error count on success
                
                # Wait for page to settle after action
                await asyncio.sleep(1)
                
                # Try to detect if a navigation occurred
                await self.wait_for_navigation_or_stability()
                
                current_step += 1
                
            except Exception as e:
                logger.error(f"Error during step {current_step}: {str(e)}", exc_info=True)
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    logger.error("Too many consecutive errors, aborting task")
                    break
        
        # Compile results
        result = {
            "success": self.task_complete,
            "steps_taken": current_step - 1,
            "final_url": self.page.url,
            "history": self.history,
            "error_count": self.error_count,
            "task_id": self.task_id
        }
        
        if current_step > self.max_steps and not self.task_complete:
            logger.warning("Reached maximum step limit without completing task")
            result["success"] = False
            result["reason"] = "Maximum step limit reached"
        
        # Save history to file if enabled
        if self.save_history:
            await self.save_task_history(result)
        
        return result
    
    async def save_step_screenshot(self, step: int, screenshot_base64: str) -> None:
        """Save a screenshot for a step to the history directory."""
        if not self.save_history or not self.task_id:
            return
            
        screenshots_dir = os.path.join(self.history_dir, self.task_id, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        screenshot_path = os.path.join(screenshots_dir, f"step_{step}.png")
        
        # Decode and save the screenshot
        try:
            screenshot_bytes = base64.b64decode(screenshot_base64)
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_bytes)
            logger.debug(f"Saved screenshot for step {step}")
        except Exception as e:
            logger.error(f"Failed to save screenshot for step {step}: {str(e)}")
    
    async def save_task_history(self, result: Dict) -> None:
        """Save the task execution history to a JSON file."""
        if not self.save_history or not self.task_id:
            return
            
        task_dir = os.path.join(self.history_dir, self.task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        history_path = os.path.join(task_dir, "history.json")
        
        # Include timestamp and formatted date in the history
        result["timestamp"] = time.time()
        result["date"] = datetime.datetime.now().isoformat()
        
        try:
            with open(history_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Task history saved to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save task history: {str(e)}")
    
    async def capture_screenshot(self) -> str:
        """
        Capture a screenshot of the current page and return it as a base64 string.
        
        Returns:
            Base64-encoded screenshot image
        """
        screenshot_bytes = await self.page.screenshot()
        # Resize if needed
        img = Image.open(BytesIO(screenshot_bytes))
        width, height = img.size
        
        if width > self.screenshot_width:
            # Resize maintaining aspect ratio
            new_height = int(height * (self.screenshot_width / width))
            img = img.resize((self.screenshot_width, new_height), Image.LANCZOS)
            
            # Convert back to bytes
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            screenshot_bytes = buffer.getvalue()
        
        return base64.b64encode(screenshot_bytes).decode('utf-8')
    
    async def get_visible_text(self) -> str:
        """Extract visible text from the current page."""
        return await self.page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, a, button, input, textarea, label, li'))
                    .map(element => element.textContent || element.value || '')
                    .filter(text => text.trim().length > 0)
                    .join('\\n')
                    .trim()
                    .substring(0, 5000); // Limit text length
            }
        """)
    
    async def get_llm_decision(self, 
                              task: str, 
                              screenshot_base64: str, 
                              current_url: str,
                              page_title: str,
                              current_step: int) -> Dict:
        """
        Ask the LLM to decide on the next action based on the current state.
        
        Args:
            task: The original task description
            screenshot_base64: Base64-encoded screenshot image
            current_url: URL of the current page
            page_title: Title of the current page
            current_step: Current step number
            
        Returns:
            Dict containing the action decision from the LLM
        """
        self.thinking_step += 1
        logger.info(f"Requesting LLM decision (thinking step {self.thinking_step})")
        
        # Prepare system prompt for LLM
        system_prompt = """You are WebNavigator, an AI agent that controls a web browser to complete tasks.
You are given a screenshot of the current browser view, the current URL, and page title.
Your job is to decide what action to take next to complete the user's task.

You can take these actions:
1. CLICK: Click on an element identified by either coordinates (x,y) or a CSS selector
2. TYPE: Type text into a field identified by coordinates or CSS selector
3. NAVIGATE: Go to a URL directly
4. SCROLL: Scroll the page (up/down)
5. COMPLETE: Mark the task as complete when you've successfully finished it

IMPORTANT GUIDELINES:
- Prefer coordinates (x,y) for clicking when the element is clearly visible in the screenshot
- Use CSS selectors when coordinates might be unreliable or the element needs to be precisely identified
- Be specific: describe exactly what you're doing and why
- Think step by step to break down complex tasks
- When typing into forms, be specific about what information you're entering
- Only mark a task as complete when you have definitive evidence the task is done

OUTPUT FORMAT:
You must respond in valid JSON format with these fields:
{
  "thinking": "Your step-by-step reasoning about what's on the screen and what to do next",
  "action_type": "CLICK|TYPE|NAVIGATE|SCROLL|COMPLETE",
  "action_params": {
    // Parameters depend on action_type:
    // For CLICK: "coordinates": [x, y] OR "selector": "css_selector"
    // For TYPE: "text": "text to type", AND "coordinates": [x, y] OR "selector": "css_selector"
    // For NAVIGATE: "url": "https://example.com"
    // For SCROLL: "direction": "up|down", "amount": 300
  },
  "task_complete": false,  // Set to true only when task is fully complete
  "result": "Optional explanation of results when task is complete"
}"""

        # Prepare some relevant history for context
        history_context = ""
        if len(self.history) > 0:
            recent_history = self.history[-min(3, len(self.history)):]
            history_entries = []
            for entry in recent_history:
                if entry.get("type") == "decision" and "action" in entry:
                    action = entry.get("action", {})
                    action_type = action.get("action_type", "UNKNOWN")
                    thinking = action.get("thinking", "")[:100] + "..." if len(action.get("thinking", "")) > 100 else action.get("thinking", "")
                    history_entries.append(f"Step {entry.get('step')}: {action_type} - {thinking}")
            
            if history_entries:
                history_context = "Recent actions:\n" + "\n".join(history_entries)

        # Prepare user prompt
        user_prompt = f"""Task: {task}

Current State:
- URL: {current_url}
- Page Title: {page_title}
- Step: {current_step}

{history_context}

The screenshot of the current page is provided as a base64-encoded image.
Analyze what you see and decide on the next action to complete the task.

Respond ONLY with a valid JSON object containing your thinking and the next action."""

        # Call Ollama API to get a response
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "images": [screenshot_base64],
                        "stream": False,
                        "options": {
                            "temperature": self.llm_temperature
                        }
                    },
                    timeout=self.llm_timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                    return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
                
                llm_response = response.json()
                response_text = llm_response.get("response", "")
                
                # Extract JSON from response - handle potential text wrapping
                try:
                    # Try to find JSON in the response
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}")
                    
                    if json_start >= 0 and json_end >= 0:
                        json_str = response_text[json_start:json_end+1]
                        action_decision = json.loads(json_str)
                    else:
                        # Fallback if no JSON found
                        logger.warning("Couldn't find valid JSON in LLM response")
                        action_decision = {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
                    
                    # Validate required fields
                    if "action_type" not in action_decision:
                        action_decision["action_type"] = "SCROLL"
                        action_decision["action_params"] = {"direction": "down", "amount": 300}
                    
                    if "action_params" not in action_decision:
                        action_decision["action_params"] = {}
                        
                    if "task_complete" not in action_decision:
                        action_decision["task_complete"] = False
                        
                    return action_decision
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
                    logger.debug(f"Raw response: {response_text}")
                    return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
                
            except Exception as e:
                logger.error(f"Error communicating with Ollama: {str(e)}", exc_info=True)
                # Return a safe default action
                return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
    
    async def execute_action(self, action: Dict) -> bool:
        """
        Execute the action decided by the LLM.
        
        Args:
            action: Dictionary containing action details from LLM
            
        Returns:
            Boolean indicating if the action was executed successfully
        """
        action_type = action.get("action_type", "").upper()
        params = action.get("action_params", {})
        
        logger.info(f"Executing action: {action_type} with params: {params}")
        
        try:
            if action_type == "CLICK":
                return await self.handle_click(params)
            elif action_type == "TYPE":
                return await self.handle_type(params)
            elif action_type == "NAVIGATE":
                return await self.handle_navigate(params)
            elif action_type == "SCROLL":
                return await self.handle_scroll(params)
            elif action_type == "COMPLETE":
                # Nothing to execute for COMPLETE action
                return True
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {str(e)}", exc_info=True)
            return False
    
    async def handle_click(self, params: Dict) -> bool:
        """Handle a click action."""
        if "coordinates" in params:
            try:
                x, y = params["coordinates"]
                await self.page.mouse.click(x, y)
                logger.info(f"Clicked at coordinates ({x}, {y})")
                return True
            except Exception as e:
                logger.error(f"Failed to click at coordinates: {str(e)}")
                return False
        elif "selector" in params:
            try:
                selector = params["selector"]
                # Wait for the element to be visible first
                try:
                    await self.page.wait_for_selector(selector, state="visible", timeout=5000)
                except:
                    # Element might not appear, but we'll still try to click it
                    pass
                
                await self.page.click(selector)
                logger.info(f"Clicked element with selector: {selector}")
                return True
            except Exception as e:
                logger.error(f"Failed to click selector {params['selector']}: {str(e)}")
                return False
        else:
            logger.warning("Click action missing both coordinates and selector")
            return False
    
    async def handle_type(self, params: Dict) -> bool:
        """Handle a typing action."""
        if "text" not in params:
            logger.warning("Type action missing text parameter")
            return False
            
        text = params["text"]
        
        if "selector" in params:
            try:
                selector = params["selector"]
                # Try to wait for the element to be visible first
                try:
                    await self.page.wait_for_selector(selector, state="visible", timeout=5000)
                except:
                    # Element might not appear, but we'll still try to type into it
                    pass
                
                # Clear the field first if possible
                try:
                    await self.page.fill(selector, "")
                except:
                    # If fill doesn't work, try clicking and pressing keyboard shortcuts
                    await self.page.click(selector)
                    await self.page.keyboard.press("Control+A")
                    await self.page.keyboard.press("Delete")
                
                await self.page.fill(selector, text)
                logger.info(f"Typed '{text}' into element with selector: {selector}")
                return True
            except Exception as e:
                logger.error(f"Failed to type into selector {params['selector']}: {str(e)}")
                return False
        elif "coordinates" in params:
            try:
                x, y = params["coordinates"]
                await self.page.mouse.click(x, y)
                
                # Try to select all text and delete it first
                await self.page.keyboard.press("Control+A")
                await self.page.keyboard.press("Delete")
                
                await self.page.keyboard.type(text)
                logger.info(f"Typed '{text}' at coordinates ({x}, {y})")
                return True
            except Exception as e:
                logger.error(f"Failed to type at coordinates: {str(e)}")
                return False
        else:
            logger.warning("Type action missing both coordinates and selector")
            return False
    
    async def handle_navigate(self, params: Dict) -> bool:
        """Handle a navigation action."""
        if "url" not in params:
            logger.warning("Navigate action missing URL parameter")
            return False
            
        url = params["url"]
        try:
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
                
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            logger.info(f"Navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {str(e)}")
            return False
    
    async def handle_scroll(self, params: Dict) -> bool:
        """Handle a scroll action."""
        direction = params.get("direction", "down")
        amount = params.get("amount", 300)
        
        try:
            if direction.lower() == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            else:  # Default to down
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            
            logger.info(f"Scrolled {direction} by {amount} pixels")
            return True
        except Exception as e:
            logger.error(f"Failed to scroll: {str(e)}")
            return False
    
    async def wait_for_navigation_or_stability(self) -> None:
        """Wait for either navigation to complete or page to become stable."""
        try:
            # First wait for any outstanding network requests to complete
            await self.page.wait_for_load_state("networkidle", timeout=3000)
        except:
            # Timeout is expected if the page is already stable
            pass
            
        # Additional stability check
        await asyncio.sleep(0.5)


async def main():
    """Run the WebNavigator agent with command line arguments."""
    parser = argparse.ArgumentParser(description="WebNavigator AI Agent")
    parser.add_argument("--task", type=str, help="Task description in natural language", 
                        default="Search for 'latest AI research papers' on Google and find a recent paper about LLMs.")
    parser.add_argument("--start-url", type=str, help="Starting URL for the task", 
                        default=config.DEFAULT_START_URL)
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no browser UI)")
    parser.add_argument("--model", type=str, help="Ollama model to use", default=config.MODEL_NAME)
    parser.add_argument("--no-history", action="store_true", help="Disable saving task history")
    parser.add_argument("--history-dir", type=str, help="Directory to save history files", default="history")
    args = parser.parse_args()
    
    # Override config settings with command line arguments if provided
    headless = args.headless if args.headless else config.HEADLESS
    model_name = args.model
    task_description = args.task
    starting_url = args.start_url
    save_history = not args.no_history
    history_dir = args.history_dir
    
    print(f"Starting WebNavigator with task: {task_description}")
    print(f"Model: {model_name}")
    print(f"Headless mode: {headless}")
    print(f"Starting URL: {starting_url}")
    print(f"Save history: {save_history}")
    
    async with WebNavigator(
        headless=headless, 
        model_name=model_name,
        save_history=save_history,
        history_dir=history_dir
    ) as navigator:
        result = await navigator.run_task(task_description, starting_url)
        
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


if __name__ == "__main__":
    asyncio.run(main())
