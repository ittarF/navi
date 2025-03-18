"""
Agent module for Navi, handling task management and LLM integration.

This module provides the main NaviAgent class that manages tasks, interacts with
the LLM decision maker, and coordinates browser actions.
"""

import asyncio
import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any

from navi.llm import LLMDecisionMaker
from navi.browser_controller import BrowserController
from navi.utils import save_screenshot, create_task_id, save_raw_screenshot

import config

# Configure logging
logger = logging.getLogger("navi.agent")

class NaviAgent:
    """
    Main Navi agent class that coordinates browser actions and LLM decisions.
    
    This class handles high-level task management, history tracking, and orchestrating
    the interaction between the browser controller and LLM decision maker.
    """
    
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
                 history_dir: str = "history",
                 stream_output: bool = False,
                 use_element_detection: bool = True,
                 max_elements: int = 20):
        """
        Initialize the Navi agent.
        
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
            stream_output: Whether to stream LLM responses
            use_element_detection: Whether to detect and provide page elements to the LLM
            max_elements: Maximum number of elements to extract and include
        """
        # Store configuration
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.save_history = save_history
        self.history_dir = history_dir
        self.stream_output = stream_output
        self.use_element_detection = use_element_detection
        
        # Initialize browser controller
        self.browser = BrowserController(
            headless=headless,
            screenshot_width=screenshot_width,
            browser_height=browser_height,
            user_agent=user_agent,
            max_elements=max_elements
        )
        
        # Initialize LLM decision maker
        self.llm = LLMDecisionMaker(
            model_name=model_name,
            base_url=ollama_base_url,
            temperature=llm_temperature,
            timeout=llm_timeout,
            stream=stream_output,
            on_stream_update=self._handle_stream_token if stream_output else None
        )
        
        # State tracking
        self.history = []
        self.task_complete = False
        self.error_count = 0
        self.task_id = None
        self.stream_output_buffer = []
    
    def _handle_stream_token(self, token: str):
        """Handle a token from the streaming LLM response."""
        self.stream_output_buffer.append(token)
        # Print without newline and flush the buffer
        print(token, end="", flush=True)
    
    async def setup(self) -> None:
        """Initialize the browser environment."""
        logger.info("Setting up Navi agent")
        await self.browser.setup()
    
    async def teardown(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up Navi agent")
        await self.browser.teardown()
    
    async def run_task(self, task_description: str, starting_url: str = config.DEFAULT_START_URL, 
                     save_raw_screenshots: bool = False) -> Dict:
        """
        Execute a complete task described in natural language.
        
        Args:
            task_description: Natural language description of the task to perform
            starting_url: URL to start the navigation from
            save_raw_screenshots: Whether to save raw screenshots for debugging
            
        Returns:
            Dict with task results including success status and execution trace
        """
        # Initialize browser if not already done
        if not self.browser.browser:
            await self.setup()
            
        # Create a unique task ID 
        self.task_id = create_task_id()
            
        # Reset state
        self.history = []
        self.task_complete = False
        self.error_count = 0
        
        # Initialize task and navigate to starting point
        logger.info(f"Starting task: {task_description}")
        self.history.append({
            "step": 0,
            "type": "task_start",
            "task": task_description,
            "url": starting_url,
            "timestamp": time.time()
        })
        
        # Navigate to the starting URL
        nav_success = await self.browser.navigate_to(starting_url)
        if not nav_success:
            logger.error(f"Failed to navigate to starting URL: {starting_url}")
            self.history.append({
                "step": 1,
                "type": "error",
                "error": "Failed to navigate to starting URL",
                "url": starting_url,
                "timestamp": time.time()
            })
            return {
                "success": False,
                "steps_taken": 0,
                "error": "Failed to navigate to starting URL",
                "history": self.history,
                "task_id": self.task_id
            }
        
        # Wait for the page to stabilize
        logger.info("Waiting for initial page load to stabilize")
        await self.browser.wait_for_navigation_or_stability()
        
        # Main agent loop
        current_step = 1
        
        while not self.task_complete and current_step <= self.max_steps:
            if self.stream_output:
                print(f"\n⏳ Step {current_step}: ", end="", flush=True)
            else:
                logger.info(f"Step {current_step}: Processing current page")
            
            try:
                # Save raw screenshot if debugging is enabled
                if save_raw_screenshots:
                    await save_raw_screenshot(self.task_id, current_step, self.browser.page, self.history_dir)
                
                # Get current page state
                logger.info("Capturing fresh page state (URL, title, screenshot)")
                page_state = await self.browser.get_page_state()
                current_url = page_state["url"]
                title = page_state["title"]
                screenshot_base64 = page_state["screenshot"]
                
                # Log the first 50 characters of the screenshot base64 data for debugging
                logger.debug(f"Screenshot base64 data starts with: {screenshot_base64[:50]}...")
                
                # Extract page elements if enabled
                page_elements_text = ""
                if self.use_element_detection:
                    if self.stream_output:
                        print("\n🔍 Detecting interactive elements... ", end="", flush=True)
                        
                    try:
                        elements, page_elements_text = await self.browser.extract_elements()
                        
                        if self.stream_output:
                            detected_count = len(elements)
                            print(f"Found {detected_count} elements.", flush=True)
                    except Exception as e:
                        logger.error(f"Error extracting page elements: {str(e)}")
                        if self.stream_output:
                            print(f"Error: {str(e)}", flush=True)
                
                # Get any error context that needs to be sent to the LLM
                error_context = self.browser.get_error_context_for_llm()
                
                # Reset streaming buffer
                self.stream_output_buffer = []
                
                # Process with LLM to decide next action
                logger.info("Sending current page state to LLM for decision")
                action = await self.llm.get_next_action(
                    task_description, 
                    screenshot_base64, 
                    current_url,
                    title,
                    current_step,
                    self.history,
                    page_elements_text,
                    error_context
                )
                
                # Add a newline after streaming output
                if self.stream_output:
                    print()
                
                # Record the action in history
                history_entry = {
                    "step": current_step,
                    "type": "decision",
                    "url": current_url,
                    "title": title,
                    "action": action,
                    "timestamp": time.time()
                }
                
                # Also record detected elements and errors in history if available
                if page_elements_text and page_elements_text != "No interactive elements detected on the page.":
                    history_entry["elements_detected"] = True
                
                if error_context:
                    history_entry["had_errors"] = True
                    history_entry["error_context"] = error_context
                
                self.history.append(history_entry)
                
                # Save a screenshot for the history if enabled
                if self.save_history:
                    await save_screenshot(
                        self.task_id, 
                        current_step, 
                        screenshot_base64, 
                        self.history_dir
                    )
                
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
                logger.info(f"Executing action: {action.get('action_type', 'UNKNOWN')}")
                success = await self.browser.execute_action(action)
                if not success:
                    logger.warning(f"Failed to execute action: {action}")
                    self.error_count += 1
                    
                    # Record the error in history
                    self.history.append({
                        "step": current_step,
                        "type": "action_error",
                        "action": action,
                        "error": self.browser.last_error or "Unknown error",
                        "timestamp": time.time()
                    })
                    
                    if self.error_count >= self.max_errors:
                        logger.error("Too many consecutive errors, aborting task")
                        break
                else:
                    self.error_count = 0  # Reset consecutive error count on success
                
                # Wait for page to settle after action
                logger.info("Waiting for page to stabilize after action")
                await asyncio.sleep(1.5)  # Increased from 1.0 to give more time for content to load
                
                # Try to detect if a navigation occurred
                await self.browser.wait_for_navigation_or_stability()
                
                # Force a new screenshot for visual verification of changes (if debug is enabled)
                if logger.isEnabledFor(logging.DEBUG) or save_raw_screenshots:
                    logger.debug("Capturing debug verification screenshot")
                    debug_page_state = await self.browser.get_page_state()
                    logger.debug(f"Debug screenshot base64 data starts with: {debug_page_state['screenshot'][:50]}...")
                    
                    if save_raw_screenshots:
                        await save_raw_screenshot(self.task_id, f"{current_step}_after", self.browser.page, self.history_dir)
                
                current_step += 1
                
            except Exception as e:
                logger.error(f"Error during step {current_step}: {str(e)}", exc_info=True)
                self.error_count += 1
                
                # Record the error in history
                self.history.append({
                    "step": current_step,
                    "type": "step_error",
                    "error": str(e),
                    "timestamp": time.time()
                })
                
                if self.error_count >= self.max_errors:
                    logger.error("Too many consecutive errors, aborting task")
                    break
        
        # Compile results
        result = {
            "success": self.task_complete,
            "steps_taken": current_step - 1,
            "final_url": self.browser.page.url,
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
    
    async def describe_current_page(self) -> str:
        """
        Get a description of the current page from the LLM.
        
        Returns:
            String description of the current page
        """
        # Get current page state
        page_state = await self.browser.get_page_state()
        screenshot_base64 = page_state["screenshot"]
        url = page_state["url"]
        title = page_state["title"]
        
        # Extract page elements if enabled
        page_elements_text = ""
        if self.use_element_detection:
            try:
                _, page_elements_text = await self.browser.extract_elements()
            except Exception as e:
                logger.error(f"Error extracting page elements: {str(e)}")
        
        # Use streaming if enabled
        if self.stream_output:
            print("📝 Generating page description: ", end="", flush=True)
            self.stream_output_buffer = []
            
            description = await self.llm.describe_page(
                screenshot_base64, 
                url, 
                title,
                page_elements_text,
                stream=True,
                on_token=self._handle_stream_token
            )
            print()  # Add a newline after streaming
            return description
        else:
            return await self.llm.describe_page(
                screenshot_base64, 
                url, 
                title,
                page_elements_text
            ) 