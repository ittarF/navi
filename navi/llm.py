"""
LLM module for Navi, handling interactions with Ollama-based models.

This module provides functionality for querying LLMs via Ollama's API,
primarily to determine the next action to take during web navigation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator

import httpx

import config

# Configure logging
logger = logging.getLogger("navi.llm")

class LLMDecisionMaker:
    """Class to handle decision making via LLM."""
    
    def __init__(self, 
                 model_name: str = config.MODEL_NAME,
                 base_url: str = config.OLLAMA_BASE_URL,
                 temperature: float = config.LLM_TEMPERATURE,
                 timeout: float = config.LLM_TIMEOUT,
                 stream: bool = False,
                 on_stream_update: Optional[Callable[[str], None]] = None):
        """
        Initialize the LLM decision maker.
        
        Args:
            model_name: Name of the LLM model to use
            base_url: Base URL for Ollama API
            temperature: Temperature setting for LLM generation
            timeout: Timeout in seconds for LLM API calls
            stream: Whether to stream responses from the LLM
            on_stream_update: Callback function to handle streaming updates
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.stream = stream
        self.on_stream_update = on_stream_update
        
    async def get_next_action(self, 
                              task: str, 
                              screenshot_base64: str, 
                              current_url: str,
                              page_title: str,
                              current_step: int,
                              history: List[Dict] = None,
                              page_elements: str = "",
                              error_context: str = "") -> Dict:
        """
        Get the next action decision from the LLM.
        
        Args:
            task: Original task description
            screenshot_base64: Base64-encoded screenshot image
            current_url: Current URL being viewed
            page_title: Title of the current page
            current_step: Current step number in the task
            history: Previous actions taken in the task
            page_elements: Formatted string of interactive page elements
            error_context: Any error information to provide to the LLM
            
        Returns:
            Dictionary containing the action decision
        """
        logger.info(f"Requesting LLM decision for step {current_step}")
        
        # Log if we have error context
        if error_context:
            logger.info(f"Including error context in LLM request: {error_context[:100]}...")
        
        # Prepare system prompt
        system_prompt = self._get_system_prompt()
        
        # Prepare history context
        history_context = self._get_history_context(history) if history else ""
        
        # Prepare user prompt
        user_prompt = self._get_user_prompt(task, current_url, page_title, current_step, history_context, page_elements, error_context)
        
        # Call Ollama API to get a response
        try:
            if self.stream and self.on_stream_update:
                action_decision = await self._call_ollama_api_streaming(system_prompt, user_prompt, screenshot_base64)
            else:
                action_decision = await self._call_ollama_api(system_prompt, user_prompt, screenshot_base64)
            return action_decision
        except Exception as e:
            logger.error(f"Error getting LLM decision: {str(e)}", exc_info=True)
            # Return a safe default action if something goes wrong
            return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are Navi, an AI agent that controls a web browser to complete tasks.
You are given a screenshot of the current browser view, the current URL, and page title.
Your job is to decide what action to take next to complete the user's task.

You can take these actions:
1. CLICK: Click on an element identified by either coordinates (x,y), a CSS selector, or an element number from the list of detected elements
2. TYPE: Type text into a field identified by coordinates, CSS selector, or element number
3. NAVIGATE: Go to a URL directly
4. SCROLL: Scroll the page (up/down)
5. COMPLETE: Mark the task as complete when you've successfully finished it

IMPORTANT GUIDELINES:
- PRIORITIZE HANDLING COOKIE BANNERS FIRST! If a cookie banner or privacy notice is detected, you should accept it before doing anything else
- When interactive elements are detected on the page, prefer using ELEMENT NUMBERS from the list rather than coordinates
- Only fall back to coordinates (x,y) when the element is not in the detected list or selectors might be unreliable
- Use CSS selectors as a last resort when element detection fails and coordinates might be unreliable
- Be concise in your reasoning - keep explanations brief and to the point
- Avoid verbose descriptions and focus only on key observations and decisions
- When typing into forms, be specific about what information you're entering
- If errors occurred during element detection, consider that some elements might not be listed
- Only mark a task as complete when you have definitive evidence the task is done

ERROR HANDLING:
- When provided with error information, analyze it carefully to adjust your next action
- If an action has failed, try an alternative approach or provide feedback on what went wrong
- For element-related errors, consider using a different targeting method (e.g. switch from selectors to coordinates)
- If navigation or page loading errors occur, consider retrying or using a different URL
- If actions consistently fail on a page, consider whether the page might be blocking automated interactions
- Handle timeouts by simplifying your actions or breaking them into smaller steps

COOKIE BANNERS HANDLING:
- Cookie banners/consent notices are very common on websites and often block interaction
- They are typically displayed at the top or bottom of the page, or as modal overlays
- When a cookie banner is detected, your first action should be to click the "Accept" or "Accept All" button
- Cookie banners may be labeled with "🍪 COOKIE BANNER" in the element list to make them easier to identify
- If the banner has multiple options, prefer the option that accepts all cookies to continue browsing with full functionality

OUTPUT FORMAT:
You must respond in valid JSON format with these fields:
{
  "thinking": "Your very brief reasoning about what to do next (keep this under 100 words!)",
  "action_type": "CLICK|TYPE|NAVIGATE|SCROLL|COMPLETE",
  "action_params": {
    // Parameters depend on action_type:
    // For CLICK: "element_number": 1 OR "coordinates": [x, y] OR "selector": "css_selector"
    // For TYPE: "text": "text to type", AND "element_number": 1 OR "coordinates": [x, y] OR "selector": "css_selector"
    // For NAVIGATE: "url": "https://example.com"
    // For SCROLL: "direction": "up|down", "amount": 300
  },
  "task_complete": false,  // Set to true only when task is fully complete
  "result": "Optional brief explanation of results when task is complete"
}"""
    
    def _get_user_prompt(self, task: str, current_url: str, page_title: str, current_step: int, 
                        history_context: str, page_elements: str, error_context: str = "") -> str:
        """Get the user prompt for the LLM."""
        prompt = f"""Task: {task}

Current State:
- URL: {current_url}
- Page Title: {page_title}
- Step: {current_step}

{history_context}
"""
        
        # Add error context if available
        if error_context:
            prompt += f"""
Important Error Information:
{error_context}

These errors may affect how you should respond. Consider alternative approaches if your previous actions have failed.
"""
        
        # Add page elements if available
        if page_elements:
            prompt += f"\nInteractive Elements:\n{page_elements}\n"

        prompt += """
The screenshot of the current page is provided as a base64-encoded image.
Be extremely concise in your response and decision-making.
Focus only on the immediate next action to take.

Remember: If you see a cookie banner or consent notice, handle it first by accepting it.

Respond ONLY with a valid JSON object containing your thinking and the next action."""
        
        return prompt
    
    def _get_history_context(self, history: List[Dict]) -> str:
        """Format recent history entries for context."""
        if not history or len(history) == 0:
            return ""
            
        # Get the last 3 decision steps and any error steps
        recent_steps = []
        error_steps = []
        
        for entry in reversed(history):
            if entry.get("type") == "decision" and "action" in entry:
                if len(recent_steps) < 3:
                    recent_steps.append(entry)
            elif entry.get("type") in ["action_error", "step_error"]:
                if len(error_steps) < 2:  # Limit to last 2 errors
                    error_steps.append(entry)
        
        recent_steps.reverse()  # Put back in chronological order
        error_steps.reverse()
        
        history_entries = []
        
        # Add the recent decisions
        if recent_steps:
            history_entries.append("Recent actions:")
            for entry in recent_steps:
                action = entry.get("action", {})
                action_type = action.get("action_type", "UNKNOWN")
                thinking = action.get("thinking", "")[:50] + "..." if len(action.get("thinking", "")) > 50 else action.get("thinking", "")
                history_entries.append(f"Step {entry.get('step')}: {action_type} - {thinking}")
        
        # Add the recent errors
        if error_steps:
            history_entries.append("\nRecent errors:")
            for entry in error_steps:
                error_msg = entry.get("error", "Unknown error")
                history_entries.append(f"Step {entry.get('step')}: {error_msg}")
        
        return "\n".join(history_entries)
        
    async def _call_ollama_api(self, system_prompt: str, user_prompt: str, screenshot_base64: str) -> Dict:
        """Call Ollama API to get a response."""
        async with httpx.AsyncClient() as client:
            try:
                # Add more specific instructions about using element IDs from the element list
                enhanced_system_prompt = system_prompt + """

IMPORTANT ADDITIONAL INSTRUCTIONS:
- When selecting elements to interact with, ONLY use element IDs or numbers that are actually present in the provided element list
- Focus carefully on the image to identify the correct elements
- Your response MUST be in the exact JSON format specified above with the correct keys
- NEVER make up element IDs or selectors that aren't visible in the elements list
- Use ONLY the following keys: "thinking", "action_type", "action_params", "task_complete", and optionally "result"
- For action_params, use ONLY "element_number", "coordinates", "selector", "text", "url", "direction", or "amount" as appropriate
"""

                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": enhanced_system_prompt,
                        "images": [screenshot_base64],
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": 1024,  # Increased to ensure complete JSON response
                            "num_ctx": 4096  # Increase context window for image processing
                        }
                    },
                    timeout=self.timeout * 1.5  # Increase timeout for image processing
                )
                
                if response.status_code != 200:
                    logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                    return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
                
                llm_response = response.json()
                response_text = llm_response.get("response", "")
                
                # Log the raw response for debugging
                logger.debug(f"Raw LLM response: {response_text[:500]}...")
                
                # Extract and validate JSON from response
                return self._extract_json_from_response(response_text)
                
            except httpx.TimeoutException:
                logger.error(f"Timeout while communicating with Ollama - timeout value: {self.timeout} seconds")
                return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
            except Exception as e:
                logger.error(f"Error communicating with Ollama: {str(e)}", exc_info=True)
                return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
    
    async def _call_ollama_api_streaming(self, system_prompt: str, user_prompt: str, screenshot_base64: str) -> Dict:
        """Call Ollama API with streaming enabled."""
        full_response = ""
        
        # Add more specific instructions about using element IDs from the element list
        enhanced_system_prompt = system_prompt + """

IMPORTANT ADDITIONAL INSTRUCTIONS:
- When selecting elements to interact with, ONLY use element IDs or numbers that are actually present in the provided element list
- Focus carefully on the image to identify the correct elements
- Your response MUST be in the exact JSON format specified above with the correct keys
- NEVER make up element IDs or selectors that aren't visible in the elements list
- Use ONLY the following keys: "thinking", "action_type", "action_params", "task_complete", and optionally "result"
- For action_params, use ONLY "element_number", "coordinates", "selector", "text", "url", "direction", or "amount" as appropriate
"""
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": enhanced_system_prompt,
                        "images": [screenshot_base64],
                        "stream": True,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": 1024,  # Increased to ensure complete JSON response
                            "num_ctx": 4096  # Increase context window for image processing
                        }
                    },
                    timeout=self.timeout * 1.5  # Increase timeout for image processing
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Error from Ollama API: {response.status_code}")
                        return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
                    
                    # Process streaming response
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                            
                        try:
                            # Parse each chunk as a JSON object
                            chunk_str = chunk.decode('utf-8')
                            for line in chunk_str.strip().split('\n'):
                                if not line:
                                    continue
                                    
                                chunk_data = json.loads(line)
                                if "response" in chunk_data:
                                    token = chunk_data["response"]
                                    full_response += token
                                    if self.on_stream_update:
                                        self.on_stream_update(token)
                        except Exception as e:
                            logger.error(f"Error parsing streaming chunk: {str(e)}")
                
                # Log the raw response for debugging
                logger.debug(f"Raw streaming LLM response: {full_response[:500]}...")
                
                # Extract and validate JSON from the full response
                return self._extract_json_from_response(full_response)
                
            except httpx.TimeoutException:
                logger.error(f"Streaming timeout while communicating with Ollama - timeout value: {self.timeout} seconds")
                return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
            except Exception as e:
                logger.error(f"Error with streaming request: {str(e)}", exc_info=True)
                return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
    
    async def stream_text_response(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """Stream a text response from the LLM without an image."""
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "stream": True,
                        "options": {
                            "temperature": self.temperature
                        }
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status_code != 200:
                        yield f"Error from Ollama API: {response.status_code}"
                        return
                    
                    # Process streaming response
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                            
                        try:
                            # Parse each chunk as a JSON object
                            chunk_str = chunk.decode('utf-8')
                            for line in chunk_str.strip().split('\n'):
                                if not line:
                                    continue
                                    
                                chunk_data = json.loads(line)
                                if "response" in chunk_data:
                                    token = chunk_data["response"]
                                    yield token
                        except Exception as e:
                            yield f"Error parsing streaming chunk: {str(e)}"
                            logger.error(f"Error parsing streaming chunk: {str(e)}")
                            
            except Exception as e:
                yield f"Error with streaming request: {str(e)}"
                logger.error(f"Error with streaming request: {str(e)}", exc_info=True)
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract and validate JSON from LLM response text."""
        try:
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = response_text[json_start:json_end+1]
                
                # Log the extracted JSON string for debugging
                logger.debug(f"Extracted JSON string: {json_str}")
                
                try:
                    action_decision = json.loads(json_str)
                except json.JSONDecodeError as json_err:
                    # Try to fix common JSON format issues
                    logger.warning(f"JSON decode error: {json_err}. Attempting to fix JSON format...")
                    fixed_json_str = self._fix_json_format(json_str)
                    action_decision = json.loads(fixed_json_str)
            else:
                # Fallback if no JSON found
                logger.warning("Couldn't find valid JSON in LLM response")
                action_decision = {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
            
            # Fix JSON structure if it has incorrect keys
            if "action" in action_decision and "action_type" not in action_decision:
                action_decision["action_type"] = action_decision.pop("action")
                logger.info("Fixed JSON: renamed 'action' to 'action_type'")
            
            if "element" in action_decision and "action_params" not in action_decision:
                action_decision["action_params"] = action_decision.pop("element")
                logger.info("Fixed JSON: renamed 'element' to 'action_params'")
            
            # Validate required fields
            if "action_type" not in action_decision:
                action_decision["action_type"] = "SCROLL"
                action_decision["action_params"] = {"direction": "down", "amount": 300}
            
            if "action_params" not in action_decision:
                action_decision["action_params"] = {}
                
            if "task_complete" not in action_decision:
                action_decision["task_complete"] = False
            
            # Ensure action_type is in the expected format
            if isinstance(action_decision["action_type"], str):
                action_decision["action_type"] = action_decision["action_type"].upper()
                
            # Convert action_type aliases to standard names
            if action_decision["action_type"] == "CLICK" or action_decision["action_type"] == "PRESS":
                action_decision["action_type"] = "CLICK"
            elif action_decision["action_type"] == "TYPE" or action_decision["action_type"] == "INPUT" or action_decision["action_type"] == "ENTER":
                action_decision["action_type"] = "TYPE"
            elif action_decision["action_type"] == "GOTO" or action_decision["action_type"] == "GO":
                action_decision["action_type"] = "NAVIGATE"
            elif action_decision["action_type"] == "DONE" or action_decision["action_type"] == "FINISH":
                action_decision["action_type"] = "COMPLETE"
                
            # Validate action_params format
            if "selector" in action_decision["action_params"]:
                # Ensure selector is a string
                action_decision["action_params"]["selector"] = str(action_decision["action_params"]["selector"])
            
            if "element_number" in action_decision["action_params"]:
                # Ensure element_number is an integer
                try:
                    action_decision["action_params"]["element_number"] = int(action_decision["action_params"]["element_number"])
                except (ValueError, TypeError):
                    # If it can't be converted to int, remove it
                    logger.warning(f"Invalid element_number: {action_decision['action_params']['element_number']}")
                    del action_decision["action_params"]["element_number"]
                
            return action_decision
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
        except Exception as e:
            logger.error(f"Unexpected error processing LLM response: {e}", exc_info=True)
            return {"action_type": "SCROLL", "action_params": {"direction": "down", "amount": 300}}
    
    def _fix_json_format(self, json_str: str) -> str:
        """Attempt to fix common JSON format issues."""
        # Replace single quotes with double quotes
        fixed_json = json_str.replace("'", '"')
        
        # Fix missing quotes around keys
        import re
        fixed_json = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', fixed_json)
        
        # Fix any trailing commas before closing brackets
        fixed_json = re.sub(r',\s*}', r'}', fixed_json)
        
        # Handle unquoted values that should be strings
        # This is more complex and might miss some cases
        fixed_json = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', fixed_json)
        
        logger.debug(f"Fixed JSON: {fixed_json}")
        return fixed_json
        
    async def describe_page(self, screenshot_base64: str, url: str, title: str, 
                          page_elements: str = "",
                          stream: bool = False, on_token: Optional[Callable[[str], None]] = None) -> str:
        """
        Request a description of a webpage from the LLM.
        
        Args:
            screenshot_base64: Base64-encoded screenshot
            url: URL of the webpage
            title: Title of the webpage
            page_elements: Formatted string of interactive page elements
            stream: Whether to stream the response
            on_token: Callback function for streaming tokens
            
        Returns:
            String description of the page contents
        """
        # Prepare system prompt
        system_prompt = """You are a helpful assistant that describes webpage content.
You are given a screenshot of a webpage, the URL, and the page title.
Your job is to:
1. Describe the overall layout and content of the page
2. Identify the main purpose of the page
3. List any key elements like navigation menus, search bars, articles, etc.
4. Mention any prominent images, videos, or interactive elements
5. Specifically note any cookie banners or consent notices that need to be addressed
6. Summarize the main textual content

Be concise and to the point. Avoid unnecessary words and verbose descriptions.
Focus on the most important information a user would want to know about the page."""

        # Prepare user prompt
        user_prompt = f"""URL: {url}
Page Title: {title}
"""

        # Add page elements if available
        if page_elements:
            user_prompt += f"\nInteractive Elements:\n{page_elements}\n"

        user_prompt += """
The screenshot of the webpage is provided as a base64-encoded image.
Provide a brief, concise description of this webpage, focusing only on the most important elements.
If you see a cookie banner or consent notice, make sure to mention it prominently."""

        # Handle streaming if enabled
        if stream and on_token:
            full_response = ""
            
            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": user_prompt,
                            "system": system_prompt,
                            "images": [screenshot_base64],
                            "stream": True,
                            "options": {
                                "temperature": 0.1  # Lower temperature for descriptions
                            }
                        },
                        timeout=self.timeout
                    ) as response:
                        if response.status_code != 200:
                            return f"Error from Ollama API: {response.status_code}"
                        
                        # Process streaming response
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                                
                            try:
                                chunk_str = chunk.decode('utf-8')
                                for line in chunk_str.strip().split('\n'):
                                    if not line:
                                        continue
                                        
                                    chunk_data = json.loads(line)
                                    if "response" in chunk_data:
                                        token = chunk_data["response"]
                                        full_response += token
                                        on_token(token)
                            except Exception as e:
                                logger.error(f"Error parsing streaming chunk: {str(e)}")
                    
                    return full_response
                    
                except Exception as e:
                    return f"Error with streaming request: {str(e)}"
        
        # Non-streaming request
        else:
            # Call Ollama API
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": user_prompt,
                            "system": system_prompt,
                            "images": [screenshot_base64],
                            "stream": False,
                            "options": {
                                "temperature": 0.1  # Lower temperature for descriptions
                            }
                        },
                        timeout=self.timeout
                    )
                    
                    if response.status_code != 200:
                        return f"Error from Ollama API: {response.status_code} {response.text}"
                    
                    result = response.json()
                    return result.get("response", "No response received")
                    
                except Exception as e:
                    return f"Error communicating with Ollama: {str(e)}" 