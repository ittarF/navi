"""
Browser controller module for Navi.

This module handles browser initialization, control, and action execution.
It provides lower-level functions to interact with the browser.
"""

import asyncio
import base64
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Union

from PIL import Image
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, ElementHandle

from navi.utils import extract_page_elements, format_page_elements_for_llm

# Configure logging
logger = logging.getLogger("navi.browser")

class BrowserController:
    """Class that handles browser control and action execution."""
    
    def __init__(self, 
                 headless: bool = True, 
                 screenshot_width: int = 1280, 
                 browser_height: int = 800,
                 user_agent: str = None,
                 max_elements: int = 20):
        """
        Initialize the browser controller.
        
        Args:
            headless: Whether to run the browser in headless mode
            screenshot_width: Width of screenshots to capture
            browser_height: Height of the browser window
            user_agent: User agent string to use for the browser
            max_elements: Maximum number of elements to extract and include
        """
        self.headless = headless
        self.screenshot_width = screenshot_width
        self.browser_height = browser_height
        self.user_agent = user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        self.max_elements = max_elements
        
        # Initialize browser-related attributes
        self.browser = None
        self.context = None
        self.page = None
        
        # Error tracking
        self.last_error = None
        self.error_context = {}
    
    async def setup(self) -> None:
        """Initialize the browser and context."""
        logger.info("Setting up browser environment")
        try:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context(
                viewport={"width": self.screenshot_width, "height": self.browser_height},
                user_agent=self.user_agent
            )
            self.page = await self.context.new_page()
            
            # Enable JS dialogs auto-dismissal
            self.page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
            
            # Clear any previous errors
            self.last_error = None
            self.error_context = {}
            
            logger.info("Browser environment ready")
        except Exception as e:
            self.last_error = f"Browser setup error: {str(e)}"
            self.error_context["setup_error"] = str(e)
            logger.error(f"Failed to set up browser: {str(e)}", exc_info=True)
            raise
    
    async def teardown(self) -> None:
        """Close the browser and clean up resources."""
        logger.info("Cleaning up browser environment")
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            logger.info("Browser environment closed")
        except Exception as e:
            self.last_error = f"Browser teardown error: {str(e)}"
            logger.error(f"Error during browser teardown: {str(e)}", exc_info=True)
    
    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            Success status
        """
        try:
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
                
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            logger.info(f"Navigated to: {url}")
            return True
        except Exception as e:
            self.last_error = f"Navigation error: {str(e)}"
            self.error_context["navigation_error"] = {
                "url": url,
                "error": str(e)
            }
            logger.error(f"Failed to navigate to {url}: {str(e)}")
            return False
    
    async def capture_screenshot(self) -> str:
        """
        Capture a screenshot of the current page and return it as a base64 string.
        
        Returns:
            Base64-encoded screenshot image
        """
        try:
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
        except Exception as e:
            self.last_error = f"Screenshot error: {str(e)}"
            self.error_context["screenshot_error"] = str(e)
            logger.error(f"Failed to capture screenshot: {str(e)}")
            # Return a base64 encoded transparent 1x1 pixel as fallback
            return "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    
    async def force_refresh_screenshot(self) -> str:
        """
        Force a fresh screenshot by waiting for viewport stability and ensuring the 
        page is fully rendered before capturing.
        
        Returns:
            Base64-encoded screenshot image
        """
        try:
            # Try to force a visual update by scrolling just a tiny bit and back
            await self.page.evaluate("window.scrollBy(0, 1)")
            await asyncio.sleep(0.1)
            await self.page.evaluate("window.scrollBy(0, -1)")
            
            # Make sure all network activity has settled
            try:
                await self.page.wait_for_load_state("networkidle", timeout=1000)
            except Exception as e:
                logger.debug(f"Timeout waiting for networkidle during refresh: {str(e)}")
                
            # Additional delay to ensure animations are complete
            await asyncio.sleep(0.5)
            
            # Now capture the screenshot
            return await self.capture_screenshot()
        except Exception as e:
            self.last_error = f"Screenshot refresh error: {str(e)}"
            self.error_context["screenshot_refresh_error"] = str(e)
            logger.error(f"Failed to refresh screenshot: {str(e)}")
            return await self.capture_screenshot()  # Fall back to regular screenshot
    
    async def get_page_state(self) -> Dict[str, Any]:
        """
        Get the current state of the page including URL, title, and screenshot.
        
        Returns:
            Dictionary with the page state
        """
        try:
            # Try to get a fresh screenshot
            screenshot = await self.force_refresh_screenshot()
            
            state = {
                "url": self.page.url,
                "title": await self.page.title(),
                "screenshot": screenshot,
                "elements": None,
                "elements_text": "",
                "error": self.last_error,
                "error_context": self.error_context
            }
            
            # Clear error state after reporting it
            self.last_error = None
            self.error_context = {}
            
            return state
        except Exception as e:
            logger.error(f"Error getting page state: {str(e)}", exc_info=True)
            # Return a minimal state with error info
            return {
                "url": self.page.url if self.page else "unknown",
                "title": "Error getting page state",
                "screenshot": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",  # 1x1 transparent pixel
                "elements": None,
                "elements_text": "",
                "error": f"Page state error: {str(e)}",
                "error_context": {"page_state_error": str(e)}
            }
    
    async def extract_elements(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extract interactive elements from the current page.
        
        Returns:
            Tuple of (element list, formatted element text)
        """
        try:
            elements = await extract_page_elements(self.page)
            elements_text = format_page_elements_for_llm(elements, self.max_elements)
            return elements, elements_text
        except Exception as e:
            self.last_error = f"Element extraction error: {str(e)}"
            self.error_context["element_extraction_error"] = str(e)
            logger.error(f"Error extracting page elements: {str(e)}")
            return [], "Error detecting interactive elements on the page."
    
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
                self.last_error = f"Unknown action type: {action_type}"
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            self.last_error = f"Action execution error ({action_type}): {str(e)}"
            self.error_context["action_error"] = {
                "action_type": action_type,
                "params": params,
                "error": str(e)
            }
            logger.error(f"Error executing action {action_type}: {str(e)}", exc_info=True)
            return False
    
    async def handle_click(self, params: Dict) -> bool:
        """Handle a click action."""
        # Support for element numbers from the detected elements list
        if "element_number" in params:
            try:
                element_number = int(params["element_number"])
                
                # Re-extract elements to get the current state
                elements, _ = await self.extract_elements()
                
                # Sort elements using the same priority function as in format_page_elements_for_llm
                def element_priority(element):
                    score = 0
                    
                    # Prioritize by type
                    type_priority = {
                        "button": 5,
                        "input": 4,
                        "link": 3,
                        "dropdown": 3,
                        "checkbox": 2,
                        "interactive": 1
                    }
                    element_type = element.get("type", "unknown")
                    score += type_priority.get(element_type, 0)
                    
                    # Prioritize by vertical position (higher = more important)
                    y_pos = element.get("position", {}).get("y", 1000)
                    score += max(0, (1000 - y_pos) / 1000)
                    
                    # Prioritize elements with text
                    if element.get("text"):
                        score += 2
                        
                    # Prioritize elements with certain attributes
                    attributes = element.get("attributes", {})
                    if "id" in attributes:
                        score += 1
                    if "name" in attributes:
                        score += 1
                    if any(attr for attr in attributes if "search" in attr.lower()):
                        score += 3
                    if any(attr for attr in attributes if "login" in attr.lower() or "submit" in attr.lower()):
                        score += 3
                        
                    # Prioritize by size (within reason)
                    width = element.get("position", {}).get("width", 0)
                    height = element.get("position", {}).get("height", 0)
                    size = width * height
                    if 100 < size < 40000:  # Avoid tiny or huge elements
                        score += min(size / 10000, 2)
                        
                    return score
                
                # First prioritize cookie banners if that's what we're trying to click
                cookie_banners = [el for el in elements if el.get("is_cookie_banner", False)]
                regular_elements = [el for el in elements if not el.get("is_cookie_banner", False)]
                sorted_regular = sorted(regular_elements, key=element_priority, reverse=True)
                
                # Combine in order: cookie banners first, then regular sorted elements
                sorted_elements = cookie_banners + sorted_regular
                
                # Check if element number is valid
                if 1 <= element_number <= len(sorted_elements):
                    target_element = sorted_elements[element_number - 1]
                    
                    # Try to use selector if available
                    if "selector" in target_element:
                        try:
                            await self.page.click(target_element["selector"])
                            logger.info(f"Clicked element {element_number} using selector: {target_element['selector']}")
                            return True
                        except Exception as click_error:
                            logger.warning(f"Selector click failed, falling back to coordinates: {str(click_error)}")
                            # Fallback to coordinates
                    
                    # Use coordinates as fallback
                    x = target_element["position"]["x"]
                    y = target_element["position"]["y"]
                    await self.page.mouse.click(x, y)
                    logger.info(f"Clicked element {element_number} at coordinates ({x}, {y})")
                    
                    # Small delay after clicking to let any interactions settle
                    await asyncio.sleep(0.5)
                    return True
                else:
                    self.last_error = f"Invalid element number: {element_number}"
                    self.error_context["element_number_error"] = {
                        "requested": element_number,
                        "available": len(sorted_elements)
                    }
                    logger.warning(f"Invalid element number: {element_number}")
                    return False
            except Exception as e:
                self.last_error = f"Element click error: {str(e)}"
                self.error_context["element_click_error"] = {
                    "element_number": params.get('element_number'),
                    "error": str(e)
                }
                logger.error(f"Failed to click element number {params.get('element_number')}: {str(e)}")
                return False

        if "coordinates" in params:
            try:
                x, y = params["coordinates"]
                await self.page.mouse.click(x, y)
                logger.info(f"Clicked at coordinates ({x}, {y})")
                # Small delay after clicking to let any interactions settle
                await asyncio.sleep(0.5)
                return True
            except Exception as e:
                self.last_error = f"Coordinate click error: {str(e)}"
                self.error_context["coordinate_click_error"] = {
                    "coordinates": params["coordinates"],
                    "error": str(e)
                }
                logger.error(f"Failed to click at coordinates: {str(e)}")
                return False
        elif "selector" in params:
            try:
                selector = params["selector"]
                
                # Check for common cookie consent button IDs
                cookie_selectors = [
                    "#L2AGLb",  # Google cookie consent
                    "#onetrust-accept-btn-handler",  # OneTrust
                    ".cc-accept", 
                    ".cc-dismiss",
                    "#accept-cookie-banner",
                    ".accept-cookies",
                    "#accept-all-cookies",
                    ".acceptCookies",
                    "#acceptCookies",
                    "#accept-cookies-button",
                    "[aria-label='Accept cookies']",
                    "[aria-label='Accept all cookies']",
                    "button:has-text('Accept all')",
                    "button:has-text('Accept cookies')",
                    "button:has-text('I accept')",
                    "button:has-text('Allow cookies')",
                    "button:has-text('Accepte tutto')",  # Italian
                    "button:has-text('Acepto')",  # Spanish
                    "button:has-text('Ich akzeptiere')",  # German
                    "button:has-text('J'accepte')"  # French
                ]
                
                # If this appears to be a cookie banner click but the selector isn't a known one
                if any(cookie_keyword in selector.lower() for cookie_keyword in ["cookie", "consent", "accept", "agree"]):
                    logger.info(f"Detected potential cookie banner selector: {selector}")
                    
                    # Try direct selector first
                    try:
                        await self.page.wait_for_selector(selector, state="visible", timeout=2000)
                        await self.page.click(selector)
                        logger.info(f"Clicked cookie banner with selector: {selector}")
                        await asyncio.sleep(0.5)
                        return True
                    except Exception as direct_error:
                        logger.warning(f"Direct cookie selector failed: {str(direct_error)}")
                        
                        # Try to find a cookie banner using known selectors
                        for cookie_selector in cookie_selectors:
                            try:
                                if await self.page.query_selector(cookie_selector):
                                    await self.page.click(cookie_selector)
                                    logger.info(f"Clicked cookie banner with alternative selector: {cookie_selector}")
                                    await asyncio.sleep(0.5)
                                    return True
                            except Exception:
                                continue
                        
                        # If we get here, none of the known selectors worked
                        logger.warning("Failed to find a cookie banner with known selectors")
                
                # Wait for the element to be visible first
                try:
                    await self.page.wait_for_selector(selector, state="visible", timeout=5000)
                except Exception as wait_error:
                    logger.warning(f"Element not visible, attempting click anyway: {str(wait_error)}")
                    # Element might not appear, but we'll still try to click it
                
                # Try multiple click approaches
                try:
                    await self.page.click(selector)
                    logger.info(f"Clicked element with selector: {selector}")
                    await asyncio.sleep(0.5)
                    return True
                except Exception as click_error:
                    logger.warning(f"Standard click failed: {str(click_error)}, trying JS click")
                    
                    # Try using JavaScript to click the element
                    try:
                        await self.page.evaluate("""() => {
                            const element = document.querySelector('%s');
                            if (element) element.click();
                        }""" % selector.replace("'", "\\'"))
                        logger.info(f"Clicked element with JS click for selector: {selector}")
                        await asyncio.sleep(0.5)
                        return True
                    except Exception as js_error:
                        self.last_error = f"Selector click error (both standard and JS): {str(click_error)}, {str(js_error)}"
                        self.error_context["selector_click_error"] = {
                            "selector": selector,
                            "error": f"{str(click_error)}; {str(js_error)}"
                        }
                        logger.error(f"Failed to click selector {selector} with both methods")
                        return False
                
            except Exception as e:
                self.last_error = f"Selector click error: {str(e)}"
                self.error_context["selector_click_error"] = {
                    "selector": params["selector"],
                    "error": str(e)
                }
                logger.error(f"Failed to click selector {params['selector']}: {str(e)}")
                return False
        else:
            self.last_error = "Click action missing target (element number, coordinates, or selector)"
            logger.warning("Click action missing both coordinates and selector")
            return False
    
    async def handle_type(self, params: Dict) -> bool:
        """Handle a typing action."""
        if "text" not in params:
            self.last_error = "Type action missing text parameter"
            logger.warning("Type action missing text parameter")
            return False
            
        text = params["text"]
        
        # Support for element numbers from the detected elements list
        if "element_number" in params:
            try:
                element_number = int(params["element_number"])
                
                # Re-extract elements to get the current state
                elements, _ = await self.extract_elements()
                
                # Filter to just input elements
                input_elements = [
                    el for el in elements 
                    if el.get("type") in ["input", "dropdown", "checkbox"]
                ]
                
                # Check if element number is valid for inputs
                if 1 <= element_number <= len(input_elements):
                    target_element = input_elements[element_number - 1]
                    
                    # Try to use selector if available
                    if "selector" in target_element:
                        try:
                            # Clear the field first
                            await self.page.fill(target_element["selector"], "")
                            # Then type the text
                            await self.page.fill(target_element["selector"], text)
                            logger.info(f"Typed '{text}' into element {element_number} using selector: {target_element['selector']}")
                            return True
                        except Exception as fill_error:
                            logger.warning(f"Selector fill failed, falling back to coordinates: {str(fill_error)}")
                            # Fallback to coordinates
                    
                    # Use coordinates as fallback
                    x = target_element["position"]["x"]
                    y = target_element["position"]["y"]
                    await self.page.mouse.click(x, y)
                    
                    # Try to select all text and delete it first
                    await self.page.keyboard.press("Control+A")
                    await self.page.keyboard.press("Delete")
                    
                    await self.page.keyboard.type(text)
                    logger.info(f"Typed '{text}' into element {element_number} at coordinates ({x}, {y})")
                    return True
                else:
                    self.last_error = f"Invalid input element number: {element_number}"
                    self.error_context["input_element_error"] = {
                        "requested": element_number,
                        "available": len(input_elements)
                    }
                    logger.warning(f"Invalid input element number: {element_number}")
                    return False
            except Exception as e:
                self.last_error = f"Element type error: {str(e)}"
                self.error_context["element_type_error"] = {
                    "element_number": params.get('element_number'),
                    "error": str(e)
                }
                logger.error(f"Failed to type into element number {params.get('element_number')}: {str(e)}")
                return False
        
        if "selector" in params:
            try:
                selector = params["selector"]
                # Try to wait for the element to be visible first
                try:
                    await self.page.wait_for_selector(selector, state="visible", timeout=5000)
                except Exception as wait_error:
                    logger.warning(f"Element not visible, attempting to type anyway: {str(wait_error)}")
                    # Element might not appear, but we'll still try to type into it
                
                # Clear the field first if possible
                try:
                    await self.page.fill(selector, "")
                except Exception as clear_error:
                    logger.warning(f"Failed to clear field with fill, trying keyboard shortcuts: {str(clear_error)}")
                    # If fill doesn't work, try clicking and pressing keyboard shortcuts
                    await self.page.click(selector)
                    await self.page.keyboard.press("Control+A")
                    await self.page.keyboard.press("Delete")
                
                await self.page.fill(selector, text)
                logger.info(f"Typed '{text}' into element with selector: {selector}")
                return True
            except Exception as e:
                self.last_error = f"Selector type error: {str(e)}"
                self.error_context["selector_type_error"] = {
                    "selector": params["selector"],
                    "error": str(e)
                }
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
                self.last_error = f"Coordinate type error: {str(e)}"
                self.error_context["coordinate_type_error"] = {
                    "coordinates": params["coordinates"],
                    "error": str(e)
                }
                logger.error(f"Failed to type at coordinates: {str(e)}")
                return False
        else:
            self.last_error = "Type action missing target (element number, coordinates, or selector)"
            logger.warning("Type action missing target specification")
            return False
    
    async def handle_navigate(self, params: Dict) -> bool:
        """Handle a navigation action."""
        if "url" not in params:
            self.last_error = "Navigate action missing URL parameter"
            logger.warning("Navigate action missing URL parameter")
            return False
            
        return await self.navigate_to(params["url"])
    
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
            self.last_error = f"Scroll error: {str(e)}"
            self.error_context["scroll_error"] = {
                "direction": direction,
                "amount": amount,
                "error": str(e)
            }
            logger.error(f"Failed to scroll: {str(e)}")
            return False
    
    async def wait_for_navigation_or_stability(self) -> None:
        """Wait for either navigation to complete or page to become stable."""
        try:
            # First wait for any outstanding network requests to complete
            try:
                await self.page.wait_for_load_state("networkidle", timeout=3000)
                logger.debug("Network is idle")
            except Exception as e:
                # Timeout is expected if the page is already stable
                logger.debug(f"Timeout waiting for networkidle: {str(e)}")
            
            # Also wait for the DOM content to be loaded
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
                logger.debug("DOM content loaded")
            except Exception as e:
                logger.debug(f"Timeout waiting for domcontentloaded: {str(e)}")
            
            # Check DOM stability by comparing element counts over time
            try:
                element_count_1 = await self.page.evaluate("document.querySelectorAll('*').length")
                await asyncio.sleep(0.5)
                element_count_2 = await self.page.evaluate("document.querySelectorAll('*').length")
                
                # If the element count is still changing significantly, wait a bit longer
                if abs(element_count_2 - element_count_1) > 5:
                    logger.debug(f"DOM still changing: {element_count_1} vs {element_count_2}, waiting longer")
                    await asyncio.sleep(1.0)
                else:
                    logger.debug(f"DOM appears stable: {element_count_2} elements")
            except Exception as e:
                logger.debug(f"Error during DOM stability check: {str(e)}")
            
            # Additional stability check for visual stability
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error during wait_for_navigation_or_stability: {str(e)}")
            # Still need to wait a bit even on error
            await asyncio.sleep(1.0)
    
    def get_error_context_for_llm(self) -> str:
        """
        Format error context for the LLM.
        
        Returns:
            Formatted error context string
        """
        if not self.last_error and not self.error_context:
            return ""
        
        error_parts = []
        
        if self.last_error:
            error_parts.append(f"Last error: {self.last_error}")
        
        if self.error_context:
            for error_type, error_details in self.error_context.items():
                if isinstance(error_details, dict):
                    details_str = ", ".join(f"{k}: {v}" for k, v in error_details.items())
                    error_parts.append(f"{error_type}: {details_str}")
                else:
                    error_parts.append(f"{error_type}: {error_details}")
        
        return "\n".join(error_parts) 