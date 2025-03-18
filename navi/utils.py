"""
Utility functions for the Navi agent.

This module provides various helper functions for tasks like screenshot handling,
ID generation, and element detection.
"""

import base64
import os
import time
import logging
import json
import re
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple

from PIL import Image
from playwright.async_api import Page, ElementHandle

# Configure logging
logger = logging.getLogger("navi.utils")

def create_task_id() -> str:
    """
    Create a unique task ID based on timestamp.
    
    Returns:
        String task ID
    """
    timestamp = int(time.time())
    return f"task_{timestamp}"

async def save_screenshot(task_id: str, step: int, screenshot_base64: str, history_dir: str = "history") -> str:
    """
    Save a screenshot for a step to the history directory.
    
    Args:
        task_id: ID of the current task
        step: Step number
        screenshot_base64: Base64-encoded screenshot image
        history_dir: Directory to save history files
    """
    if not task_id:
        return
        
    screenshots_dir = os.path.join(history_dir, task_id, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    screenshot_path = os.path.join(screenshots_dir, f"step_{step}.png")
    
    # Decode and save the screenshot
    try:
        screenshot_bytes = base64.b64decode(screenshot_base64)
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
        logger.debug(f"Saved screenshot for step {step}")
        return screenshot_path
    except Exception as e:
        logger.error(f"Failed to save screenshot for step {step}: {str(e)}")
        return None

async def save_raw_screenshot(task_id: str, step: int, page, history_dir: str) -> None:
    """
    Save a raw screenshot directly from the page object for debugging purposes.
    This bypasses any processing that might be happening in the regular screenshot pipeline.
    
    Args:
        task_id: Unique identifier for the task
        step: Current step number
        page: Playwright page object
        history_dir: Directory to save history files
    """
    # Create directories if they don't exist
    task_dir = os.path.join(history_dir, task_id)
    debug_dir = os.path.join(task_dir, "debug_screenshots")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save the raw screenshot
    screenshot_path = os.path.join(debug_dir, f"raw_step_{step}.png")
    try:
        await page.screenshot(path=screenshot_path, full_page=True)
        logger.debug(f"Saved raw screenshot to {screenshot_path}")
    except Exception as e:
        logger.error(f"Failed to save raw screenshot: {str(e)}")
        
    # Also save a screenshot of the viewport only
    viewport_path = os.path.join(debug_dir, f"viewport_step_{step}.png")
    try:
        await page.screenshot(path=viewport_path, full_page=False)
        logger.debug(f"Saved viewport screenshot to {viewport_path}")
    except Exception as e:
        logger.error(f"Failed to save viewport screenshot: {str(e)}")

def format_action_for_display(action: Dict) -> str:
    """
    Format an action for display in the console.
    
    Args:
        action: Action dictionary from LLM
        
    Returns:
        Formatted string representation of the action
    """
    action_type = action.get("action_type", "UNKNOWN")
    params = action.get("action_params", {})
    
    if action_type == "CLICK":
        if "element_number" in params:
            return f"CLICK on element #{params['element_number']}"
        elif "coordinates" in params:
            return f"CLICK at {params['coordinates']}"
        elif "selector" in params:
            return f"CLICK on '{params['selector']}'"
        return "CLICK (no target)"
    
    elif action_type == "TYPE":
        text = params.get("text", "")
        if len(text) > 20:
            text = text[:17] + "..."
        
        if "element_number" in params:
            return f"TYPE '{text}' into element #{params['element_number']}"
        elif "coordinates" in params:
            return f"TYPE '{text}' at {params['coordinates']}"
        elif "selector" in params:
            return f"TYPE '{text}' in '{params['selector']}'"
        return f"TYPE '{text}' (no target)"
    
    elif action_type == "NAVIGATE":
        url = params.get("url", "")
        if len(url) > 30:
            url = url[:27] + "..."
        return f"NAVIGATE to {url}"
    
    elif action_type == "SCROLL":
        direction = params.get("direction", "down")
        amount = params.get("amount", 0)
        return f"SCROLL {direction} by {amount}px"
    
    elif action_type == "COMPLETE":
        return "COMPLETE task"
    
    return f"{action_type} {str(params)}"

def safe_str(obj: any) -> str:
    """
    Safely convert an object to string, handling potential encoding issues.
    
    Args:
        obj: Object to convert to string
        
    Returns:
        String representation of the object
    """
    try:
        return str(obj)
    except Exception:
        return "[Object could not be converted to string]"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length and add ellipsis if needed.
    
    Args:
        text: Text to truncate
        max_length: Maximum length for the text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def is_likely_cookie_banner_element(element_info: Dict[str, Any]) -> bool:
    """
    Determine if an element is likely part of a cookie banner.
    
    Args:
        element_info: Dictionary with element information
    
    Returns:
        Boolean indicating if the element is likely part of a cookie banner
    """
    # Check text content for cookie-related terms
    text = element_info.get("text", "").lower() if element_info.get("text") else ""
    
    cookie_terms = [
        "cookie", "cookies", "consent", "accept", "privacy", "gdpr", "ccpa", 
        "agree", "preferences", "accept all", "accept cookies", "policy",
        "tracking", "terms", "privacy policy", "cookie policy", "privacy settings",
        "manage cookies", "necessary cookies", "we use cookies"
    ]
    
    # Check if any of the terms are in the text
    if any(term in text for term in cookie_terms):
        return True
    
    # Check attributes for cookie-related IDs, classes, or roles
    attributes = element_info.get("attributes", {})
    for attr_name, attr_value in attributes.items():
        if not attr_value:
            continue
            
        attr_value_lower = attr_value.lower()
        if any(term in attr_value_lower for term in cookie_terms):
            return True
            
        # Check for common cookie banner IDs and classes
        if attr_name == "id" or attr_name == "class":
            banner_patterns = [
                r"cookie[s-]?banner", r"cookie[s-]?consent", r"cookie[s-]?policy", 
                r"cookie[s-]?notice", r"cookie[s-]?notification", r"cookie[s-]?alert",
                r"privacy[s-]?banner", r"privacy[s-]?notice", r"consent[s-]?banner",
                r"gdpr", r"ccpa", r"consent", r"privacy-overlay", r"cookie-bar"
            ]
            
            for pattern in banner_patterns:
                if re.search(pattern, attr_value_lower):
                    return True
    
    # Check for specific selectors that often indicate cookie banners
    if "selector" in element_info:
        selector = element_info["selector"].lower()
        selector_patterns = [
            r"#cookie", r"#consent", r"#gdpr", r"#ccpa", r"\.cookie", r"\.consent",
            r"\.gdpr", r"\.ccpa", r"cookie-banner", r"cookie-consent", r"privacy-banner"
        ]
        
        for pattern in selector_patterns:
            if re.search(pattern, selector):
                return True
    
    return False

async def extract_page_elements(page: Page) -> List[Dict[str, Any]]:
    """
    Extract interactive elements from the page to help the LLM make better decisions.
    
    Args:
        page: Playwright page object
        
    Returns:
        List of dictionaries containing element information
    """
    elements = []
    errors = {}
    
    # Function to get element position and dimensions
    async def get_element_box(element: ElementHandle) -> Optional[Dict[str, float]]:
        try:
            box = await element.bounding_box()
            if box:
                return {
                    "x": box["x"] + box["width"] / 2,  # Center X
                    "y": box["y"] + box["height"] / 2,  # Center Y
                    "width": box["width"],
                    "height": box["height"]
                }
        except Exception as e:
            logger.debug(f"Error getting bounding box: {str(e)}")
        return None
    
    # Function to extract basic attributes from an element
    async def extract_element_info(element: ElementHandle, element_type: str) -> Optional[Dict[str, Any]]:
        try:
            # Get element attributes
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            
            # Get text content
            text_content = await element.evaluate("el => el.textContent || ''")
            text_content = text_content.strip()
            if len(text_content) > 50:
                text_content = text_content[:47] + "..."
            
            # Get relevant attributes
            attributes = {}
            
            # Universal attributes
            for attr in ["id", "name", "class", "role", "aria-label", "title", "placeholder"]:
                value = await element.get_attribute(attr)
                if value:
                    attributes[attr] = value
            
            # Type-specific attributes
            if element_type == "input":
                for attr in ["type", "value"]:
                    value = await element.get_attribute(attr)
                    if value:
                        attributes[attr] = value
            elif element_type == "link":
                href = await element.get_attribute("href")
                if href:
                    attributes["href"] = href
            
            # Additional checks for cookie banners
            is_visible = await element.is_visible()
            if not is_visible:
                return None  # Skip invisible elements
                
            # Get element box for positioning
            box = await get_element_box(element)
            if not box:
                return None  # Skip if we can't get position
            
            # Create element info
            element_info = {
                "type": element_type,
                "tag": tag_name,
                "text": text_content if text_content else None,
                "attributes": attributes,
                "position": box,
                "visible": is_visible
            }
            
            # Check if this might be a cookie banner element
            element_info["is_cookie_banner"] = is_likely_cookie_banner_element(element_info)
            
            # Generate a CSS selector for this element
            try:
                selector = await element.evaluate("""el => {
                    // Try to create a unique selector
                    if (el.id) {
                        return `#${el.id}`;
                    }
                    
                    // Next try name for form elements
                    if (el.name && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA')) {
                        return `${el.tagName.toLowerCase()}[name="${el.name}"]`;
                    }
                    
                    // Fallback to a more complex selector
                    let parts = [];
                    parts.push(el.tagName.toLowerCase());
                    
                    if (el.className) {
                        const classes = Array.from(el.classList).join('.');
                        if (classes) {
                            parts[0] += `.${classes}`;
                        }
                    }
                    
                    // Add position among siblings
                    const parent = el.parentNode;
                    if (parent) {
                        const siblings = Array.from(parent.children);
                        const index = siblings.indexOf(el);
                        parts[0] += `:nth-child(${index + 1})`;
                    }
                    
                    return parts.join(' ');
                }""")
                element_info["selector"] = selector
            except Exception as e:
                logger.debug(f"Error creating selector: {str(e)}")
                # If we can't get a selector, still keep the element
            
            return element_info
        except Exception as e:
            logger.debug(f"Error extracting element info: {str(e)}")
            return None
    
    # Find specific cookie banners first
    try:
        # Common cookie banner patterns
        cookie_banner_selectors = [
            "#cookieBanner", "#cookie-banner", "#cookie-consent", "#cookie-notice",
            "#gdpr-banner", "#privacy-banner", "#consent-banner", "#onetrust-banner-sdk",
            ".cookie-banner", ".cookie-consent", ".cookie-notice", ".gdpr-banner",
            ".privacy-banner", ".consent-banner", "#CybotCookiebotDialog",
            "[aria-label*='cookie']", "[aria-label*='consent']", "[role='dialog'][aria-label*='cookie']",
            "div[class*='cookie-banner']", "div[class*='cookie-consent']", "div[class*='cookie-notice']",
            "div[id*='cookie-banner']", "div[id*='cookie-consent']", "div[id*='cookie-notice']"
        ]
        
        for selector in cookie_banner_selectors:
            try:
                cookie_elements = await page.query_selector_all(selector)
                for element in cookie_elements:
                    if await element.is_visible():
                        element_info = await extract_element_info(element, "cookie_banner")
                        if element_info:
                            element_info["is_cookie_banner"] = True
                            elements.append(element_info)
            except Exception as e:
                errors[f"cookie_banner_{selector}"] = str(e)
                logger.debug(f"Error finding cookie banner with selector {selector}: {str(e)}")
    except Exception as e:
        errors["cookie_banner_extraction"] = str(e)
        logger.error(f"Error finding cookie banners: {str(e)}")
    
    # Find buttons
    try:
        # Get semantic buttons (both <button> elements and button-like elements)
        button_elements = await page.query_selector_all("button, [role='button'], input[type='button'], input[type='submit']")
        
        for element in button_elements:
            element_info = await extract_element_info(element, "button")
            if element_info:
                elements.append(element_info)
    except Exception as e:
        errors["button_extraction"] = str(e)
        logger.error(f"Error extracting buttons: {e}")
    
    # Find links
    try:
        # Get all links
        link_elements = await page.query_selector_all("a[href]")
        
        for element in link_elements:
            element_info = await extract_element_info(element, "link")
            if element_info:
                elements.append(element_info)
    except Exception as e:
        errors["link_extraction"] = str(e)
        logger.error(f"Error extracting links: {e}")
    
    # Find input fields
    try:
        # Get all input elements
        input_elements = await page.query_selector_all("input:not([type='hidden']):not([type='button']):not([type='submit']), textarea, select")
        
        for element in input_elements:
            element_info = await extract_element_info(element, "input")
            if element_info:
                elements.append(element_info)
    except Exception as e:
        errors["input_extraction"] = str(e)
        logger.error(f"Error extracting input fields: {e}")
    
    # Find checkboxes and radio buttons
    try:
        # Get all checkboxes and radio buttons
        checkbox_elements = await page.query_selector_all("input[type='checkbox'], input[type='radio']")
        
        for element in checkbox_elements:
            element_info = await extract_element_info(element, "checkbox")
            if element_info:
                elements.append(element_info)
    except Exception as e:
        errors["checkbox_extraction"] = str(e)
        logger.error(f"Error extracting checkboxes and radio buttons: {e}")
    
    # Find dropdown menus
    try:
        # Get all select elements
        select_elements = await page.query_selector_all("select")
        
        for element in select_elements:
            element_info = await extract_element_info(element, "dropdown")
            if element_info:
                elements.append(element_info)
    except Exception as e:
        errors["dropdown_extraction"] = str(e)
        logger.error(f"Error extracting dropdown menus: {e}")
    
    # Add some other potentially interactive elements that don't fit the categories above
    try:
        # Get elements that might be interactive based on common patterns
        other_elements = await page.query_selector_all("[onclick], [tabindex]:not([tabindex='-1']), [role='tab'], [role='menuitem'], [role='option']")
        
        for element in other_elements:
            element_info = await extract_element_info(element, "interactive")
            if element_info:
                elements.append(element_info)
    except Exception as e:
        errors["interactive_extraction"] = str(e)
        logger.error(f"Error extracting other interactive elements: {e}")
    
    # Special case: Try to find cookie banners directly in iframes
    try:
        frames = page.frames
        for frame in frames:
            if frame != page.main_frame:
                try:
                    # Try to detect cookie banners in this frame
                    for selector in cookie_banner_selectors:
                        try:
                            elements_in_frame = await frame.query_selector_all(selector)
                            for el in elements_in_frame:
                                if await el.is_visible():
                                    # We found something, but can't extract proper details from iframe
                                    # Add a placeholder element with frame information
                                    frame_url = await frame.evaluate("window.location.href")
                                    frame_name = await frame.evaluate("window.name || ''")
                                    
                                    element_info = {
                                        "type": "iframe_cookie_banner",
                                        "tag": "iframe",
                                        "text": "Cookie consent banner in iframe",
                                        "attributes": {
                                            "frame_url": frame_url,
                                            "frame_name": frame_name
                                        },
                                        "position": {"x": 300, "y": 200, "width": 400, "height": 100},
                                        "is_cookie_banner": True,
                                        "visible": True
                                    }
                                    elements.append(element_info)
                                    break
                        except:
                            continue
                except:
                    continue
    except Exception as e:
        errors["iframe_extraction"] = str(e)
        logger.error(f"Error examining iframes: {e}")
    
    # Add error information if we had failures but still found some elements
    if errors and elements:
        elements.append({
            "type": "error_info",
            "errors": errors,
            "text": "Some elements might be missing due to extraction errors",
            "position": {"x": 0, "y": 0, "width": 0, "height": 0}
        })
    
    # Remove duplicates based on position and type
    unique_elements = []
    seen_positions = set()
    
    for element in elements:
        position = element.get("position", {})
        position_key = (
            round(position.get("x", 0) / 5) * 5,  # Round to nearest 5px
            round(position.get("y", 0) / 5) * 5,
            element.get("type", "unknown")
        )
        
        if position_key not in seen_positions:
            seen_positions.add(position_key)
            unique_elements.append(element)
    
    return unique_elements

def format_page_elements_for_llm(elements: List[Dict[str, Any]], max_elements: int = 20) -> str:
    """
    Format the page elements into a concise text representation for the LLM.
    
    Args:
        elements: List of element dictionaries from extract_page_elements
        max_elements: Maximum number of elements to include
        
    Returns:
        Formatted string for the LLM
    """
    if not elements:
        return "No interactive elements detected on the page."
    
    # First extract any cookie banners and error info
    cookie_banners = [el for el in elements if el.get("is_cookie_banner", False)]
    error_infos = [el for el in elements if el.get("type") == "error_info"]
    regular_elements = [el for el in elements if not el.get("is_cookie_banner", False) and el.get("type") != "error_info"]
    
    # Sort elements by likely importance/prominence
    # - Buttons and inputs near the top of the page
    # - Elements with text content
    # - Larger elements
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
    
    sorted_elements = sorted(regular_elements, key=element_priority, reverse=True)
    
    # Limit the number of regular elements (keep space for cookie banners and errors)
    max_regular_elements = max_elements - len(cookie_banners) - len(error_infos)
    if len(sorted_elements) > max_regular_elements:
        sorted_elements = sorted_elements[:max_regular_elements]
    
    # Combine elements in this order: cookie banners first, then regular elements, then errors
    display_elements = cookie_banners + sorted_elements + error_infos
    
    result = []
    cookie_banner_count = len(cookie_banners)
    
    if cookie_banner_count > 0:
        result.append(f"*** DETECTED {cookie_banner_count} COOKIE CONSENT/PRIVACY BANNER(S) ***")
    
    result.append(f"Detected {len(display_elements)} interactive elements (showing most important):")
    
    for i, element in enumerate(display_elements):
        element_type = element.get("type", "unknown")
        tag = element.get("tag", "")
        text = element.get("text", "")
        attributes = element.get("attributes", {})
        position = element.get("position", {})
        
        # Format element description
        description = f"[{i+1}] "
        
        # Highlight cookie banners
        if element.get("is_cookie_banner", False):
            description += "🍪 COOKIE BANNER: "
        
        description += f"{element_type.upper()} "
        
        # Add descriptive text based on element type
        if element_type == "error_info":
            description = f"[ERROR] Some elements might be missing. Extraction errors occurred."
        elif element_type == "cookie_banner" or element_type == "iframe_cookie_banner":
            description += "Cookie consent/privacy notice"
            if text:
                description += f" \"{text}\""
        elif element_type == "button":
            description += f"button"
            if text:
                description += f" \"{text}\""
            elif "value" in attributes:
                description += f" with value \"{attributes['value']}\""
        elif element_type == "input":
            input_type = attributes.get("type", "text")
            description += f"{input_type} field"
            if "placeholder" in attributes:
                description += f" placeholder=\"{attributes['placeholder']}\""
            elif "name" in attributes:
                description += f" name=\"{attributes['name']}\""
        elif element_type == "link":
            description += f"link"
            if text:
                description += f" \"{text}\""
            if "href" in attributes:
                href = attributes["href"]
                if len(href) > 30:
                    href = href[:27] + "..."
                description += f" → {href}"
        elif element_type == "dropdown":
            description += f"dropdown"
            if "name" in attributes:
                description += f" name=\"{attributes['name']}\""
        elif element_type == "checkbox":
            checkbox_type = attributes.get("type", "checkbox")
            description += f"{checkbox_type}"
            if "name" in attributes:
                description += f" name=\"{attributes['name']}\""
            
        # Add position info
        if element_type != "error_info":
            description += f" at ({position.get('x', 0):.0f}, {position.get('y', 0):.0f})"
        
        # Add selector if available
        if "selector" in element and element_type != "error_info":
            selector = element["selector"]
            if len(selector) > 40:
                selector = selector[:37] + "..."
            description += f" selector: {selector}"
            
        result.append(description)
    
    if cookie_banner_count > 0:
        result.append("\nNOTE: Cookie banners are common on websites and often need to be accepted. Consider clicking the accept/agree button if you see a cookie banner.")
    
    return "\n".join(result) 