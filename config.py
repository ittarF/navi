"""
Configuration settings for the Navi application.

This module contains default settings that can be overridden via command-line arguments.
"""

import os

# Browser Settings
HEADLESS = os.environ.get("NAVI_HEADLESS", "False").lower() in ("true", "1", "t")
SCREENSHOT_WIDTH = int(os.environ.get("NAVI_SCREENSHOT_WIDTH", "1280"))
BROWSER_HEIGHT = int(os.environ.get("NAVI_BROWSER_HEIGHT", "800"))
USER_AGENT = os.environ.get(
    "NAVI_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
)
DEFAULT_START_URL = os.environ.get("NAVI_DEFAULT_START_URL", "https://www.google.com")

# LLM Settings
OLLAMA_BASE_URL = os.environ.get("NAVI_OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("NAVI_MODEL_NAME", "gemma3")
LLM_TEMPERATURE = float(os.environ.get("NAVI_LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = float(os.environ.get("NAVI_LLM_TIMEOUT", "120.0"))
LLM_MAX_TOKENS = int(os.environ.get("NAVI_LLM_MAX_TOKENS", "1024"))
LLM_STREAM_DEFAULT = os.environ.get("NAVI_LLM_STREAM", "False").lower() in ("true", "1", "t")

# Task Settings
MAX_STEPS = int(os.environ.get("NAVI_MAX_STEPS", "15"))
MAX_ERRORS = int(os.environ.get("NAVI_MAX_ERRORS", "3"))

# History Settings
SAVE_HISTORY_DEFAULT = os.environ.get("NAVI_SAVE_HISTORY", "True").lower() in ("true", "1", "t")
HISTORY_DIR = os.environ.get("NAVI_HISTORY_DIR", "history")

# Output Formatting
EMOJI_ENABLED = os.environ.get("NAVI_EMOJI_ENABLED", "True").lower() in ("true", "1", "t")

# Debug Settings
DEBUG = os.environ.get("NAVI_DEBUG", "False").lower() in ("true", "1", "t") 