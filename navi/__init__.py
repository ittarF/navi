"""
Navi - A web navigation AI agent that uses Ollama for decision making.

This package provides a system for autonomous web navigation using
local LLMs via Ollama to make decisions based on screenshots.
"""

from navi.agent import NaviAgent
from navi.llm import LLMDecisionMaker
from navi.browser_controller import BrowserController

__version__ = "0.1.0"
__all__ = ["NaviAgent", "LLMDecisionMaker", "BrowserController"] 