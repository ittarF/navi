#!/usr/bin/env python3
"""
Example script showing how to use WebNavigator for a custom search task.
"""

import os
import sys
import asyncio

# Add the parent directory to the path so we can import the WebNavigator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_navigator import WebNavigator

async def run_search_task():
    """Run a simple search task using WebNavigator."""
    # Define the task
    task_description = "Search for 'climate change solutions' and find information about renewable energy options"
    
    # Create WebNavigator with custom settings
    async with WebNavigator(
        headless=False,         # Show the browser window
        model_name="gemma3",    # Use llama3 model
        save_history=True,      # Save task history
        history_dir="examples/history"  # Save history in examples/history directory
    ) as navigator:
        # Run the task with Google as the starting point
        result = await navigator.run_task(
            task_description=task_description,
            starting_url="https://www.google.com"
        )
        
        # Print results
        print("\n--- Task Results ---")
        print(f"Success: {result['success']}")
        print(f"Steps taken: {result['steps_taken']}")
        print(f"Final URL: {result['final_url']}")
        
        if result['success']:
            print("\nTask successfully completed!")
            print(f"Result: {result['history'][-1].get('result', 'No result provided')}")
        else:
            print("\nTask not completed.")
            print(f"Reason: {result.get('reason', 'Unknown reason')}")
            
        # Show where history was saved
        print(f"\nHistory saved to: {os.path.join('examples/history', result['task_id'])}")
        print("\nYou can replay this task using:")
        print(f"python replay_task.py {result['task_id']} --history-dir examples/history")

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples/history", exist_ok=True)
    
    # Run the example task
    asyncio.run(run_search_task()) 