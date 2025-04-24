#!/usr/bin/env python3
import logging
from utils.common import DialogMessages

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Create a DialogMessages instance
d = DialogMessages(logger_for_agent_logs=logger)

# Add a user prompt
d.add_user_prompt("How many geese are in this image?")

# Print the initial question
print(f"Initial question: {d.initial_question}")

# Get the messages for the LLM client
messages = d.get_messages_for_llm_client()
if messages:
    print(f"Message text (first 200 chars): {messages[0][0].text[:200]}")
else:
    print("No messages returned")

# Clear the dialog
d.clear()

# Check if initial_question was reset
print(f"After clear, initial question: {d.initial_question}")
