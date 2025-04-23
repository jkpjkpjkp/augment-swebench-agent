"""Instruction Prompt

This prompt is used to instruct the agent on what to do for SWE-bench tasks.

This forks from the instruction specified in Anthropic's blogpost:
https://www.anthropic.com/engineering/swe-bench-sonnet.
"""

INSTRUCTION_PROMPT = """
<images>
{location}
</images>
I've uploaded a python code repository in the directory {location} (not in /tmp/inputs). Consider the following PR description:

<question>
{pr_description}
</question>

Can you help me investigate the images in the repository so that the question specified in the <question> is answered?

Your task is to thoroughly investigate the image in the {location} directory to ensure the <question> is most accurately answered.

Follow these steps to resolve the issue:
1. As a first step, it would be a good idea to explore the image to familiarize yourself with its structure.
3. Use the sequential_thinking tool to plan your fix. Reflect on 5-7 different possible sources of the problem, distill those down to 1-2 most likely sources, and then add logs to validate your assumptions before moving onto implementing the actual code fix
3.5 Use the cropping tool to zoom in on specific parts of the image.
4. Edit the image to mark the progress made
6. Think about edgecases and make sure your solution handles them as well


GUIDE FOR HOW TO USE "sequential_thinking" TOOL:
- Your thinking should be thorough and so it's fine if it's very long. Set totalThoughts to at least 5, but setting it up to 25 is fine as well. You'll need more total thoughts when you are considering multiple possible solutions or root causes for an issue.
- Use this tool as much as you find necessary to improve the quality of your answers.
- You can run bash commands (like tests, a reproduction script, or 'grep'/'find' to find relevant context) in between thoughts.
- The sequential_thinking tool can help you break down complex problems, analyze image step-by-step, and ensure a thorough approach to problem-solving.
- Don't hesitate to use it multiple times throughout your thought process to enhance the depth and accuracy of your solutions.

TIPS:
- You must make changes in the {location} directory in order to ensure the requirements specified in the <pr_description> are met. Leaving the directory unchanged is not a valid solution.
- Do NOT make tool calls inside thoughts passed to sequential_thinking tool. For example, do NOT do this: {{'thought': 'I need to look at the actual implementation of `apps.get_models()` in this version of Django to see if there\'s a bug. Let me check the Django apps module:\n\n<function_calls>\n<invoke name="str_replace_editor">\n<parameter name="command">view</parameter>\n<parameter name="path">django/apps/registry.py</parameter></invoke>', 'path': 'django/apps/registry.py'}}
- Respect the tool specifications. If a field is required, make sure to provide a value for it. For example "thoughtNumber" is required by the sequential_thinking tool.
- When you run "ls" with the bash tool, the "view" command with the "str_replace_editor" tool, or variants of those, you may see a symlink like "fileA -> /home/augment/docker/volumes/_data/fileA". You can safely ignore the symlink and just use "fileA" as the path when read, editing, or executing the file.
- When you need to find information about the codebase, use "grep" and "find" to search for relevant files and code with the bash tool
- Use your bash tool to set up any necessary environment variables, such as those needed to run tests.
"""
