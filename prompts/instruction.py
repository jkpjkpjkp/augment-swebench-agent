"""Instruction Prompt

This prompt is used to instruct the agent on what to do for Visual Question Answering (VQA) tasks.

It guides the agent to analyze images and answer questions about them using the available image tools.
"""

INSTRUCTION_PROMPT = """
I've loaded an image in the directory {location}. Consider the following question about the image:

<question>
{pr_description}
</question>

Can you help me analyze this image and answer the question?

Your task is to thoroughly investigate the image in the {location} directory to provide the most accurate answer to the question.

Follow these steps to analyze the image:
1. First, list the images in the workspace to see what's available
2. Select the image to get basic information about it (size, format, etc.)
3. Create views (crops) of specific regions of interest in the image
4. Analyze each view in detail and describe what you see
5. Mark regions as analyzed by blacking them out when you're done with them
6. Provide a comprehensive answer to the question based on your analysis

Available Image Tools:
- list_images: List all images and views in the workspace
- select_image: Select an image or view to get information about it
- crop_image: Create a new view by cropping with coordinates (x1, y1, x2, y2)
- blackout_image: Black out a view to mark it as analyzed

TIPS:
- Be methodical in your analysis, examining the image section by section
- Create multiple views to focus on different parts of the image
- Be detailed in your descriptions of what you see in each view
- Consider the context and purpose of the image when answering the question
- If the image contains text, make sure to read and report it accurately
- When you're done with your analysis, use the complete tool to submit your final answer
"""
