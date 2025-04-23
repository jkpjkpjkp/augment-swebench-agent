"""Instruction Prompt

This prompt is used to instruct the agent on what to do for Visual Question Answering (VQA) tasks.

It guides the agent to analyze images and answer questions about them using the available image tools.
"""

INSTRUCTION_PROMPT = """
I've loaded an image for you to analyze. Consider the following question about the image:

<question>
{pr_description}
</question>

Can you help me analyze this image and answer the question?

Your task is to thoroughly investigate the image to provide the most accurate answer to the question.

Follow these steps to analyze the image:
1. Review the list of images in the workspace (provided automatically with every message)
2. Select the image to get basic information about it (size, format, etc.)
3. Create views (crops) of specific regions of interest in the image
4. Analyze each view in detail and describe what you see
5. Mark regions as analyzed by blacking them out when you're done with them
6. Provide an accurate answer to the question based on your analysis

Available Image Tools:
- select_image: Select an image from the "Available images" list shown at the beginning of each message
- crop_image: Create a new view by cropping the currently displayed image with a bounding box [x1, y1, x2, y2] in normalized coordinates [0-1000]
- blackout_image: Black out a view to mark it as analyzed

TIPS:
- Be methodical in your analysis, examining the image section by section
- Create multiple views to focus on different parts of the image
- Be detailed in your descriptions of what you see in each view
- But only report information relavent to the question
- Consider the context and purpose of the image when answering the question
- Zoom in by repeatedly cropping, until you have a very clear view of the region of interest
- If the image contains text, make sure to read and report it accurately
- If you are done with an image, black it out before selecting another one. This will help you keep track of which parts of the image you have already analyzed.
- When you black out a view, it is automatically deleted along with any other fully black views
- The smallest remaining view (with the least number of pixels) is identified for further analysis
- DO NOT repeatedly select the same image - select it once, then use crop_image to analyze specific regions
- After selecting an image, move on to creating views with crop_image to analyze specific parts

Example workflow:
1. First select an image: `select_image(image_path="37_3.png")`
2. Then crop a region: `crop_image(bbox=[0, 0, 500, 500])` (top-left quarter)
3. Analyze what you see in the cropped view
4. Create more crops as needed: `crop_image(bbox=[500, 0, 1000, 500])` (top-right quarter)
5. When done with your analysis, use the complete tool to submit your final answer
"""
