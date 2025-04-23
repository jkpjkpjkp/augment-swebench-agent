# Visual Question Answering (VQA) Extension

This extension repurposes the SWE-bench agent for Visual Question Answering (VQA) tasks. It allows you to analyze images by creating and manipulating views (crops) of the original images.

## Features

- **Image Management**: Add images to the workspace and manage them
- **View Creation**: Create views (crops) of images for detailed analysis
- **Synchronized Editing**: Changes to one view are reflected in all other views and the original image
- **Analysis Tools**: Tools for selecting and marking analyzed regions

## Image Tools

The following tools are available for working with images:

### 1. Add Image Tool (`add_image`)

Add a new image to the workspace for analysis.

```
{
  "image_path": "path/to/image.jpg",
  "image_name": "optional_custom_name.jpg"
}
```

### 2. Crop Tool (`crop_image`)

Create a new view by cropping an image with 4 coordinates.

```
{
  "image_path": "images/example.jpg",
  "x1": 100,
  "y1": 100,
  "x2": 300,
  "y2": 300,
  "view_id": "optional_view_id"
}
```

### 3. Select Tool (`select_image`)

Select an entire image or view for further processing.

```
{
  "image_path": "views/example__view1__100_100_300_300.jpg"
}
```

### 4. Blackout Tool (`blackout_image`)

Black out an image or view, marking it as analyzed.

```
{
  "image_path": "views/example__view1__100_100_300_300.jpg"
}
```

### Note on Image Listing

A list of all images and views in the workspace is automatically included in every message, so there's no need for a separate tool to list images.

## Workflow Example

1. Add an image to the workspace:
   ```
   add_image(image_path="path/to/image.jpg")
   ```

2. Create a view (crop) of the image:
   ```
   crop_image(image_path="images/image.jpg", x1=100, y1=100, x2=300, y2=300)
   ```

3. Analyze the view and when done, mark it as analyzed:
   ```
   blackout_image(image_path="views/image__view1__100_100_300_300.jpg")
   ```

4. Create another view and continue analysis:
   ```
   crop_image(image_path="images/image.jpg", x1=300, y1=100, x2=500, y2=300)
   ```

## Implementation Details

- Images are stored in the `images` directory in the workspace
- Views are stored in the `views` directory with filenames that encode their relationship to the original image
- The `ImageManager` class handles the relationships between images and their views
- Changes to one view are propagated to all overlapping views and the original image
