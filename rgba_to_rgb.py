from PIL import Image
import os

# Directory containing the RGBA images
input_dir = 'datasets/before'
# Directory to save the converted RGB images
output_dir = 'datasets/rgb_screenshots'

# Iterate over the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        # Open the image in RGBA mode
        image_path = os.path.join(input_dir, filename)
        image_rgba = Image.open(image_path).convert('RGBA')

        # Convert the image to RGB mode
        image_rgb = image_rgba.convert('RGB')

        # Save the converted image in the output directory
        output_filename = filename
        output_path = os.path.join(output_dir, output_filename)
        image_rgb.save(output_path, format='JPEG')