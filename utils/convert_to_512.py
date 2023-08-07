import os
from PIL import Image


def letterbox_resize(image, size):
    width, height = image.size
    target_width, target_height = size

    # Calculate the scale factor for resizing
    scale = min(target_width / width, target_height / height)

    # Calculate the new size after maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a blank canvas of the target size with a black background
    letterboxed_image = Image.new("RGB", size, (0, 0, 0))

    # Calculate the position to paste the resized image on the canvas
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the resized image onto the canvas
    letterboxed_image.paste(resized_image, (x_offset, y_offset))

    return letterboxed_image

# Folder containing the input images
input_folders = ["VOCdevkit/VOC2007/JPEGImages", "VOCdevkit/VOC2007/SegmentationClass"]
output_folders = ["VOCdevkit/VOC2007/JPEGImages", "VOCdevkit/VOC2007/SegmentationClass"]

# Desired size for resizing
desired_size = (512, 512)

for i in range(2):
    input_folder = input_folders[i]
    output_folder = output_folders[i]
    for filename in os.listdir(input_folder):
        if "md" in filename:
            continue
        # Get the full path of the input image file
        input_path = os.path.join(input_folder, filename)

        # Open the image
        image = Image.open(input_path)


        # Resize the image
        resized_image = letterbox_resize(image, desired_size)
        if (i == 1):
            pass
            #resized_image = resized_image.convert("P", palette=Image.ADAPTIVE)  # optinal, depends on binary index

        # Get the filename without extension
        file_name_without_extension = os.path.splitext(filename)[0]

        #original image
        if (i==0):
            output_path = os.path.join(output_folder, file_name_without_extension + '.jpg')
        #label image
        else:
            resized_image = resized_image.convert("P", palette=Image.ADAPTIVE)  # optinal, depends on binary index
            output_path = os.path.join(output_folder, file_name_without_extension + '.png')
        # Save the resized image
        resized_image.save(output_path)