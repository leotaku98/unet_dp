import os

from PIL import Image

from unet_func import Unet
if __name__ == "__main__":
    name_classes = ["background", "dark pattern"]
    unet = Unet()
    folder_path = 'test_img'
    output_folder = 'test_img/result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            img = os.path.join(folder_path, file_name)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                r_image = unet.detect_image(image)
                r_image.save(os.path.join(output_folder, file_name))
    print('Prediction complete')