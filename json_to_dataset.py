import base64
import json
import os
import os.path as osp
import time

import numpy as np
import PIL.Image
from labelme import utils
import subprocess


'''
制作自己的语义分割数据集需要注意以下几点：
1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误，
   具体错误为：Too many dimensions: 3 > 2
   安装方式为命令行pip install labelme==3.16.7
2、此处生成的标签图是8位彩色图，与视频中看起来的数据集格式不太一样。
   虽然看起来是彩图，但事实上只有8位，此时每个像素点的值就是这个像素点所属的种类。
   所以其实和视频中VOC数据集的格式一样。因此这样制作出来的数据集是可以正常使用的。也是正常的。
'''
if __name__ == '__main__':
    jpgs_path   = "VOCdevkit/VOC2007/JPEGImages"
    pngs_path   = "VOCdevkit/VOC2007/SegmentationClass"
    classes = ["_background_", "bait", "distraction"]
    
    count = os.listdir("./datasets/before/") 
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            try:
                img = img[:, :, :3]  # rgba to rgb
            except IndexError:
                # Handle the IndexError here
                # For example, print an error message or perform an alternative action
                print("Error: Image array does not have a third dimension.")
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')



    # try:
    #     # Execute the Python script using subprocess
    #     subprocess.run(['python', 'utils/convert_to_512.py'], check=True)
    #     print("convert to 512 executed successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing the script: {e}")
    # time.sleep(10)
    try:
        # Execute the Python script using subprocess
        subprocess.run(['python', 'utils/voc_annotation.py'], check=True)
        print("voc_annotate executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")