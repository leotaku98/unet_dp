#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#

from PIL import Image

from unet_func import Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = False
    #name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes    = ["background","dark pattern"]
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"


    unet = Unet()

    '''
    predict.py有几个注意点
    1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
    具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
    2、如果想要保存，利用r_image.save("img.jpg")即可保存。
    3、如果想要原图和分割图不混合，可以把blend参数设置成False。
    4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
    seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
    for c in range(self.num_classes):
        seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
        seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
        seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
    '''
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = unet.detect_image(image)
            r_image.show()
