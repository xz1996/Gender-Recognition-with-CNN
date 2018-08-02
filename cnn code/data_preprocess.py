import numpy as np
import cv2 as cv


def imgPreprocess(imgsList):
    '''
    将输入神经网络的图片进行预处理

    Args:
        imgsList: 所需处理的图片列表

    Returns:
        处理后的图片数组
    '''

    img_arraylist = []
    for i in imgsList:
        # print("face_array:", face_array)
        # print("face_array_shape:",face_array.shape)
        new_img = cv.resize(i,(80,80))
        face_array = cv.equalizeHist(new_img)  # 直方图均衡化
        face_array = face_array.astype(np.float32)
        face_array = np.multiply(face_array, 1.0 / 255.0)   # 将所有像素转为0~1之间的小数
        img_arraylist.append(face_array)
    return np.array(img_arraylist)


def one_hot(label_list):
    '''
    将输入神经网络的图片的”标签“包装成合适的数组

    Args:
        label_list: 所需处理的”标签“列表

    Returns:
        处理后的标签数组
    '''

    label_arraylist = []
    for i in label_list:
        if i == 0:
            label_arraylist.append(np.array([1, 0]))
        else:
            label_arraylist.append(np.array([0, 1]))

    return np.array(label_arraylist)