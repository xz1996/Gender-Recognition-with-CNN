import cv2 as cv
import threading
from  threading import Thread


class DetectFaces(Thread):
    __coord = []
    def __init__(self, img, threadID = threading.active_count(),
                 name = "detect_thread" + str(threading.active_count())):
        Thread.__init__(self)
        self.img = img
        self.threadID = threadID
        self.name = name

    def run(self):
        print(self.name + " is starting...")
        self.__coord = self.detectFaces(self.img)
        print(self.name + " is over...")

    def get_detect_faces(self):
        return self.__coord


    def detectFaces(self, time_out = None):
        '''
        检测所给图片的人脸
        Args:
            img:需要检测的图像(尺寸最小为30x30)，类型为numpy.ndarray

        Returns:
            返回人脸所在区域的坐标，类型为list
        '''

        # 人脸检测模型路径
        lbp_cascade_path = "G:\\OpenCV\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml"
        haar__cascade_path = "G:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"

        # 加载人脸检测模型
        face_cascade = cv.CascadeClassifier(haar__cascade_path)

        # img为源图，1.15表示每次搜索人脸时搜索窗口扩大15%
        face = face_cascade.detectMultiScale(self.img, 1.15, 4)

        # # 如果未检测到人脸，则将参数调小再检测（主要针对人脸占据大部分区域的图像）
        # if len(face) == 0:
        #     face = face_cascade.detectMultiScale(self.img, 1.0005, 3)

        # result列表用来保存检测到的人脸的左上角和右下角坐标
        result = []
        for (x, y, w, h) in face:
            result.append((x, y, x + w, y + h))
        return result


def getFaceImg(img, newSize=(80, 80)):
    '''
    获得所给图片的人脸
    Args:
        img:需要处理的图像，类型为numpy.ndarray
        newSize:获得人脸图像后缩放的大小，默认为80x80

    Returns:
        返回该图含有的所有人脸的列表
    '''

    img_copy = img.copy()
    result = DetectFaces(img_copy).detectFaces()
    faces = []
    for (x1, y1, x2, y2) in result:
        faces.append(cv.resize(img_copy[y1:y2, x1:x2], newSize))
    return faces


def drawFace(img, text = None):
    '''
    用矩形框绘出图片的人脸并在该区域左上角输出指定文字
    
    Args:
        img:需要处理的图像，类型为numpy.ndarray
        text:要在图像上绘制的文本
    Returns:
        返回标记了人脸的图像
    '''
    img_copy = img.copy()
    result = DetectFaces(img_copy).detectFaces()
    for i in range(len(result)):
        (x1, y1) = (result[i][0], result[i][1])
        (x2, y2) = (result[i][2], result[i][3])
        cv.rectangle(img, (x1, y1), (x2, y2), (255, 255,255), 2)

        # 若是有文本，在face区域左上方输出文字
        if text and i < len(text):
            cv.putText(img, text[i], (x1, y1-5), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       1, (255, 255, 255))
    return img