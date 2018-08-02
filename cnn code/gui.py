from tkinter.filedialog import *
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
from face_util import drawFace, getFaceImg
from data_preprocess import  imgPreprocess



class Application(Frame):
    def __init__(self, master = None, model = None, image = None):
        Frame.__init__(self, master)
        self.grid()
        self.master = master
        self.tk_img = image
        self.model = model

        self.cv_img = None

        self.create_widgets()

    def create_widgets(self):

        #为窗口的关闭按钮绑定”quit“事件
        self.master.protocol("WM_DELETE_WINDOW", self.quit)

        self.imageLabel = Label(self, text = "Image", image = self.tk_img,relief=RIDGE,
                                 width = 600, height = 600)
        self.videoLabel = Label(self, text = "Video", image = self.tk_img, relief=RIDGE,
                                 width = 600, height = 600)

        self.select_button = Button(self,text="Select", fg = "green", command=self.openfile,
                                    relief=RIDGE, width = 12)

        self.recognition_button = Button(self,text = "Recognition",fg = "blue",
                                         command = self.recognition, relief=RIDGE,
                                         width = 12)

        self.play_button = Button(self,text = "Play", fg = "orange",relief=RIDGE, width = 12,
                                  command = self.video_recognition)
        self.quit_button = Button(self,text = "quit", fg = "red",relief=RIDGE, width = 12,
                                  command = self.quit)

        self.imageLabel.grid(row = 0, column = 0, padx = 15)
        self.select_button.grid(row=1, column=0, sticky = W, padx = 250, pady = 5)
        self.recognition_button.grid(row=2, column=0, sticky = W,padx = 250,pady = 5)

        self.videoLabel.grid(row = 0, column = 1, padx = 15)
        self.play_button.grid(row = 1, column = 1, sticky = W,padx = 250, pady = 5)
        self.quit_button.grid(row = 2,column = 1, sticky = W, padx = 250, pady = 5)

        self.select_button.bind("<Enter>",self.buttonEnterStyle)
        self.select_button.bind("<Leave>", self.buttonLeaveStyle)
        self.recognition_button.bind("<Enter>", self.buttonEnterStyle)
        self.recognition_button.bind("<Leave>", self.buttonLeaveStyle)
        self.play_button.bind("<Enter>", self.buttonEnterStyle)
        self.play_button.bind("<Leave>", self.buttonLeaveStyle)
        self.quit_button.bind("<Enter>", self.buttonEnterStyle)
        self.quit_button.bind("<Leave>", self.buttonLeaveStyle)

    def buttonEnterStyle(self, event):
        #鼠标移入button区域时改变style

        fg = event.widget["fg"]
        event.widget.configure(fg=event.widget["bg"])
        event.widget.configure(bg=fg)

    def buttonLeaveStyle(self, event):
        #鼠标移出button区域时改变style

        fg = event.widget["fg"]
        event.widget.configure(fg=event.widget["bg"])
        event.widget.configure(bg=fg)


    def arrayTotkImg(self, img):
        '''
        将图片由数组存储格式转为tkinter显示的格式
        :param img: 所传入的需要转化的图片
        :return: 转化后的适合tkinter显示格式的图片
        '''
        img = Image.fromarray(img)
        tkImg = ImageTk.PhotoImage(img)
        return tkImg

    def recognition(self):
        #进行人脸检测和性别识别

        cv_img = self.cv_img.copy()
        gray_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
        faces = getFaceImg(gray_img)
        faces = imgPreprocess(faces)
        text_list = []
        for i in faces:
            p_label = self.model.predict_label(i.reshape((-1, 80, 80, 1)))
            if np.argmax(p_label) == 0:
                text = "female"
            else:
                text = "male"
            text_list.append(text)
        if len(faces) != 0:
            # 绘制出人脸
            face = drawFace(self.cv_img, text_list)

            #图像转化为方便显示的格式
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            self.tk_img = self.arrayTotkImg(face)
            self.imageLabel.configure(image=self.tk_img)

        else:
            messagebox.showinfo(title = "problem", message = "No faces were detected!")

    def openfile(self, event=None):

        #获得选中文件的路径
        file_path = askopenfilename()
        print(file_path)
        self.cv_img = cv.imread(file_path)
        cv_img = self.cv_img.copy()
        #将选中的图片通过cv读出（BGR模式），再转为RGB模式
        img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        self.tk_img = self.arrayTotkImg(img)

        self.imageLabel.configure(image=self.tk_img)

    def quit(self):
        capture = cv.VideoCapture(0)
        if capture.isOpened():
            capture.release()
        self.master.destroy()

    def play_video(self):
        pass

    def video_recognition(self):
        #打开笔记本自带摄像头
        capture = cv.VideoCapture(0)

        while capture.isOpened():

            # 读取每一帧画面
            ok, frame = capture.read()
            if not ok:
                break

            #视频帧预处理
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = getFaceImg(gray_frame)
            faces = imgPreprocess(faces)

            text_list = []
            for i in faces:
                p_label = self.model.predict_label(i.reshape((-1, 80, 80, 1)))
                prediction = self.model.predict(i.reshape((-1, 80, 80, 1)))
                print(prediction)
                if np.argmax(p_label) == 0:
                    text = "female"
                else:
                    text = "male"
                text_list.append(text)
                # cv.imshow("gray_face", i)
            if len(faces) == 0:
                    frame = self.arrayTotkImg(frame)
                    self.videoLabel.configure(image = frame)
            else:
                # 绘制出人脸
                face = drawFace(frame, text_list)
                face = self.arrayTotkImg(face)
                self.videoLabel.configure(image = face)

            self.videoLabel.update()

        capture.release()

if __name__ == "__main__":
    root = Tk()
    root.title("Gender Recognition")
    # root.geometry("1280x720")
    img1 = Image.fromarray(cv.resize(cv.imread("D:\\g1.jpg", 0), (600, 600)))
    img2 = Image.fromarray(cv.resize(cv.imread("D:\\g2.jpg", 0), (600, 600)))

    img1 = ImageTk.PhotoImage(img1)
    img2 = ImageTk.PhotoImage(img2)
    app = Application(master = root, image = img1)

    root.mainloop()