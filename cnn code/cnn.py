from tflearn.models.dnn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from data_preprocess import one_hot
from gui import *




def createCnn(learning_rate):
    '''
        创建卷积神经网络
        Args:
            learning_rate:学习率

        Returns:
            network: tensor类型
    '''
    #输入层
    network = input_data(shape=[None, 80, 80, 1], name = "Input")

    #卷积层1
    network = conv_2d(network, 32, 5, activation='relu', name="Conv2D_1")

    #池化层1
    network = max_pool_2d(network, 2, name="MaxPool2D_1")

    #卷积层2
    network = conv_2d(network, 64, 3, activation='relu', name="Conv2D_2")

    # 卷积层3
    network = conv_2d(network, 64, 3, activation='relu', name="Conv2D_3")

    #池化层2
    network = max_pool_2d(network, 2, name="MaxPool2D_2")

    #全连接层
    network = fully_connected(network, 512, activation='relu', name="FullyConnected_1")

    #drop层，防止过拟合，0.8表示保持80%的neurals激活
    network = dropout(network, 0.8, name="DropOut")

    #全连接层(输出)
    network = fully_connected(network, 2, activation='softmax',  name="Output")

    #回归操作
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate = learning_rate)
    return network


def readCSV(filePath, separator = ';'):
    '''
        读取指定文件，该文件每一行指明了图片的路径和该人脸图像的性别

        Args:
            filePath: csv文件所在路径
            separator: 图像路径信息和性别信息的分隔符

        Returns:
            返回读取到的图像列表和对应的标签列表
    '''
    face_list = []
    label_list = []
    with open(filePath) as file:
        for line in file:
            line_list = line.rstrip('\n').split(separator)
            img_path = line_list[0]
            if line_list[1] == 'M':
                labelnum = 1
            else:
                labelnum = 0
            face_list.append(cv.imread(img_path, 0))
            label_list.append(labelnum)

    return face_list, label_list



def test(test_faces, test_labels, cnn_model):
    '''
        测试模型

        Args:
            test_faces: 测试图像的图片列表
            test_labels: 测试图像的标签
            cnn_model：cnn模型
        Returns:
            识别正确率，出错图像的下标
    '''
    count = 0
    match_count = 0
    accuracy = 0.0
    wrong_index = []
    for i in test_faces:
        p_label = cnn_model.predict_label(i.reshape((-1, 80, 80, 1)))
        if np.argmax(p_label) == test_labels[count]:
            match_count += 1
        else:
            wrong_index.append(count)
        count += 1

    accuracy = match_count / count
    return  accuracy, wrong_index

if __name__ == "__main__":

    root = Tk()
    root.title("Gender Recognition")
    # root.geometry("1280x720")

    ##从训练集中读取数据
    #csvPath = "H:\\FaceImageDataset\\FERET_80_80\\gendercsv.txt"
    #face_list, label_list = readCSV(csvPath)

    ##对即将输入网络的数据进行预处理
    #face_array = imgPreprocess(face_list)
    #face_array = face_array.reshape((-1, 80, 80, 1))
    #label_array = one_hot(label_list)

    #搭建卷积神经网络
    network = createCnn(0.0008)
    data_save_path = "H:\\FaceImageDataset\\FERET_80_80\\gender_classfier_data_0.0008"
    tensorboard_dir = data_save_path + "\\tensorboard_data"
    cnn_model = DNN(network, tensorboard_verbose=3,
                    tensorboard_dir = tensorboard_dir,
                    checkpoint_path=tensorboard_dir + "\\gender_classfier.tfl.ckpt")

    # #训练
    # cnn_model.fit(face_array, label_array, n_epoch= 60,
    #               validation_set=0.2,
    #               show_metric=True, batch_size=200, shuffle=True,
    #               snapshot_epoch=False,
    #               run_id="gender_classfier")
    #
    # cnn_model.save(data_save_path + "\\gender_classfier.tfl")
    # print("Network trained and saved as gender_classfier.tfl!")

    #载入模型
    cnn_model.load(data_save_path + "\\gender_classfier.tfl")


    cv_img = cv.imread("D:\\img_bg.jpg")
    img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    tk_img = ImageTk.PhotoImage(img)

    app = Application(master=root, model=cnn_model, image = tk_img)
    root.mainloop()