# This is my bachelor graduation project

## Environment

- Python3.5

- Opencv-python

- TensorFlow1.1（安装之前请先安装numpy库）

- TFLearn0.3

Windows平台下Python各类库的安装：http://www.lfd.uci.edu/~gohlke/pythonlibs/

## How to use

若要测试本模型，请将cnn model文件夹里的文件放到合适的路径（全英文路径），再将```cnn.py```的第125行中的```data_save_path```的值改为
所存放的路径即可。

若要训练模型，将cnn.py里的第115~121行和第132~139行代码解除屏蔽。将第115行的```csvPath```的值改为你存放的csv文件的路径。
csv文件的格式为：路径;标签，如：

  H:\FaceImageDataset\FERET_80_80\FERET-003\07.tif;M

  H:\FaceImageDataset\FERET_80_80\FERET-004\01.tif;F

  其中分号前的表示图片路径，分号后的M表示男性，F表示女性。

注意，cnn code里cnn.py才是主文件，请从cnn.py开始运行

想要看ppt的话请将gender recognition slide.zip文件解压出来，然后运行里面的play.exe文件，