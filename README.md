# Automatic-Meningiomas-Segmentation-Based-on-U-shaped-Network

## 01 文件功能

generate_database.py   #将nii文件处理成TCGA-LGG公开数据集格式

br.py  #背景去除

data.py #数据读取文件

seg_loss.py #存放损失函数

encoder文件夹中存放编码网络模型

decoder9.py #解码网络，可修改其中第266行中的第二个参数，以选择编码网络

train.py #训练文件，修改26-28行分别修改训练文件路径，测试文件路径以及初始权重



