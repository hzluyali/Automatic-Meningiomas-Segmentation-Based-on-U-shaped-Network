# Automatic-Meningiomas-Segmentation-Based-on-U-shaped-Network

## 01 文件功能

generate_database.py   #将nii文件处理成TCGA-LGG公开数据集格式

br.py  #背景去除

data.py #数据读取文件

seg_loss.py #存放损失函数

encoder文件夹中存放编码网络模型

decoder9.py #解码网络，可修改其中第266行中的第二个参数，以选择编码网络

train.py #训练文件，修改26-28行分别修改训练文件路径，测试文件路径以及初始权重

test.py #生成测试集的预测结果，并且输出Dice，敏感度和IOU这三项指标，需修改第24行的权重路径

predict.py #在线预测

## 02 运行过程

Step 1: 运行python generate_database.py将nii文件转化为TCGA-LGG公开数据集格式

Step 2: 运行python br.py进行颅骨去除从而提高肿瘤区域的比例

Step 3: 运行train.py训练模型

Step 4: 运行test.py评估模型在测试集上的泛化能力，并通过predict.py进行在线测试，实验结果如下所示：



