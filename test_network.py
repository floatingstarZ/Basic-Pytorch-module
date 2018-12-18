from torch import nn
from torch import optim
from torchvision import models
import os
import re
from torch.utils import data
from PIL import Image
from torchvision import transforms
import PIL as pil
import numpy as np
import file_save_load as sl
import data_loader
from build_network import *
import torch
import csv


image_folder = './catvsdog/test'
label_file_path = './catvsdog/labels.txt'
log_file_path = './model/log_test.txt'
info_file_path = './model/info.info'
test_result_path = './model/test_result.csv'
# 建立当前训练的模型的文件夹（精确到天）

# 建立模型文件夹
model_folder = sl.ModelFloder()
# 建立model_built_file，保存建立的信息，现在只有分割验证集的顺序需要保存
log = sl.LogFile(log_file_path)
info = sl.InfoFile(info_file_path)

TEST_SIZE = 2000
BATCH_SIZE = 200
IMG_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(IMG_SIZE),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
    #transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # R,G,B每层的归一化用到的均值和方差
])

# 设置Dataset， 用于loader
test_data = data_loader.DefaultDataset('', image_folder,
                                       transform=image_transform, load_index=range(TEST_SIZE))
# 设置loader， batch为100，打乱
test_loader = data.DataLoader(test_data, BATCH_SIZE, shuffle=False)
print('Data load Success')

# 载入模型
test_model = model_folder.load_model()

# 调用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('USE cuda')
else:
    print('USE CPU')

test_model = test_model.to(device)
result = []
for [iter, data] in enumerate(test_loader):
    with torch.no_grad():
        [name, data] = data
        inputs = data.to(device)
        outputs = test_model(inputs)
        _, predicts = torch.max(outputs, 1)
        for i, predict in enumerate(predicts):
            index_in_name = int(re.findall('\d+', name[i])[0])
            if float(predict) == 0:
                cate = 'Cat'
            else:
                cate = 'Dog'
            result.append([index_in_name, cate])
            log.write(name[i] +'    ' + str(float(predict)) + '\n')
        print('iter %d DONE' % iter)
        del predicts
        del inputs
        del outputs

result.sort()
with open(test_result_path, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for row in result:
        writer.writerow(row)

def cal_acc(basic_model, inputs, labels):
    correct = 0
    predicts = []
    real_output = []
    count = 0
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = basic_model(inputs)
        # predict为每一行最大的值得下标
        _, predicts = torch.max(outputs, 1)
        # print('count %d, predict : %d, label : %d' % (count, predict, label))
        correct += (predicts == labels).sum()
        acc = float(correct) / float(len(labels))
        print('acc %f' % acc)
        #print('labels', labels)
        #print('predicts', predicts)
        # print('output', real_output)
        log.write('acc: %f\n' % acc)
        log.write('epoch_loss: %d\n\n' % epoch_loss)
        del inputs
        del outputs
        del predicts
        correct
        del acc
        model_folder.save_model(basic_model)
    return float(correct)
