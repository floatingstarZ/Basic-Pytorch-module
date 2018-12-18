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

image_folder = './catvsdog/train'
label_file_path = './catvsdog/labels.txt'
log_file_path = './model/log.txt'
info_file_path = './model/info.info'
# 建立当前训练的模型的文件夹（精确到天）

# 建立模型文件夹
model_folder = sl.ModelFloder()
# 建立model_built_file，保存建立的信息，现在只有分割验证集的顺序需要保存
log = sl.LogFile(log_file_path)
info = sl.InfoFile(info_file_path)

TRAIN_SIZE = 7000
VALIDATE_SIZE = 1000
BATCH_SIZE = 200
IMG_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(IMG_SIZE),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
    #transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # R,G,B每层的归一化用到的均值和方差
])
# 分割训练集与验证集，如果已经存在了初始化文件里，则读出来
train_index = []
validata_index = []
if info.data != None:
    [train_index, validata_index] = info.data
else:
    train_index = list(np.random.choice(range(TRAIN_SIZE + VALIDATE_SIZE), TRAIN_SIZE, replace=False))
    validata_index = [x for x in range(TRAIN_SIZE + VALIDATE_SIZE) if x not in train_index]
    info.dump([train_index, validata_index])

# 设置Dataset， 用于loader
train_data = data_loader.DefaultDataset(label_file_path,
                                image_folder, transform=image_transform, load_index=train_index)
validata_data = data_loader.DefaultDataset(label_file_path,
                                image_folder, transform=image_transform, load_index=validata_index)
# 设置loader， batch为100，打乱
train_loader = data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
validate_loader = data.DataLoader(validata_data, int(VALIDATE_SIZE / 2), shuffle=True)
print('Data load Success')

# 打开记录文档，记录训练过程

EPOCH = 100
PRE_EPOCH = model_folder.epoch # 之前的epoch
loss_list = []
premodel = model_folder.load_model()

# 调用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('USE cuda')
else:
    print('USE CPU')

if premodel:
    basic_model = premodel
    print('load premodel')

basic_model = basic_model.to(device)
# for params in list(basic_model.parameters()):
#     print(params.device)

#torch.nn.Module.cuda(basic_model, device=device)
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
        del correct
        del acc
        model_folder.save_model( )
    # correct = 0


for epoch in range(PRE_EPOCH, EPOCH):
    epoch_loss: float = 0
    log.write('epoch: %d\n' % epoch)
    print('epoch: %d' % epoch)
    for [iter, data] in enumerate(train_loader):
        [inputs, labels] = data
        #inputs, labels = inputs.float(), labels.long()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = basic_model(inputs)

        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()

        print('iter: %d, loss: %f' % (iter, loss))
        loss_list.append(loss)
        log.write('iter: %d, loss: %f\n' % (iter, loss))
        epoch_loss = float(epoch_loss + loss)
        #cal_acc(basic_model, inputs, labels)
        del outputs
        del loss
        del inputs

    if epoch % 2  == 0:
        for validate in validate_loader:
            [inputs, labels] = validate
            inputs, labels = inputs.to(device), labels.to(device)
            cal_acc(basic_model, inputs, labels)
            del inputs
            del labels
        # correct = 0
        # labels = []
        # predicts = []
        # real_output = []
        # count = 0
        # for data in validata_data:
        #     [inputs, label] = data
        #     inputs = inputs.reshape(1, 3, 224, 224).to(device)
        #     inputs = inputs.to(device)
        #     outputs = basic_model(inputs)
        #     # predict为每一行最大的值得下标
        #     values, predict = torch.max(outputs, 1)
        #
        #     labels.append(label)
        #     predicts.append(predict.tolist()[0])
        #     real_output.append(outputs.tolist()[0])
        #     #print('count %d, predict : %d, label : %d' % (count, predict, label))
        #     if label == predict:
        #         correct = correct + 1
        #     del outputs
        # acc = float(correct) / float(len(validata_data))
        # print('acc %f' % acc)
        # # print('labels', labels)
        # # print('predicts', predicts)
        # # print('output', real_output)
        # log.write('acc: %f\n' % acc)
        # log.write('epoch_loss: %d\n\n' % epoch_loss)
        # model_folder.save_model(basic_model)





















