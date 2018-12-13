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
import pickle
import time
import data_loader
from build_network import *

image_folder = './catvsdog/train'
label_file_path = './catvsdog/labels.txt'
model_base_folder = './model'
# 建立当前训练的模型的文件夹（精确到天）
model_folder = os.path.join(model_base_folder, time.strftime('%Y-%m-%d',time.localtime(time.time())))
model_build_information_file_path = os.path.join(model_base_folder, 'model initial information.info')
MODEL_BUILD_FILE_EXIST = False

# 建立模型文件夹
if not os.path.exists(model_base_folder):
    print('Make Dir : %s' % model_base_folder)
    os.makedirs(model_base_folder)
    if not os.path.exists(model_folder):
        print('Make Dir : %s' % model_folder)
        os.makedirs(model_folder)
# 建立model_built_file，保存建立的信息，现在只有分割验证集的顺序需要保存
if not os.path.exists(model_build_information_file_path):
    print('Make file : %s' % model_build_information_file_path)
    model_built_file = open(model_build_information_file_path, 'wb+')
else:
    print('Open file : %s' % model_build_information_file_path)
    model_built_file = open(model_build_information_file_path, 'rb')   # 'a'为续写
    MODEL_BUILD_FILE_EXIST = True


TRAIN_SIZE = 7000
VALIDATE_SIZE = 1000
BATCH_SIZE = 100
IMG_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(IMG_SIZE),  # 先四周填充0，在吧图像随机裁剪成IMG_SIZE
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # R,G,B每层的归一化用到的均值和方差
])
# 分割训练集与验证集，如果已经存在了初始化文件里，则读出来
train_index = []
validata_index = []
if MODEL_BUILD_FILE_EXIST:
    [train_index, validata_index] = pickle.load(model_built_file)
    model_built_file.close()
else:
    # 无放回的抽取7000个作为训练样本，剩下的作为验证
    train_index = list(np.random.choice(range(TRAIN_SIZE), TRAIN_SIZE, replace=False))
    validata_index = [x for x in range(TRAIN_SIZE + VALIDATE_SIZE) if x not in train_index]
    pickle.dump([train_index, validata_index], model_built_file)
    model_built_file.close()

# 设置Dataset， 用于loader
train_data = data_loader.DefaultDataset(label_file_path,
                                image_folder, transform=image_transform, load_index=train_index)
validata_data = data_loader.DefaultDataset(label_file_path,
                                image_folder, transform=image_transform, load_index=validata_index)
# 设置loader， batch为100，打乱
train_loader = data.DataLoader(train_data, 100, shuffle=True)
validate_loader = data.DataLoader(train_data, VALIDATE_SIZE, shuffle=False)
a = 0

