from torch import nn
from torch import optim
from torchvision import models
ClassNum = 2

# 如果要自己定义一个ResNet，可以直接仿照resnet50的做法，使用ResNet类。
basic_model = models.resnet18(pretrained=True)
# 修改全连接层
basic_model.fc = nn.Linear(512, ClassNum)
# 定义损失函数和优化方式
# #这个损失函数是softmax和交叉熵的结合，输入是一个任意向量x和一个类别c
# 先对这个向量x，取softmax，得到x[c]的概率，之后，取负对数。
optimizer = optim.SGD(basic_model.fc.parameters(), lr=0.0001,weight_decay=5e-4)
loss_fun = nn.CrossEntropyLoss(size_average=False)
