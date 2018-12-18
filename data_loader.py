import os
import re
from torch.utils import data
from PIL import Image
from torchvision import transforms
import PIL as pil

#这是另外一种方法
from torchvision.datasets import ImageFolder

class DefaultDataset(data.Dataset):
    # load_index 需要加载的样本下标
    # image_folder: 图片文件夹
    # transform: transforms类型的变量，用transforms.Compose定义
    def __init__(self, label_file_path, image_folder,
                 transform: transforms=None, load_index=None):
        super(data.Dataset, self).__init__()
        self.load_index = load_index
        self.transform = transform
        # 加载图像，标签
        self.label_loader = LabelLoader(label_file_path, load_index)
        self.image_loader = ImageLoader(image_folder, load_index)
        if len(self.image_loader) != len(self.label_loader):
            raise Exception('Number of label images does not match')

    def __getitem__(self, index):
        img = self.image_loader[index]
        label = self.label_loader[index]
        if self.transform:
            img = self.transform(img)
        return [img, label]

    def __len__(self):
        return len(self.label_loader)

class BasicLoader(object):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise Exception('%s FILE does not Exist', file_path)
        self.data = None
        self.path = file_path

    def __getitem__(self, index):
        raise NotImplementedError

class LabelLoader(BasicLoader):
    def __init__(self, label_file_path, load_index=None):
        super(LabelLoader, self).__init__(label_file_path)
        self.file = open(label_file_path, 'r')
        self.format = '\d+ \d+'     # for check
        self.labels = []
        counter = 0
        for line in self.file.readlines():
            self.labels.append(int(line[:-1].split(' ')[1]))        # 获得切分后第一个元素为label
        if load_index:
            try:
                self.labels = [self.labels[i] for i in load_index]
            except IndexError:
                raise Exception('load_index in LabelLoader out of range')
        self.file.close()

    def check_format(self) -> bool:
        self.file = open(self.path, 'r')
        for line in self.file.readlines():
            if re.findall(self.format, line):
                raise Exception('Label File format wrong')
        return True

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def __add__(self, other):
        return


# load_index 表示的是截取的下标（为了产生训练集与验证集），使用range表示，如range(1000)
class ImageLoader(BasicLoader):
    def __init__(self, image_folder, load_index=None):
        super(ImageLoader, self).__init__(image_folder)
        # self.format = name_formats  # such as ['\'cat.%d\'i', '']
        # os.splittext可以用来且分出图片归属于哪一个类
        # 可以对文件名进行了排序，但是要小心label是否还与之对应

        self.file_names: list = sorted(os.listdir(image_folder))
        if load_index:
            try:
                self.file_names = [self.file_names[i] for i in load_index]
            except IndexError:
                raise Exception('load_index in ImageLoader out of range')

    def __getitem__(self, index):
        img_name = os.path.join(self.path, self.file_names[index])
        # 可以在这里添加Transform
        # transforms.Compose([
        #     transforms.CenterCrop(10),
        #     transforms.ToTensor(),
        # ])
        img = pil.Image.open(img_name)
        # transforms.ToTensor()(img) # tensor转变
        return img

    def __len__(self):
        return len(self.file_names)

def form_label_file(image_folder, label_file_path):
    lable_file = open(label_file_path, 'wt+')
    file_names: list = sorted(os.listdir(image_folder))
    for name in file_names:
        lable_file.write(name + ' ')
        if 'cat' in name:
            lable_file.write(str(0))
        else:
            lable_file.write(str(1))
        lable_file.write('\n')
    lable_file.close()


transform_train = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(224),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

if __name__ == '__main__':
    image_folder = './catvsdog/train'
    label_file_path = './catvsdog/labels.txt'
    form_label_file(image_folder, label_file_path)
    train_data = DefaultDataset(label_file_path,
                                image_folder, transform_train)
    loader = data.DataLoader(train_data, 100, True)






