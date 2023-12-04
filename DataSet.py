import os
import string
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from setting import *

#  使用自定义的数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#
char_list = list(string.ascii_letters + string.digits)
chars_dict = {char: idx for idx, char in enumerate(char_list)}


class CustomImageDataset(Dataset, ):
    def __init__(self, data_path, trans, max_label_length=captcha_len, ):
        self.root = data_path
        self.transform = trans
        self.images = os.listdir(self.root)
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_index = self.images[item]
        image_path = os.path.join(self.root, image_index)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = os.path.basename(image_path).split(".")[0]
        new_label = [chars_dict.get(i, 0) for i in label]
        while len(new_label) < self.max_label_length:
            new_label.append(0)  # 假设 0 是一个填充值
        new_label = torch.tensor(new_label, dtype=torch.long)

        one_hot = torch.nn.functional.one_hot(new_label, num_classes=num_classes)
        one_hot = one_hot.view(1, -1)
        return image, one_hot


def load_train_data():
    print("加载训练集")
    dataset = CustomImageDataset(train_path, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


def load_test_data():
    print("加载测试集")
    dataset = CustomImageDataset(test_path, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
