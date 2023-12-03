import os
import string
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#  使用自定义的数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#
char_list = list(string.ascii_letters + string.digits)
chars_dict = {char: idx for idx, char in enumerate(char_list)}


class CustomImageDataset(Dataset):
    def __init__(self, root, trans, max_label_length=4):
        self.root = root
        self.transform = trans
        self.images = os.listdir(self.root)
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_index = self.images[item]
        image_path = os.path.join(self.root, image_index)
        # if "DS_Store" not in image_path:
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = os.path.basename(image_path).split(".")[0]
        # 处理标签
        new_label = [chars_dict.get(i, 0) for i in label]  # 使用 get 方法避免不存在的字符引发错误
        # 填充标签
        while len(new_label) < self.max_label_length:
            new_label.append(0)  # 假设 0 是一个填充值
        # 转换为张量
        new_label = torch.tensor(new_label, dtype=torch.long)

        one_hot = torch.nn.functional.one_hot(new_label, num_classes=62)
        one_hot = one_hot.view(1, -1)
        # print(label)
        # print(one_hot.shape)
        return image, one_hot


def load_train_data():
    dataset = CustomImageDataset("/Users/binary/PycharmProjects/DeepLearing/captcha", transform)

    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


def load_test_data():
    dataset = CustomImageDataset("/Users/binary/PycharmProjects/DeepLearing/captcha", transform)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
