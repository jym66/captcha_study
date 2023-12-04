import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet34_Weights

import DataSet
from setting import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.resnet = models.resnet34(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, captcha_len * num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    # 加载数据集
    train_loader = DataSet.load_train_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("运行在:", device)
    # 实例化模型并移至 GPU（如果可用）
    model = Net().to(device)
    # 定义损失函数和优化器
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().squeeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), save_model_path)
