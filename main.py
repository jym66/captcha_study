import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import DataSet

# 加载数据集
train_loader = DataSet.load_train_data()


class Net(nn.Module):
    def __init__(self, num_classes=62):
        super(Net, self).__init__()
        # 使用预定义的ResNet，并修改最后一层
        self.resnet = models.resnet34(pretrained=True)  # 选择ResNet的版本，例如resnet18
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4 * num_classes)
        # 添加额外的层或修改现有层
        # 例如，增加一个Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 通过ResNet模型
        x = self.resnet(x)
        # 应用Dropout（如果需要）
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 实例化模型并移至 GPU（如果可用）
    model = Net().to(device)
    # 定义损失函数和优化器
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # 将数据和标签移至 GPU（如果可用）
            inputs, labels = inputs.to(device), labels.to(device)
            # 将标签转换为浮点数
            labels = labels.float().squeeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
