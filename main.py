import string

import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import DataSet


# 加载数据集
train_loader = DataSet.load_train_data()


class Net(nn.Module):
    def __init__(self, num_classes=62):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * 7 * 7, out_features=4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4 * num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


if __name__ == '__main__':
    # char_list = list(string.ascii_letters + string.digits)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 实例化模型并移至 GPU（如果可用）
    model = Net().to(device)
    # 定义损失函数和优化器
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # 将数据和标签移至 GPU（如果可用）
            inputs, labels = inputs.to(device), labels.to(device)
            # 将标签转换为浮点数
            labels = labels.float().squeeze(1)
            outputs = model(inputs)
            # outputs = outputs.view(-1, 62)
            # print("labels shape", labels.shape)
            # print("outputs shape", outputs.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        if num_epochs % 5 == 0:
            print(f"保存模型 mnist_cnn{loss.item():.4f}")
            torch.save(model.state_dict(), f'mnist_cnn{loss.item():.4f}.pth')

    # 保存模型
    torch.save(model.state_dict(), 'mnist_cnn.pth')
