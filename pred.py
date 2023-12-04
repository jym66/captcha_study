from main import Net
import string
import torch.utils.data
import DataSet
from setting import *

# 加载数据集
train_loader = DataSet.load_test_data()
char_list = list(string.ascii_letters + string.digits)
model = Net()
model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))
model.eval()
correct = 0
total = 0
for inputs, labels in train_loader:
    output = model(inputs)
    output = output.view(-1, num_classes)  # 假设每个输出有 62 个类别
    labels = labels.view(-1, num_classes)
    _, predicted = torch.max(output, 1)
    _, labels = torch.max(labels, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = correct / total
print("Accuracy: {:.2f}%".format(accuracy * 100))
