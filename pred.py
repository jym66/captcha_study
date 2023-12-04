from main import Net
import string
import torch.utils.data
import DataSet
# 加载数据集
train_loader = DataSet.load_test_data()

char_list = list(string.ascii_letters + string.digits)
model = Net()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()
correct = 0
for inputs, labels in train_loader:
    output = model(inputs)
    output = output.view(-1, 62)
    labels = labels.view(-1, 62)
    labels = ''.join([char_list[i] for i in torch.argmax(labels, 1)])
    output = ''.join([char_list[i] for i in torch.argmax(output, 1)])
    if output == labels:
        correct += 1
    # print(output, labels)
print(correct)
