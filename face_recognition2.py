import torch as t
from torch import nn
from face_recognition1 import load_image_data
import numpy as np
import torch.utils.data as Data
import os


os.chdir(r'D:\PycharmProjects\pytorch')
IMAGE_SIZE = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lin1 = nn.Linear(128 * (IMAGE_SIZE // 8) ** 2, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 3)

    def forward(self, x):
        cov1_x = self.cov1(x)
        cov2_x = self.cov2(cov1_x)
        cov3_x = self.cov3(cov2_x)
        reshape_x = cov3_x.view(cov3_x.shape[0], -1)
        lin1_x = self.lin1(reshape_x)
        lin2_x = self.lin2(lin1_x)
        output_x = self.lin3(lin2_x)
        return output_x


# x, y = load_image_data('trans_face_data')
# tensor_x = t.tensor(data=x, dtype=t.float32, requires_grad=False)
# tensor_y = t.tensor(data=y, dtype=t.int64, requires_grad=False)
# del x, y
# t.save(tensor_x, 'tensor_x.pt')
# t.save(tensor_y, 'tensor_y.pt')
tensor_x = t.load('tensor_x.pt').unsqueeze(1)
tensor_y = t.load('tensor_y.pt')
torch_dataset = Data.TensorDataset(tensor_x, tensor_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=500,
    shuffle=True,
    # num_workers=4
)
#
cnn_net = Net().cuda()
print(cnn_net)
optimizer = t.optim.Adam(cnn_net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
for epoch in range(10):
    acc_sorce = []
    for step, (step_x, step_y) in enumerate(loader):
        step_x = step_x.cuda()
        step_y = step_y.cuda()
        predict_y = cnn_net(step_x)
        loss = loss_function(predict_y, step_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with t.no_grad():
            acc_sorce.append(np.mean(t.argmax(predict_y.cpu(), dim=1).data.numpy() == step_y.cpu().data.numpy())*100)
    with t.no_grad():
        print('epoch:{}, 正确率：{:.2f}%'.format(epoch, np.mean(acc_sorce)))
t.save(cnn_net.cpu(), 'cnn_model.pkl')
# cnn_model = t.load('cnn_model.pkl')