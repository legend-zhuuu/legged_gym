import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
from torchsummary import summary
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

BATCH_SIZE = 64
LR = 0.0005  # learning rate
EPOCH = 10

# 模型加载选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


def plot_images(ids, input, output):
    ori_data = input.numpy()
    i = 1
    for idx in ids:
        plt.subplot(2, len(ids), i)
        plt.imshow(ori_data[idx].reshape(240, 360), cmap='gray')
        plt.subplot(2, len(ids), len(ids) + i)
        de_data = output[idx].reshape(240, 360).numpy()
        plt.imshow(de_data, cmap='gray')
        i += 1
    plt.show()


def load_depth_image():
    images_name = os.listdir('graphics_images_train')
    images = []
    for image_name in images_name:
        file_name = 'graphics_images_train/' + image_name
        img = plt.imread(file_name)
        images.append(img)
    images = np.array(images)
    images = torch.tensor(images)
    return images


train_data = load_depth_image()
print("load all depth images")
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class Denoise_AutoEncoders(nn.Module):
    def __init__(self):
        super(Denoise_AutoEncoders, self).__init__()
        # 定义Encoder编码层
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # [, 4, 240, 360]
            nn.ReLU(True),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, stride=1, padding=1),  # [, 4, 120, 180]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=5, stride=5),  # [, 4, 60, 90]  # add more one pool layer
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),  # [, 4, 60, 90]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=3),  # [, 4, 30, 45]
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # [, 4, 30, 45]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # [, 4, 15, 23]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [, 4, 8, 12]  # add more one pool layer
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # [, 4, 8, 12]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [, 8, 4, 6]
            nn.BatchNorm2d(8),
        )
        # 定义Decoder解码层，使用转置卷积
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # [, 4, 4, 6]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),  # [, 4, 16, 24]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1),  # [, 4, 16, 24]
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),  # [, 4, 32, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),  # [, 4, 32, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, 3, stride=1, padding=1),  # [, 4, 64, 96]
            nn.ConvTranspose2d(4, 4, 4, stride=3, padding=1, output_padding=1),  # [, 4, 128, 192]
            nn.ReLU(True),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 1, 3, stride=1, padding=1),  # [, 4, 128, 192]
            nn.ConvTranspose2d(1, 1, 6, stride=5, padding=1, output_padding=1),  # [, 4, 256, 384]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder


de_autoencoder = Denoise_AutoEncoders().to(device)
summary(de_autoencoder, input_size=(1, 240, 360))
optimizer = torch.optim.Adam(de_autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss().to(device)


for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
        # x torch.Tensor
        b_x = x.view(-1, 1, 240, 360).type(torch.FloatTensor).to(device) / 255.  # (BATCH_SIZE,28, 28)
        b_y = x.view(-1, 1, 240, 360).type(torch.FloatTensor).to(device) / 255.

        encoded, decoded = de_autoencoder(b_x)

        loss = loss_func(decoded, b_y)  # MSE
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # BP，计算梯度
        optimizer.step()  # 利用梯度进行更新

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())

    view_data = train_data[:200].view(-1, 1, 240, 360).type(torch.FloatTensor) / 255.
    encoded_data, decoded_data = de_autoencoder(view_data.to(device))
    idx = 1
    plt.subplot(2, EPOCH, epoch + 1)
    ori_data = train_data[idx].numpy()
    plt.imshow(ori_data, cmap='gray')
    plt.subplot(2, EPOCH, EPOCH + epoch + 1)
    de_data = decoded_data[idx].view(240, 360).detach().cpu().numpy()
    plt.imshow(de_data, cmap='gray')
torch.save(de_autoencoder.state_dict(), "DEA_depth.pt")
plt.show()
