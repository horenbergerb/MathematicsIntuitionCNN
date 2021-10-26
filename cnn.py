import torch
import torch.nn as nn
import torch.nn.functional as F


# input image is 160x120

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.layer5 = nn.Sequential(
            nn.Upsample(scale_factor=2)
            )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 1, 5),
            nn.ReLU(),
            nn.BatchNorm2d(1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        final_shape = x.shape
        x = x.view(1, -1)
        x = self.softmax(x)
        x = x.reshape(final_shape)
        return x
