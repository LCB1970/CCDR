import torch.nn as nn
import math


class VGG(nn.Module):
    def __init__(self, layer, band):
        super(VGG, self).__init__()
        self.layer = layer
        self.conv11 = nn.Conv2d(in_channels=band, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.concat1 = nn.ReLU(inplace=True)
        self.downsample1 = nn.MaxPool2d(2, stride=2)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.concat2 = nn.ReLU(inplace=True)
        self.downsample2 = nn.MaxPool2d(2, stride=2)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.concat3 = nn.ReLU(inplace=True)
        self.downsample3 = nn.MaxPool2d(2, stride=2)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.concat4 = nn.ReLU(inplace=True)
        self.downsample4 = nn.MaxPool2d(2, stride=2)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.downsample5 = nn.MaxPool2d(2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(512 * high//32 * width//32, 1024)
        # self.fc2 = nn.Linear(1024, number_classes)

        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv11(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv12(x)
        x = self.concat1(x)
        x = self.downsample1(x)

        x = self.conv21(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv22(x)
        x = self.concat2(x)
        x = self.downsample2(x)

        x = self.conv31(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv32(x)
        x = nn.ReLU(inplace=True)(x)
        if self.layer == 19:
            x = self.conv33(x)
            x = nn.ReLU(inplace=True)(x)
        x = self.conv34(x)
        x = self.concat3(x)
        x = self.downsample3(x)

        x = self.conv41(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv42(x)
        x = nn.ReLU(inplace=True)(x)
        if self.layer == 19:
            x = self.conv43(x)
            x = nn.ReLU(inplace=True)(x)
        x = self.conv44(x)
        x = self.concat4(x)
        x = self.downsample4(x)

        x = self.conv51(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv52(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv53(x)
        x = nn.ReLU(inplace=True)(x)
        if self.layer == 19:
            x = self.conv54(x)
            x = nn.ReLU(inplace=True)(x)
        x = self.downsample5(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)

        return x
