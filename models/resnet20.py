import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.clock_driven import neuron, functional, layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8',
              'conv9', 'conv10', 'conv11', 'conv12', 'conv13', 'fc14', 'fc15', 'fc16']



def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                nn.ReLU(inplace=True))

    def forward(self, x, SNN = False):
        out1 = self.conv1(x)
        out2_input = out1
        out2 = self.conv2(out2_input)

        if len(self.shortcut) > 0:
            out3 = self.shortcut(x)
        else:
            out3 = x

        out = out2 + out3
        return out, [out1, out2, out3]

class ResNet20(nn.Module):
    def __init__(self, num_class=10):
        super(ResNet20, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = BasicBlock(64, 64, stride=1)
        self.layer5 = BasicBlock(64, 64, stride=1)
        self.layer6 = BasicBlock(64, 128, stride=2)
        self.layer7 = BasicBlock(128, 128, stride=1)
        self.layer8 = BasicBlock(128, 256, stride=2)
        self.layer9 = BasicBlock(256, 256, stride=1)
        self.layer10 = BasicBlock(256, 512, stride=2)
        self.layer11 = BasicBlock(512, 512, stride=1)
        self.pool12 = nn.AvgPool2d(2, 2)
        if num_class == 200:
            self.fc13 = nn.Sequential(nn.Linear(512 * 2 * 2, 4096),
                                    nn.ReLU(inplace=True))
        else:
            self.fc13 = nn.Sequential(nn.Linear(512 * 1 * 1, 4096),
                                      nn.ReLU(inplace=True))
        self.linear = nn.Linear(4096, num_class, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, SNN = False, TTFS=False):
        x1 = self.conv1(x)
        x2_input = x1.detach() if SNN else x1
        x2 = self.conv2(x2_input)
        x3_input = x2.detach() if SNN else x2
        x3 = self.conv3(x3_input)
        x4_input = x3.detach() if SNN else x3

        x4, x4_mid = self.layer4(x4_input, SNN)
        x5_input = x4.detach() if SNN else x4
        x5, x5_mid = self.layer5(x5_input, SNN)
        x6_input = x5.detach() if SNN else x5
        x6, x6_mid = self.layer6(x6_input, SNN)
        x7_input = x6.detach() if SNN else x6
        x7, x7_mid = self.layer7(x7_input, SNN)
        x8_input = x7.detach() if SNN else x7
        x8, x8_mid = self.layer8(x8_input, SNN)
        x9_input = x8.detach() if SNN else x8
        x9, x9_mid = self.layer9(x9_input, SNN)
        x10_input = x9.detach() if SNN else x9
        x10, x10_mid = self.layer10(x10_input, SNN)
        x11_input = x10.detach() if SNN else x10
        x11, x11_mid = self.layer11(x11_input, SNN)
        x12 = self.pool12(x11)
        x12 = x12.view(x12.size(0), -1)

        x13_input = x12.detach() if SNN else x12
        x13 = self.fc13(x13_input)
        x14_input = x13.detach() if SNN else x13
        if not TTFS:
            out = self.linear(x14_input)
        else:
            out=x14_input
        return (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13), out


