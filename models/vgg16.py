import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, layer
from utils.modules import MyFloor, ScaledNeuron, BurstNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_list = ['conv1', ['conv2','pool2'], 'conv3', ['conv4','pool4'], 'conv5', 'conv6', ['conv7','pool7'], 'conv8',
              'conv9', ['conv10','pool10'], 'conv11', 'conv12', ['conv13','pool13'], 'fc14', 'fc15', 'fc16']


class VGG16(nn.Module):
    def __init__(self, num_class, dropout=0):
        super(VGG16, self).__init__()
        self.bias = True
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        self.pool4 = nn.AvgPool2d(2, 2)

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        self.pool7 = nn.AvgPool2d(2, 2)

        self.conv8 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=self.bias),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))

        self.conv10 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=self.bias),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        self.pool10 = nn.AvgPool2d(2, 2)

        self.conv11 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=self.bias),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))

        self.conv12 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=self.bias),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))

        self.conv13 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=self.bias),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        self.pool13 = nn.AvgPool2d(2, 2)

        if num_class == 1000:
            self.fc14 = nn.Sequential(nn.Linear(7 * 7 * 512, 4096, bias=self.bias),
                                      nn.ReLU(inplace=True) )
        elif num_class == 200:
            self.fc14 = nn.Sequential(nn.Linear(2 * 2 * 512, 4096, bias=self.bias),
                                      nn.ReLU(inplace=True) )
        else:
            self.fc14 = nn.Sequential(nn.Linear(1 * 1 * 512, 4096, bias=self.bias),
                                      nn.ReLU(inplace=True) )

        self.fc15 = nn.Sequential(nn.Linear(4096, 4096, bias=self.bias),
                                  nn.ReLU(inplace=True))
        self.fc16 = nn.Sequential(nn.Linear(4096, num_class, bias=self.bias),)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, SNN = False, TTFS = False):
        # Conv Layer
        x1 = self.conv1(x)
        x2_input = x1.detach() if SNN else x1
        x2 = self.conv2(x2_input)
        x2 = self.pool2(x2)
        x3_input = x2.detach() if SNN else x2
        x3 = self.conv3(x3_input)
        x4_input = x3.detach() if SNN else x3
        x4 = self.conv4(x4_input)
        x4 = self.pool4(x4)
        x5_input = x4.detach() if SNN else x4
        x5 = self.conv5(x5_input)
        x6_input = x5.detach() if SNN else x5
        x6 = self.conv6(x6_input)
        x7_input = x6.detach() if SNN else x6
        x7 = self.conv7(x7_input)
        x7 = self.pool7(x7)
        x8_input = x7.detach() if SNN else x7
        x8 = self.conv8(x8_input)
        x9_input = x8.detach() if SNN else x8
        x9 = self.conv9(x9_input)
        x10_input = x9.detach() if SNN else x9
        x10 = self.conv10(x10_input)
        x10 = self.pool10(x10)
        x11_input = x10.detach() if SNN else x10
        x11 = self.conv11(x11_input)
        x12_input = x11.detach() if SNN else x11
        x12 = self.conv12(x12_input)
        x13_input = x12.detach() if SNN else x12
        x13 = self.conv13(x13_input)
        x13 = self.pool13(x13)

        # FC Layers
        x13 = x13.view(x13.size(0), -1)
        x14_input = x13.detach() if SNN else x13
        x14 = self.fc14(x14_input)
        x15_input = x14.detach() if SNN else x14
        x15 = self.fc15(x15_input)
        x16_input = x15.detach() if SNN else x15
        if not TTFS:
            out = self.fc16(x16_input)
        else:
            out = x16_input
        hidden_act = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]

        return hidden_act, out

    def shapes(self, x):
        hidden = []
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)  \
                    or isinstance(m, nn.Dropout) or isinstance(m, MyFloor) or isinstance(m, nn.Flatten) \
                    or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d) or isinstance(m, ScaledNeuron) :
                prev_x = x
                #print(x.size())
                x = m(x)
                if isinstance(m, nn.Conv2d):
                    hidden.append(prev_x)
            elif isinstance(m, nn.Linear):
                prev_x = x
                #print(x.size())
                x = m(x.view(x.size(0), -1))

                hidden.append(prev_x)
        return hidden, x