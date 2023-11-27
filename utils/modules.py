from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Function
from spikingjelly.clock_driven import neuron

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class ScaledNeuron(nn.Module):
    def __init__(self, scale=1.0, Burst=False, gamma=5):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_threshold=1.0, v_reset=None) if not Burst else BurstNode(gamma=gamma)
    def forward(self, x):          
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.0)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale
    def reset(self):
        self.t = 0
        self.neuron.reset()

class BurstNode(nn.Module):
    """Burst neurons in hidden layers"""
    def __init__(self, gamma):
        super(BurstNode, self).__init__()
        self.mem = 0
        self.spike = 0
        self.sum = 0
        self.threshold = 1.0
        self.summem = 0
        self.gamma = gamma

    def reset(self):
        self.mem = 0
        self.spike = 0

    def forward(self, x):
        self.mem = self.mem + x
        self.spike = myfloor((self.mem / self.threshold)).clamp(min=0, max=self.gamma)
        self.mem = self.mem - self.spike * self.threshold
        out = self.spike
        return out

class MyFloor(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GradFloor_partial(Function):
    @staticmethod
    def forward(ctx, input, mask):
        out = input.floor()
        isSpike = (out > 0).float()
        out = torch.where(mask == 0, isSpike, out)
        ctx.save_for_backward(isSpike, mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)  # for debuger
        isSpike, mask = ctx.saved_tensors
        grad_output = torch.where(mask == 0, isSpike, grad_output)
        return grad_output, None

myfloor = GradFloor.apply
myfloor_partial = GradFloor_partial.apply
