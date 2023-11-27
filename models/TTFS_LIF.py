import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable
from spikingjelly.clock_driven import surrogate
import random

class TTFS_LIF(nn.Module):
    """LIF neurons in output layer"""
    def __init__(self, in_features, out_features, tau=2.0, tau_s=2.0 / 4, v_threshold=1.0, surrogate_function: Callable = surrogate.Sigmoid()):

        super().__init__()
        self.tau = tau
        self.tau_s = tau_s
        #self.T = T
        self.v_threshold = v_threshold
        self.fc = nn.Linear(in_features, out_features, bias=False)

        # v0 normalize the maximum value of PSP kernel to Vth
        t_max = (tau * tau_s * math.log(tau / tau_s)) / (tau - tau_s)
        self.v0 = self.v_threshold / (math.exp(-t_max / tau) - math.exp(-t_max / tau_s))
        self.surrogate_function = surrogate_function
        self.delta=1

    def forward(self, threshold, count_t: torch.Tensor):
        # The last hidden layer’s output
        spike_in = count_t # shape=[batch_size, out_features, T]

        # PSP kernel iteration
        psp1 = math.exp(-self.delta / self.tau) * spike_in
        psp2 = math.exp(-self.delta / self.tau_s) * spike_in
        for step in range(spike_in.size(2)):
            if step > 0:
                psp1[:, :, step] += psp1[:, :, step-1] * math.exp( -1 / self.tau)
                psp2[:, :, step] += psp2[:, :, step-1] * math.exp( -1 / self.tau_s)

        # Post synaptic membrane potential
        v_out = self.fc(self.v0 *(psp1-psp2).permute(0, 2, 1)).permute(0, 2, 1)

        # Spike generation
        if torch.is_tensor(threshold):
            spike = self.surrogate_function(v_out-threshold.unsqueeze(0).unsqueeze(0).
                                           repeat(v_out.size(0),v_out.size(1),1))
        else:
            spike = self.surrogate_function(v_out - threshold)

        return v_out.permute(0, 2, 1), spike.permute(0, 2, 1)  # [N, T, Class]


class TTFS_LIF_linear(nn.Module):
    def __init__(self, input_dim, out_dim):
        # input shape [N, H, T]
        super().__init__()
        self.LIF = TTFS_LIF(input_dim, out_dim)

    def forward(self,threshold,count_t: torch.Tensor):
        return self.LIF(threshold,count_t)
