import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, List, Union, Dict, cast
from distributed_utils.dist_helper import allaverage
from utils.modules import ScaledNeuron, StraightThrough

def isActivation(name):
    if 'relu' in name.lower():
        return True
    return False

def isSNNneuron(name):
    if 'scaledneuron' in name.lower():
        return True
    return False

def replace_activation_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_neuron(module)
        if name.startswith('relu'):  # for res BLK:
            model._modules[name] = ScaledNeuron(scale=model.residual_function[3].threshold.item())
        else:
            if isActivation(module.__class__.__name__.lower()):
                if hasattr(model[0], "threshold"):
                    model._modules[name] = ScaledNeuron(scale=model[0].threshold.item())
                else:
                    model._modules[name] = ScaledNeuron(scale=1.)
    return model

def get_layer_shape(model, data, device):
    data = data.to(device)
    hidden = model.forward(data, SNN=True, isGetShape=True)
    return [x.squeeze(0) for x in hidden]

def replace_IF_by_Burst_all(model,gamma):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_IF_by_Burst_all(module,gamma)
        if isSNNneuron(module.__class__.__name__.lower()):
            if hasattr(model[0], "threshold"):
                model._modules[name] = ScaledNeuron(scale=model[0].threshold.item(), Burst=True, gamma=gamma)
            else:
                model._modules[name] = ScaledNeuron(scale=1.0, Burst=True, gamma=gamma)
    return model

def get_neuron_threshold(model):
    threshold = []
    for name, module in model._modules.items():
        if hasattr(module,"_modules") and len(module._modules) > 1:
            if name.startswith('layer'):
                threshold.append(module.conv2[0].threshold.item())
            else:
                threshold.append(module[0].threshold.item())
    return threshold

def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model

def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias

def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ------------------------- Max Activation ---------------------------

class DataSaverHook:
    def __init__(self, momentum: Union[float, None] = 0.9, sim_length: int = 8,
                 mse: bool = True, percentile: Union[float, None] = None, channel_wise: bool = False,
                 dist_avg: bool = False):
        self.momentum = momentum
        self.max_act = None
        self.T = sim_length
        self.mse = mse
        self.percentile = percentile
        self.channel_wise = channel_wise
        self.dist_avg = dist_avg

    def __call__(self, module, input_batch, output_batch):
        def get_act_thresh(tensor):
            if self.mse:
                act_thresh = find_threshold_mse(output_batch, T=self.T, channel_wise=self.channel_wise)
            elif self.percentile is not None:
                assert 0. <= self.percentile <= 1.0
                act_thresh = quantile(output_batch, self.percentile)
            else:
                act_thresh = tensor.max()
            return act_thresh

        if self.max_act is None:
            self.max_act = get_act_thresh(output_batch)
        else:
            cur_max = get_act_thresh(output_batch)
            if self.momentum is None:
                self.max_act = self.max_act if self.max_act > cur_max else cur_max
            else:
                self.max_act = self.momentum * self.max_act + (1 - self.momentum) * cur_max
        if self.dist_avg:
            allaverage(self.max_act)
        module.threshold = self.max_act


def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p*100)).type_as(tensor)


def find_threshold_mse(tensor: torch.Tensor, T: int = 8, channel_wise: bool = True):
    """
    This function use grid search to find the best suitable
    threshold value for snn.
    :param tensor: the output batch tensor,
    :param T: simulation length
    :param channel_wise: set threshold channel-wise
    :return: threshold with MMSE
    """
    def clip_floor(tensor:torch.Tensor, T: int, Vth: Union[float, torch.Tensor]):
        snn_out = torch.clamp(tensor / Vth * T, min=0, max=T)
        return snn_out.floor() * Vth / T

    if channel_wise:
        num_channel =tensor.shape[1]
        best_Vth = torch.ones(num_channel).type_as(tensor)
        # determine the Vth channel-by-channel
        for i in range(num_channel):
            best_Vth[i] = find_threshold_mse(tensor[:, i], T, channel_wise=False)
        best_Vth = best_Vth.reshape(1, num_channel, 1, 1) if len(tensor.shape)==4 else best_Vth.reshape(1, num_channel)
    else:
        max_act = tensor.max()
        best_score = 1e5
        best_Vth = 0
        for i in range(95):
            new_Vth = max_act * (1.0 - (i * 0.01))
            mse = lp_loss(tensor, clip_floor(tensor, T, new_Vth), p=2.0, reduction='other')
            if mse < best_score:
                best_Vth = new_Vth
                best_score = mse

    return best_Vth


@torch.no_grad()
def get_maximum_activation(train_loader: torch.utils.data.DataLoader,
                           model,
                           momentum: Union[float, None] = 0.9,
                           iters: int = 20,
                           sim_length: int = 8,
                           mse: bool = True, percentile: Union[float, None] = None,
                           channel_wise: bool = False,
                           dist_avg: bool = False):
    """
    This function store the maximum activation in each convolutional or FC layer.
    :param train_loader: Data loader of the training set
    :param model: target model
    :param momentum: if use momentum, the max activation will be EMA updated
    :param iters: number of iterations to calculate the max act
    :param sim_length: sim_length when computing the mse of SNN output
    :param mse: if Ture, use MMSE to find the V_th
    :param percentile: if mse = False and percentile is in [0,1], use percentile to find the V_th
    :param channel_wise: use channel-wise mse
    :param dist_avg: if True, then compute mean between distributed nodes
    :return: model with stored max activation buffer
    """
    # do not use train mode here (avoid bn update)
    model.eval()
    device = next(model.parameters()).device
    hook_list = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            hook_list += [m.register_forward_hook(DataSaverHook(momentum, sim_length, mse, percentile, channel_wise,
                                                                dist_avg))]
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device=device)
        _ = model(input)
        if i > iters:
            break
    for h in hook_list:
        h.remove()


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    elif reduction == 'channel_split':
        return (pred-tgt).abs().pow(p).sum((0,2,3))
    else:
        return (pred-tgt).abs().pow(p).mean()
