'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import json
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random
from collections import OrderedDict
from typing import Union

from torch import Tensor

A_list=[[16,15,14,13,12,11],        #1
        [16,15,14,13,10, 9, 8, 7],  #2
        [16,15,14,12,11,10, 6, 5],  #3
        [16,15,13,12,11,9,8,4],     #4
        [16,15,14,13,12,10,9,7,6,3],#5
        [16,15,14,13,11,10,8,5],    #6
        [16,15,14,13,12,11,9,7],    #7
        [16,14,12,10,8,6,4,2],      #8
        [16,15,14,13,12,11,9,7],    #9
        [16,15,14,13,11,10,8,5],    #10
        [16,15,14,13,12,10,9,7,6,3], #11
        [16,15,13,12,11,9,8,4],    #12
        [16,15,14,12,11,10,6,5],    #13
        [16,15,14,13,10,9,8,7],     #14
        [16,15,14,13,12,11],    #15
        [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], #16
        ]


def freezeLayer(model, layer_list, layer_idx):
    ''' freeze the fine-tuned layer '''
    if isinstance(layer_list[layer_idx], list):
        for layer_name in layer_list[layer_idx]:
            layer = eval('model.' + str(layer_name))
            for param in layer.parameters():
                param.requires_grad = False
    else:
        layer = eval('model.' + str(layer_list[layer_idx]))
        for param in layer.parameters():
            param.requires_grad = False
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def Quantize_lvl(tensor, level=16):
    # print(min_float, max_float)
    min_float = tensor.min().item()
    max_float = tensor.max().item()
    tensor.clamp_(min_float, max_float)
    scale = (max_float- min_float)/level
    #min_float_adjusted = min_float + (-min_float)%scale
    tensor = (tensor - min_float).div(scale).round().mul(scale) + min_float
    return tensor

def Quantize(tensor, numBits=8):
    # print(min_float, max_float)
    min_float = tensor.min().item()
    max_float = tensor.max().item()
    tensor.clamp_(min_float, max_float)
    scale = (max_float - min_float) / (2 ** numBits - 1)
    min_float_adjusted = min_float + (-min_float) % scale
    tensor = (tensor - min_float_adjusted).div(scale).round().mul(scale) + min_float_adjusted
    return tensor

def get_SNN_layer_output(model, input, T, layer_list, iLayer):
    ''' get layer-wise output '''
    # Get the layer
    if isinstance(layer_list[iLayer], list): # is layer contain pool
        for layer_name in layer_list[iLayer]:
            if layer_name.startswith('conv'):
                layer = eval('model.conv' + str(iLayer + 1))
            elif layer_name.startswith('pool'):
                layer_pool = eval('model.pool' + str(iLayer + 1))
    else:
        if layer_list[iLayer].startswith('conv'):
            layer = eval('model.conv' + str(iLayer + 1))
        elif layer_list[iLayer].startswith('fc'):
            layer = eval('model.fc' + str(iLayer + 1))
        else:
            print('No supported layers!')

    # Inference
    x = input/T if iLayer !=0 else input # Average to each T
    spk_cnt = [0] * T
    outputs =0
    for t in range(T):
        if isinstance(layer_list[iLayer], list):
            x = layer(x)
            output = layer_pool(x)
        else:
            output = layer(x)

        outputs += output
        spk_cnt[t] += (((output > 0).float()).sum()).item()
    #spk_cnt = [x / input.size(0) for x in spk_cnt]
    return outputs/T, spk_cnt


def Channel_Norm(model, data_loader, device, HidLayer, p=99):
    """Perform channel normalization"""
    model.eval()  # Put the model in test mode

    scale_layer_list = [[]] * HidLayer
    #cnt = 0
    for i_batch, (inputs, labels) in enumerate(data_loader, 1):
        #cnt += 1
        #if cnt > 50:
        #    break

        # Transfer to GPU
        inputs = inputs.type(torch.FloatTensor).to(device)

        # forward pass to get channelwise activation values
        with torch.no_grad():
            hiddenA, _ = model.forward(inputs)

        for iLayer, A in enumerate(hiddenA):
            if len(A.shape) > 3:  # for Conv layer
                scale = percentile_ch(A, p)
                scale = scale.to(device)
            else:  # for Linear layer
                scale = percentile(A, p)
                scale = torch.tensor(scale).to(device)

            if i_batch == 1: #init the list
                scale_layer_list[iLayer] = scale
            else:
                scale_layer_list[iLayer] = torch.add(scale_layer_list[iLayer], scale)

    scale_list = [x/i_batch for x in scale_layer_list]
    return scale_list


def Channel_Norm_DDP(model, data_loader, args, p=99):
    """Perform channel normalization"""
    model.eval()  # Put the model in test mode

    scale_layer_list = [[]] * 15
    for i_batch, (inputs, labels) in enumerate(data_loader, 1):
        # Transfer to GPU
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
        #inputs = inputs.type(torch.FloatTensor).to(args)

        # forward pass to get channelwise activation values
        with torch.no_grad():
            hiddenA, _ = model.forward(inputs)

        for iLayer, A in enumerate(hiddenA):
            if len(A.shape) > 3:  # for Conv layer
                scale = percentile_ch(A, p)
                scale = scale.cuda(args.gpu, non_blocking=True)
            else:  # for Linear layer
                scale = percentile(A, p)
                scale = torch.tensor(scale).cuda(args.gpu, non_blocking=True)

            if i_batch == 1: #init the list
                scale_layer_list[iLayer] = scale
            else:
                scale_layer_list[iLayer] = torch.add(scale_layer_list[iLayer], scale)

    scale_list = [x/i_batch for x in scale_layer_list]
    return scale_list



def weights_init(m, wInit, bInit):
    if isinstance(m, nn.Conv2d):
        m.bias.data = bInit
        m.weight.data = wInit
    elif isinstance(m, nn.Linear):
        m.bias.data = bInit
        m.weight.data = wInit


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def state_dict_data_parallel(state_dict):
    """# remove 'module.' of for model trained with dataParallel """

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v

    return new_state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')    

def save_checkpoint(epoch, model, optimizer, ckp_dir, best=True):
    if not os.path.isdir(ckp_dir):
        os.mkdir(ckp_dir)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(ckp_dir, "{0}.pt.tar".format("best" if best else "last")))


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def percentile_ch(t: torch.tensor, q: float) -> Tensor:
    """
    Return the channel-wise ``q``-th percentile of the input tensor's data.
    """
    result = []
    for ch in range(t.size(1)):
        t_ch = t[:, ch, :, :]
        k_ch = 1 + round(.01 * float(q) * (t_ch.numel() - 1))
        result_ch = t_ch.reshape(-1).kthvalue(k_ch).values.item()
        if result_ch == 0:
            result_ch = 1e-8
        result.append(result_ch)
    result = torch.tensor(result)
    return result


def channel_norm(t: torch.tensor, q: float) -> Tensor:
    """
    Return the channel-wise normalization of the input tensor's data.
    """
    A_norm = []
    for ch in range(t.size(1)):
        t_ch = t[:, ch, :, :]
        k_ch = 1 + round(.01 * float(q) * (t_ch.numel() - 1))
        result_ch = t_ch.reshape(-1).kthvalue(k_ch).values.item()
        if result_ch == 0:
            result_ch = 1e-8
        t_norm = torch.clamp(t_ch / result_ch, min=0, max=1.0)
        A_norm.append(t_norm)
    A_norm = torch.stack(A_norm).permute(1, 0, 2, 3)
    return A_norm
