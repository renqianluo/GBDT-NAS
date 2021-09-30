import os
import sys
import glob
import time
import copy
import random
import math
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch import Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNet
import utils


def get_args():
    parser = argparse.ArgumentParser()

    # Basic model parameters.
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--width_stages', type=str, default='32,48,96,136,192')
    parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
    parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')
    parser.add_argument('--flops', type=int, default=6e8)

    args = parser.parse_args()
    return args


def count_convNd(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features

    m.total_ops = torch.Tensor([int(total_ops)])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    ######################################
    nn.Linear: count_linear,
    ######################################
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
    nn.BatchNorm2d: None,
}


def profile(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            # print("Not implemented for ", m_)
            pass

        if fn is not None:
            # print("Register FLOP counter for module %s" % str(m_))
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(original_device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params


def main():
    args = get_args()
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    
    model_config = utils.build_model_config(args.arch, args.width_stages, args.n_cell_stages, args.stride_stages, 0)
    model = NASNet.build_from_config(model_config)
    flops, params = profile(model, (1, 3, 224, 224))
    print("new width:{}".format(args.width_stages))
    print("param size = %fM" % (params / 1000000))
    print("multi adds = %fM" % (flops / 1000000))
    if flops > args.flops:
        #ratio = math.sqrt(args.flops/flops)
        ratio = args.flops/flops
        print(ratio)
        ws = [utils.make_divisible(v*ratio, 8, 8) for v in args.width_stages]
        model_config = utils.build_model_config(args.arch, ws, args.n_cell_stages, args.stride_stages, 0)
        model = NASNet.build_from_config(model_config)
        flops, params = profile(model, (1, 3, 224, 224))
        print("new width:{}".format(ws))
        print("new param size = %fM" % (params / 1000000))
        print("new multi adds = %fM" % (flops / 1000000))
        

if __name__ == '__main__':
    main()
