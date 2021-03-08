import logging
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import BasicUnit, ConvLayer, LinearLayer, MBInvertedConvLayer, set_layer_from_config
from utils import OPERATIONS


class WSMobileInvertedResidualBlock(BasicUnit):
    def __init__(self, in_channels, out_channels, stride):
        super(WSMobileInvertedResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.mobile_inverted_conv = nn.ModuleList()

        for k, config in OPERATIONS.items():
            config = copy.copy(config)
            name = config.get('name')
            if name != 'ZeroLayer':
                config['in_channels'] = in_channels
                config['out_channels'] = out_channels
            config['stride'] = stride
            layer = set_layer_from_config(config)
            self.mobile_inverted_conv.append(layer)

    def forward(self, x, op_id):
        conv = self.mobile_inverted_conv[op_id-1]
        if conv.is_zero_layer():
            res = x
        elif self.stride != 1 or self.in_channels != self.out_channels:
            res = conv(x)
        else:
            conv_x = conv(x)
            skip_x = x
            res = skip_x + conv_x
        return res


class MobileInvertedResidualBlock(BasicUnit):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class NASNet(BasicUnit):

    def __init__(self, width_stages, n_cell_stages, stride_stages, dropout=0):
        super(NASNet, self).__init__()

        self.width_stages = width_stages
        self.n_cell_stages = n_cell_stages
        self.stride_stages = stride_stages
        
        in_channels = 32
        first_cell_width = 16
        
        # first conv layer
        self.first_conv = ConvLayer(3, in_channels, 3, 2, 1, 1, False, False, True, 'relu6', 0, 'weight_bn_act')
        
        # first block
        first_block_config = {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": in_channels,
                "out_channels": first_cell_width,
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 1
            },
            "shortcut": None
        }
        self.first_block = MobileInvertedResidualBlock.build_from_config(first_block_config)
        in_channels = first_cell_width

        # blocks
        self.blocks = nn.ModuleList()
        for width, n_cell, s in zip(self.width_stages, self.n_cell_stages, self.stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                block = WSMobileInvertedResidualBlock(in_channels, width, stride)
                in_channels = width
                self.blocks.append(block)

        self.feature_mix_layer = ConvLayer(in_channels, 1280, 1, 1, 1, 1, False, False, True, 'relu6', 0, 'weight_bn_act')
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(1280, 1000, True, False, None, dropout, 'weight_bn_act')

    def forward(self, x, arch, bn_train=False):
        if bn_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.train()
        x = self.first_conv(x)
        x = self.first_block(x)
        for i, block in enumerate(self.blocks):
            x = block(x, arch[i])
        #x = self.last_block(x)
        if self.feature_mix_layer:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def init_model(self, model_init, init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_parameters(self):
        return self.parameters()

    @staticmethod
    def _make_divisible(v, divisor, min_val=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_val:
        :return:
        """
        if min_val is None:
            min_val = divisor
        new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v