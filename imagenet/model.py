import os
import sys
import math
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import set_layer_from_config
from utils import BasicUnit
    

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
    def unit_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.unit_str, self.shortcut.unit_str if self.shortcut is not None else None
        )

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

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(NASNet, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def unit_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': NASNet.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'feature_mix_layer': self.feature_mix_layer.config if self.feature_mix_layer is not None else None,
            'classifier': self.classifier.config,
            'blocks': [
                block.config for block in self.blocks
            ],
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        return NASNet(first_conv, blocks, feature_mix_layer, classifier)

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
