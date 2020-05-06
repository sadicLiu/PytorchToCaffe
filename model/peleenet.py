import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PeleeNet', 'BasicConv2d']


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features=None, growth_rate=None, bottleneck_width=None, cfg=None):
        """bottleneck_width: 决定了中间层卷积核的数量"""
        super(_DenseLayer, self).__init__()

        if cfg is None:
            growth_rate = int(growth_rate / 2)  # 32/2=16
            inter_channel = int(growth_rate * bottleneck_width / 4) * 4  # 16*[1, 2, 4, 4] = [16, 32, 64, 64]
            if inter_channel > num_input_features / 2:
                inter_channel = int(num_input_features / 8) * 4
                print('adjust inter_channel to ', inter_channel)

            # branch1只是比branch2多了一个conv3x3
            self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
            self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
            self.branch2a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
            self.branch2b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
            self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        else:
            self.branch1a = BasicConv2d(cfg.num_input_features, cfg.branch1a_output_features, kernel_size=1)
            self.branch1b = BasicConv2d(cfg.branch1a_output_features, cfg.branch1b_output_features,
                                        kernel_size=3, padding=1)

            self.branch2a = BasicConv2d(cfg.num_input_features, cfg.branch2a_output_features, kernel_size=1)
            self.branch2b = BasicConv2d(cfg.branch2a_output_features, cfg.branch2b_output_features,
                                        kernel_size=3, padding=1)
            self.branch2c = BasicConv2d(cfg.branch2b_output_features, cfg.branch2c_output_features,
                                        kernel_size=3, padding=1)

    def forward(self, x):
        features = [x]

        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)
        features.append(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)
        features.append(branch2)

        return torch.cat(features, 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers=None, num_input_features=None, bn_size=None, growth_rate=None, cfg=None):
        """bn_size: bottleneck width"""
        super(_DenseBlock, self).__init__()

        if cfg is None:
            assert None not in [num_layers, num_input_features, bn_size, growth_rate], 'Invalid arguments!'
            for i in range(num_layers):
                denselayer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
                self.add_module('denselayer%d' % (i + 1), denselayer)
        else:
            cfg_layers = cfg.layers  # list
            for i in range(len(cfg_layers)):
                denselayer = _DenseLayer(cfg=cfg_layers[i])
                self.add_module('denselayer%d' % (i + 1), denselayer)


class _StemBlock(nn.Module):
    """经过StemBlock, 宽高变为原来的1/4"""

    def __init__(self, num_input_channels=None, num_init_features=None, cfg=None):  # 3, 32
        super(_StemBlock, self).__init__()

        if cfg is None:
            assert None not in [num_init_features, num_input_channels], 'Invalid arguments!'
            num_stem_features = int(num_init_features / 2)  # 16
            self.stem1 = BasicConv2d(num_input_channels, num_init_features, kernel_size=3, stride=2, padding=1)
            self.stem2a = BasicConv2d(num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
            self.stem2b = BasicConv2d(num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
            self.stem3 = BasicConv2d(2 * num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        else:
            self.stem1 = BasicConv2d(cfg.num_input_features, cfg.stem1_output_features,
                                     kernel_size=3, stride=2, padding=1)
            self.stem2a = BasicConv2d(cfg.stem1_output_features, cfg.stem2a_output_features,
                                      kernel_size=1, stride=1, padding=0)
            self.stem2b = BasicConv2d(cfg.stem2a_output_features, cfg.stem2b_output_features,
                                      kernel_size=3, stride=2, padding=1)
            self.stem3 = BasicConv2d(cfg.stem1_output_features + cfg.stem2b_output_features,
                                     cfg.num_output_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch1 = self.pool(out)
        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out


class BasicConv2d(nn.Module):
    """将conv-bn-relu(当activation=True时)封装成了一个函数"""

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x, inplace=True)
            # return F.relu6(x, inplace=True)
        else:
            return x


class PeleeNet(nn.Module):
    """PeleeNet model class
    Args:
        growth_rate (int or list of 4 ints) - 每个DenseLayer中卷积核的数量 (`k` in paper)
        block_config (list of 4 ints) - 每个DenseBlock中DenseLayer的数量
        num_init_features (int) - 第一个卷积层中卷积核的数量
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer), 这个参数影响的是dense layer中中间卷积层卷积核的数量
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        net_configs (list(BlockConfig)) - 剪枝之后每个Block的配置
    """

    def __init__(
            self,
            growth_rate=32, num_init_features=32, bottleneck_width=[1, 2, 4, 4],
            block_config=[3, 4, 8, 6], num_classes=9,
            drop_rate=0,
            pretrained_model_path=None,
            net_configs=None
    ):
        super(PeleeNet, self).__init__()

        # 使用net_configs创建模型的场景:
        # 1.剪枝之后用于测试模型效果, 这种情况在test.py中加载模型参数, 这里不用管
        # 2.微调剪枝之后的模型, 这时候json配置文件中设置好了pretrained_model_path, 把参数加载进来
        if net_configs is not None:
            self._build_features_from_cfg(net_configs, num_classes=num_classes)
        else:
            self._build_features(growth_rate=growth_rate, num_init_features=num_init_features,
                                 bottleneck_width=bottleneck_width, block_config=block_config, num_classes=num_classes)

        self.drop_rate = drop_rate

        if pretrained_model_path is None and net_configs is None:
            # 直接创建模型, 并随机初始化参数, 用于从头开始训练模型
            self._initialize_weights()
        elif pretrained_model_path is not None and net_configs is None:
            # 用于微调使用默认配置创建的模型
            self._load_pretrained_model(pretrained_model_path)
        elif pretrained_model_path is not None and net_configs is not None:
            # 用于微调剪枝之后使用cfg创建的模型
            self._restore_pruned_model(pretrained_model_path)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, kernel_size=(features.size(2), features.size(3))).view(features.size(0), -1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.classifier(out)
        return out

    def _build_features(self, growth_rate=32, num_init_features=32, bottleneck_width=[1, 2, 4, 4],
                        block_config=[3, 4, 8, 6], num_classes=11):
        """使用默认配置创建模型特征提取层"""
        self.features = nn.Sequential(
            OrderedDict([('stemblock', _StemBlock(3, num_init_features)), ])
        )

        if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [bottleneck_width] * 4

        # Each denseblock
        num_features = num_init_features  # 32
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i])
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rates[i]

            self.features.add_module('transition%d' % (i + 1),
                                     BasicConv2d(num_features, num_features, kernel_size=1, stride=1, padding=0))

            if i != len(block_config) - 1:
                self.features.add_module('transition%d_pool' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))
                num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def _build_features_from_cfg(self, net_configs, num_classes=11):
        """根据配置信息还原出剪枝后的模型"""
        index = 0
        stem = _StemBlock(cfg=net_configs[index])
        index += 1

        self.features = nn.Sequential(OrderedDict([('stemblock', stem), ]))

        while index < len(net_configs):
            block_cfg = net_configs[index]
            block = _DenseBlock(cfg=block_cfg)
            self.features.add_module('denseblock%d' % index, block)

            self.features.add_module('transition%d' % index,
                                     BasicConv2d(block_cfg.num_output_features, block_cfg.transition_output_features,
                                                 kernel_size=1, stride=1, padding=0))
            if index != len(net_configs) - 1:
                self.features.add_module('transition%d_pool' % index, nn.AvgPool2d(kernel_size=2, stride=2))

            index += 1

        # Linear layer
        self.classifier = nn.Linear(net_configs[-1].transition_output_features, num_classes)

    def _initialize_weights(self):
        """随机初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrained_model_path):
        """加载预训练模型"""
        pretrained_model = Path(pretrained_model_path)
        state_dict_load = torch.load(pretrained_model)

        if type(state_dict_load) is dict and len(state_dict_load) == 6:
            # 加载自己保存的模型
            state_dict_load = state_dict_load['state_dict']
            self.load_state_dict(state_dict_load)
            for param in self.parameters():
                param.requires_grad = True
        else:
            # 加载ImageNet预训练模型
            exclude_str = 'classifier'
            model_dict = self.state_dict()
            state_dict = state_dict_load['state_dict']
            pretrained_dict = {k.strip('module.'): v for k, v in state_dict.items()
                               if k.strip('module.') in model_dict and not k.strip('module.').startswith(exclude_str)}

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            keys = pretrained_dict.keys()
            for name, param in self.named_parameters():
                if name in keys:
                    param.requires_grad = False

    # todo: test
    def _restore_pruned_model(self, pruned_model_path):
        checkpoint = torch.load(pruned_model_path)
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    model = PeleeNet()
    print(model)


    def print_size(self, input, output):
        print(torch.typename(self).split('.')[-1], ' output size:', output.data.size())


    for layer in model.features:
        layer.register_forward_hook(print_size)

    input_var = torch.autograd.Variable(torch.Tensor(1, 3, 112, 112))
    output = model.forward(input_var)
