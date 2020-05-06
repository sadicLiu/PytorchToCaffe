from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, num_classes=9, pretrained_model_path=None):
        super(ResNet, self).__init__()
        self.net = models.resnet18()
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        if pretrained_model_path is not None:
            pretrained_model = Path(pretrained_model_path)
            state_dict_load = torch.load(pretrained_model)
            self._load_pretrained_model(state_dict_load)

    def forward(self, x):
        return self.net(x)

    def _load_pretrained_model(self, state_dict_load):
        if type(state_dict_load) is dict:
            super(ResNet, self)._load_pretrained_model(state_dict_load)
        else:
            exclude_str = 'fc.'
            state_dict_load = OrderedDict([(k, v) for k, v in state_dict_load.items() if not k.startswith(exclude_str)])
            self.net.load_state_dict(state_dict_load, strict=False)
            keys = state_dict_load.keys()
            for name, param in self.net.named_parameters():
                if name in keys:
                    param.requires_grad = False
