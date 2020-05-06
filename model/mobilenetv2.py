from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    def __init__(self, num_classes=9, pretrained_model_path=None):
        super(MobileNet, self).__init__()
        self.net = models.mobilenet_v2()
        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.net.classifier[1].in_features, num_classes)
        )

        if pretrained_model_path is not None:
            state_dict_load = torch.load(Path(pretrained_model_path))
            self._load_pretrained_model(state_dict_load)

    def forward(self, x):
        return self.net(x)

    def _load_pretrained_model(self, state_dict_load):
        if type(state_dict_load) is dict:
            super(MobileNet, self)._load_pretrained_model(state_dict_load)
        else:
            exclude_str = 'classifier'
            state_dict_load = OrderedDict([(k, v) for k, v in state_dict_load.items() if not k.startswith(exclude_str)])
            self.net.load_state_dict(state_dict_load, strict=False)
            keys = state_dict_load.keys()
            for name, param in self.net.named_parameters():
                if name in keys:
                    param.requires_grad = False


if __name__ == '__main__':
    net = MobileNet(pretrained_model_path='../../assets/mobilenet_v2-b0353104.pth')
    net.model_info()
