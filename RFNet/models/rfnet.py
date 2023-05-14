import torch.nn as nn
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, upsample


class RFNet(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet, self).__init__()
        self.  backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs, depth_inputs = None):
        x, additional = self.backbone(rgb_inputs, depth_inputs)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        return logits_upsample


    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()


class RFNet_2DPass(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet_2DPass, self).__init__()
        self.  backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, data_dict):
        data_dict['depth'] = data_dict['depth'][0].permute(2, 0, 1)
        x, additional = self.backbone(data_dict['img'], data_dict['depth'])
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        return logits_upsample


    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()