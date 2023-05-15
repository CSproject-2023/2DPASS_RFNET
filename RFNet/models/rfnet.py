import torch.nn as nn
from itertools import chain # 串联多个迭代对象
import torch

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
        logits_upsample = upsample(logits, data_dict['img'].shape[2:])
        # return logits_upsample

        e1 = torch.nn.functional.interpolate(logits, size=(480, 480), mode='bilinear', align_corners=False)
        e1 = e1.repeat(1, 3, 1, 1)
        e2 = e1[:, :4, :, :]
        e1 = torch.cat((e1, e2), dim=1)
        data_dict['img_scale2'] = e1
        data_dict['img_scale4'] = e1


        # s1 = torch.nn.functional.interpolate(logits_upsample, size=(480, 480), mode='bilinear', align_corners=False)
        s1 = logits_upsample.repeat(1, 3, 1, 1)
        s2 = s1[:, :4, :, :]
        s1 = torch.cat((s1, s2), dim=1)
        data_dict['img_scale8'] = s1
        data_dict['img_scale16'] = s1

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                # print(data_dict[k].size())
                # print(img_indices[i][:, 0], img_indices[i][:, 1])
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)


        return data_dict

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()