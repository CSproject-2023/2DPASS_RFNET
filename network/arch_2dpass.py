import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from network.basic_block import Lovasz_loss
from network.spvcnn import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNextFCN
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")

class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def fusion_to_single_KD(self, data_dict, idx):
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']

        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

        # modality fusion
        feat_learner = F.relu(self.leaners[idx](pts_feat))
        feat_cat = torch.cat([img_feat, feat_learner], 1)
        feat_cat = self.fcs1[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2[idx](feat_cat))
        fuse_feat = F.relu(feat_cat * feat_weight)

        # fusion prediction
        fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

        # KL divergence
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales

        return loss, fuse_feat

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []

        for idx in range(self.num_scales):
            singlescale_loss, fuse_feat = self.fusion_to_single_KD(data_dict, idx)
            img_seg_feat.append(fuse_feat)
            loss += singlescale_loss

        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['loss'] += loss

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)

        pytorch_device = torch.device('cuda:0')

        config_path = args.config_path

        configs = load_config_data(config_path)

        dataset_config = configs['dataset_params']
        train_dataloader_config = configs['train_data_loader']
        val_dataloader_config = configs['val_data_loader']

        val_batch_size = val_dataloader_config['batch_size']
        train_batch_size = train_dataloader_config['batch_size']

        model_config = configs['model_params']
        train_hypers = configs['train_params']

        grid_size = model_config['output_shape']
        num_class = model_config['num_class']
        ignore_label = dataset_config['ignore_label']

        model_load_path = train_hypers['model_load_path']
        model_save_path = train_hypers['model_save_path']
        wd = train_hypers['weight_decay']      # weight decay
        amp = train_hypers['mixed_fp16']
        SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
        unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

        self.model_3d = model_builder.build(model_config)

        if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

        my_model.to(pytorch_device)
        
        if not self.baseline_only:
            self.model_2d = ResNextFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
            self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        if self.training and not self.baseline_only:
            data_dict = self.model_2d(data_dict)
            data_dict = self.fusion(data_dict)

        return data_dict