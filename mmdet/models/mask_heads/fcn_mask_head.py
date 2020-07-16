import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from ..mask_heads.attention import PAM_Module, CAM_Module
from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
import torch.nn.functional as F
import cv2

@HEADS.register_module
class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_edge=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.loss_edge = build_loss(loss_edge)
        kernel = torch.FloatTensor([[0, 1, 0],
                                    [1, 100, 1],
                                    [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.reduction = 16

        padding = (self.conv_kernel_size - 1) // 2
        # channel attention
        self.se_convs = nn.Sequential(nn.Conv2d(
                self.conv_out_channels, self.conv_out_channels, 5),
            nn.Conv2d(
                self.conv_out_channels, self.conv_out_channels, 5))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attehtion = nn.Sequential(nn.Linear(self.in_channels, self.in_channels // self.reduction,
                                                          bias=False),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(self.in_channels // self.reduction, self.in_channels,
                                                          bias=False),
                                               nn.Sigmoid())
        self.sa = PAM_Module(self.in_channels)
        self.ca = CAM_Module(self.in_channels)
        self.cat_convs = nn.Sequential(
            nn.Conv2d(
                self.in_channels*2, self.in_channels, 1),
            nn.Conv2d(
                self.in_channels, self.in_channels, 1))

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
           #  padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        # edge_attention
        self.ea_convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
           #  padding = (self.conv_kernel_size - 1) // 2
            self.ea_convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
            self.upsample_ea = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
            self.upsample_ea = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)
            self.upsample_ea = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(
                self.conv_out_channels, self.conv_out_channels, self.conv_kernel_size, padding=padding),
            nn.Conv2d(
                self.conv_out_channels, 1, 1))

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_trans = nn.Conv2d(logits_in_channel, logits_in_channel, 1)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in self.channel_attehtion:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(self.fc_cls.bias, 0)
        nn.init.constant_(self.sa.gamma, 0)
        nn.init.constant_(self.ca.gamma, 0)
        for m in self.cat_convs:
            nn.init.kaiming_normal_ (
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in self.edge_conv:
            nn.init.kaiming_normal_ (
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in self.se_convs:
            nn.init.kaiming_normal_ (
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.sa.query_conv,self.sa.key_conv,self.sa.value_conv]:
            nn.init.kaiming_normal_ (
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.upsample, self.upsample_ea, self.conv_trans, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        # mask_edge_pred = self.edge_conv1(x)
        # mask_edge_pred = self.edge_conv2(mask_edge_pred)
        b, c, _, _ = x.size()
        # ca = self.ca(x)
        ca = self.se_convs(x)
        ca = self.avg_pool(ca).view(b, c)
        ca = self.channel_attehtion(ca).view(b, c, 1, 1)
        
        ea = self.sa(x)
        #ea = x.clone()
        for ea_conv in self.ea_convs:
            ea = ea_conv(ea)

        ca = x*ca.expand_as(x)

        for conv in self.convs:
            x = conv(x)

        x = torch.cat((x,ca),1)
        x = self.cat_convs(x)
        if self.upsample is not None:
            x = self.upsample(x)
            ea = self.upsample_ea(ea)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
                ea = self.relu(ea)
        mask_edge_pred = self.edge_conv(ea)
        x = x + ea
        x = self.conv_trans(x)
        mask_pred = self.conv_logits(x)

        return mask_pred, mask_edge_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def get_edge_gt(self, gt_masks):
        edge_mask = map(self.get_edge_gt_single, gt_masks)
        edge_mask = torch.cat(list(edge_mask))
        return edge_mask

    def get_edge_gt_single(self, gt_masks_single):
        edge_mask = torch.zeros_like(gt_masks_single)
        contours, _ = cv2.findContours(np.array(gt_masks_single[0].cpu().detach().numpy(), 'uint8'),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            for i in range(len(contour)):
                pos = contour[i][0]
                edge_mask[0][pos[1]][pos[0]] = 1
        return edge_mask[None]

    def get_edge(self, gt_masks):

        gt_masks = gt_masks[:, None, :, :]
        # gt_masks_down = F.interpolate(gt_masks, scale_factor=0.5)
        # # tic = time.time()
        # edge_gt = self.get_edge_gt(gt_masks_down)
        edge_gt = F.conv2d(gt_masks, self.weight, padding=1)
        edge_gt_new = torch.zeros_like(edge_gt)
        edge_gt_id = ((edge_gt > 99) & (edge_gt < 104))
        edge_gt_new[edge_gt_id] = 1
        # img1 = edge_gt_new.cpu().numpy()
        edge_gt_w_id = (edge_gt_new == 1)

        # edge_gt_posw = edge_gt_w_id.cpu().numpy().sum(axis=(1,2,3))
        # edge_gt_nw = np.ones_like(edge_gt_posw)*(edge_gt_new.shape[2]*edge_gt_new.shape[3]) - edge_gt_posw
        # edge_gt_pnw = np.stack([edge_gt_posw, edge_gt_nw],1)
        # edge_gt_pnw = 1/np.log(edge_gt_pnw/(edge_gt_new.shape[2]*edge_gt_new.shape[3])+1.2)
        edge_gt_w = torch.ones_like(edge_gt)
        edge_gt_w[edge_gt_w_id] = 5
        # img5 = edge_gt_w.cpu().numpy()
        return edge_gt_new, edge_gt_w

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        
        return loss

    @force_fp32(apply_to=('edge_pred', ))
    def loss_e(self, edge_pred, edge_targets, edge_targets_w):
        # edge_pred_sig = edge_pred.sigmoid()
        # img = edge_pred.cpu().detach().numpy()
        # img2 = edge_pred_sig.cpu().detach().numpy()
        loss = dict()
        loss_edge = self.loss_edge(edge_pred, edge_targets, edge_targets_w)
        loss['loss_edge'] = loss_edge
        return loss


    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms
