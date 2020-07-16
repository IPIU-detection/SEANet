import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, delta2bbox, force_fp32,
                        multiclass_nms, distance2bbox, bbox_goverlaps)
from mmdet.core.bbox.bbox_target import fc_target, get_points
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import numpy as np



INF = 1e8

@HEADS.register_module
class FCBBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_iou=dict(
                     type='SmoothL1Loss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(FCBBoxHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.reg_class_agnostic = False
        self.target_means = [0., 0., 0., 0.],
        self.target_stds = [0.1, 0.1, 0.2, 0.2]


        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.iou_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            chn2 = self.in_channels*2 if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.iou_convs.append(
                ConvModule(
                    chn2,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.fcos_bbox_iou = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.iou_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    @auto_fp16()
    def forward(self, x):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        # centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        # bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        bbox_pred = self.fcos_reg(reg_feat).float().exp()  # 512,4, 7,7

        bbox_feat_cat = torch.cat((x, reg_feat), dim=1)
        for iou_layer in self.iou_convs:
            bbox_feat_cat = iou_layer(bbox_feat_cat)
        bbox_iou = self.fcos_bbox_iou(bbox_feat_cat)

        return cls_score, bbox_pred, bbox_iou

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]  # ori sizes
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]  # [x1, y1, x2, y2]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]  # True cls
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = fcos_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg)
        return cls_reg_targets


    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_iou'))
    def loss(self,
             gt_bboxes,
             cls_scores,
             bbox_preds,
             bbox_iou,
             labels,
             label_weight,
             bbox_targets,
             bbox_weights,
             points,
             reduction_override=None):
        assert len(cls_scores) == len(bbox_preds)

        num_imgs = cls_scores.size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        flatten_bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)
        flatten_bbox_iou = bbox_iou.permute(0, 2, 3, 1).reshape(-1, 1)
        flatten_labels = labels.reshape(-1)
        flatten_bbox_targets = bbox_targets.reshape(-1, 4)
        # repeat points to align with bbox_preds
        flatten_points = points.reshape(-1, 2)

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_iou = flatten_bbox_iou[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            # pos_gt_bbox = gt_bboxes[0][pos_inds]
            # a = pos_gt_bbox.cpu().detach().numpy()
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                avg_factor=num_pos)
            bbox_iou_targets = bbox_goverlaps(pos_decoded_bbox_preds,
                                             pos_decoded_target_preds, is_aligned=True).clamp(min=1e-6)[:, None]
            loss_bbox_iou = self.loss_iou(pos_bbox_iou,
                                                 bbox_iou_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_iou = pos_bbox_iou.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_bbox_iou)

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_iou'))
    def get_det_bboxes(self,
                       rois,
                       cls_scores,  # 1000. 81, 7, 7
                       bbox_preds,  # 1000, 4, 7, 7
                       bbox_iou,
                       img_shape,
                       scale_factor=None,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_scores, list):
            cls_score = sum(cls_scores) / float(len(cls_scores))
        # scores = F.softmax(cls_scores, dim=1) if cls_scores is not None else None
        points = get_points(rois[:, 1:], bbox_preds.size(-1), bbox_preds.dtype, bbox_preds.device)
        if bbox_preds is not None:
            bboxes = bbox_preds.new_zeros(bbox_preds.size(0), bbox_preds.size(1))
            scores = cls_scores.new_zeros(cls_scores.size(0), cls_scores.size(1))
            for img_id in range(len(cls_scores)):
                cls_score_list = cls_scores[img_id].detach()
                bbox_pred_list = bbox_preds[img_id].detach()
                bbox_iou_pred_list = bbox_iou[img_id].detach()
                points_list = points[img_id].detach()
                # img_shape = img_metas[img_id]['img_shape']
                # scale_factor = img_metas[img_id]['scale_factor']
                det_bboxes, det_labels = self.get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    bbox_iou_pred_list,
                                                    points_list,
                                                    img_shape,
                                                    cfg, rescale)
                bboxes[img_id] = det_bboxes
                scores[img_id] = det_labels
            # scores = F.softmax(label_list, dim=1) if cls_scores is not None else None
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([padding, scores], dim=1)
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    def get_bboxes_single(self,
                          cls_score,
                          bbox_pred,
                          bbox_iou,
                          points,
                          img_shape,
                          cfg,
                          rescale=False,
                          scale_factor=None):
        # assert len(cls_score) == len(bbox_pred)
        # mlvl_bboxes = []
        # mlvl_scores = []
        # mlvl_bboxiou = []
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(1, 2, 0).reshape(
            -1, self.cls_out_channels).sigmoid()
        bbox_iou = bbox_iou.permute(1, 2, 0).reshape(-1).sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        points = points[0]
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = (scores * bbox_iou[:, None]).max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            points = points[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            bbox_iou = bbox_iou[topk_inds]
        bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)

        scores_cpu = np.array(scores.cpu().detach())
        # bbox_iou = np.array(bbox_iou.cpu().detach())
        # bboxes = np.array(bboxes.cpu().detach())

        max_cls = np.unravel_index(scores_cpu.argmax(), scores_cpu.shape)[1]
        scores_iou = scores[:, max_cls] * bbox_iou
        bbox_ind = scores_iou.argmax()
        det_bboxes = bboxes[bbox_ind]
        det_labels = scores[bbox_ind]*bbox_iou[bbox_ind]

        # mlvl_bboxes.append(bboxes)
        # mlvl_scores.append(scores)
        # mlvl_bboxiou.append(bbox_iou)
        # mlvl_bboxes = torch.cat(mlvl_bboxes)
        # if rescale:
        #     mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        # mlvl_scores = torch.cat(mlvl_scores)  # 49, 81
        # padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)  # 49, 1
        # mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)  # 49, 82
        # mlvl_bboxiou = torch.cat(mlvl_bboxiou)
        #
        # #
        # det_bboxes, det_labels = multiclass_nms(
        #     mlvl_bboxes,
        #     mlvl_scores,
        #     cfg.score_thr,
        #     cfg.nms,
        #     cfg.max_per_img,
        #     score_factors=mlvl_bboxiou)

        return det_bboxes, det_labels


    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
