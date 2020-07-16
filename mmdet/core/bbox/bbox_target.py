import torch

from ..utils import multi_apply
from .transforms import bbox2delta

def get_points(pre_bboxes_list, featmap_size, dtype, device):
    """Get points according to feature map sizes.

    Args:
        featmap_size : 7.
        dtype (torch.dtype): Type of points.
        device (torch.device): Device of points.

    Returns:
        tuple: points of each image.
    """
    mlvl_points = []
    for i in range(len(pre_bboxes_list)):
        mlvl_points.append(
            get_points_single(pre_bboxes_list[i], featmap_size,
                                   dtype, device))
    #
    # mlvl_points = multi_apply(
    #     get_points_single,
    #     pre_bboxes_list,
    #     featmap_size=featmap_size,
    #     dtype=dtype,
    #     device=device)
    return mlvl_points

def get_points_single(pre_bbox, featmap_size, dtype, device):
    x1, y1, x2, y2 = pre_bbox[0], pre_bbox[1], pre_bbox[2], pre_bbox[3]
    if x1>=x2-1:
        x2 = x1 + 1
    if y1>=y2-1:
        y2 = y1 + 1
    h, w = y2 - y1, x2 - x1

    x_range = torch.floor(torch.arange(
        x1 + w/(featmap_size*2), x2, w/featmap_size, dtype=dtype, device=device))
    y_range = torch.floor(torch.arange(
        y1 + h/(featmap_size*2), y2, h/featmap_size, dtype=dtype, device=device))
    a = x_range.size(0)
    if x_range.size(0)<7 or y_range.size(0)<7:
        b = 1
    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack(
        (x.reshape(-1), y.reshape(-1)), dim=-1)[None]
    return points


def fc_target(pos_bboxes,
                neg_bboxes,
                pos_gt_bboxes,
                pos_gt_labels,
                cfg,
                featmap_size=7,
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights, points = multi_apply(
        fcos_target_single,
        pos_bboxes,
        neg_bboxes,
        pos_gt_bboxes,
        pos_gt_labels,
        featmap_size=featmap_size,
        cfg=cfg)

    if concat:
        labels = torch.cat(labels, 0)
        points = torch.cat(points, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights, points


def fcos_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes, # posnum, 4
                       pos_gt_labels,
                       featmap_size,
                       cfg):  # gt_bboxes [x1, y1, x2, y2]

    # points list(tensor) posnum, 49, 2
    points = get_points(pos_bboxes, featmap_size, pos_gt_bboxes.dtype, pos_gt_bboxes.device)
    points = torch.cat(points, 0)
    # get labels and bbox_targets of each image
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    label_weights = pos_gt_bboxes.new_zeros(num_samples)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    bbox_targets = pos_bboxes.new_zeros(num_samples, featmap_size**2, 4)
    point_all = pos_bboxes.new_zeros(num_samples, featmap_size**2, 2)
    if num_pos > 0:
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_pos:] = 1.0
    
    xs, ys = points[..., 0], points[..., 1]
    pos_gt_bboxes = pos_gt_bboxes[:, None, :].expand(num_pos, xs.size(-1), 4)
    left = xs - pos_gt_bboxes[..., 0]
    right = pos_gt_bboxes[..., 2] - xs
    top = ys - pos_gt_bboxes[..., 1]
    bottom = pos_gt_bboxes[..., 3] - ys
    bbox_targets[:num_pos] = torch.stack((left, top, right, bottom), -1) # pos_num, 49, 4
    point_all[:num_pos] = points

    # condition1: inside a gt bbox
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    labels = torch.clone(inside_gt_bbox_mask).long()
    pos_label_ind = labels[:num_pos] > 0
    # a = pos_label_ind.cpu().detach().numpy()
    # labels[pos_label_ind] = pos_gt_labels
    pos_gt_labels = pos_gt_labels[:, None].expand(num_pos, 49)
    # b = pos_gt_labels.cpu().detach().numpy()
    labels[:num_pos] = pos_gt_labels *pos_label_ind.long()

    return labels, label_weights, bbox_targets, bbox_weights, point_all

def centerness_target(pos_bbox_targets):
    # only calculate pos centerness targets, otherwise there may be nan
    left_right = pos_bbox_targets[:, [0, 2]]
    top_bottom = pos_bbox_targets[:, [1, 3]]
    centerness_targets = (
        left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness_targets)


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
