import sys
import copy
import numba
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.init import kaiming_normal_
from .target_assigner.center_assigner import CenterAssigner
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ..model_utils import centernet_box_utils
from ...utils import loss_utils
from ...ops.dcn import DeformConv
from ...ops.iou3d_nms import iou3d_nms_cuda
from ...ops.center_ops import center_ops_cuda


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict


class CenterHead_(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.post_cfg = model_cfg.TEST_CONFIG
        self.in_channels = input_channels
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        self.num_classes = [len(t["class_names"]) for t in model_cfg.TASKS]
        self.class_names = [t["class_names"] for t in model_cfg.TASKS]

        self.code_weights = model_cfg.LOSS_CONFIG.code_weights
        self.weight = model_cfg.LOSS_CONFIG.weight # weight between hm loss and loc loss

        self.dataset = model_cfg.DATASET
        self.box_n_dim = 9 if self.dataset == 'nuscenes' else 7

        self.encode_background_as_zeros = True
        self.use_sigmoid_score = True
        self.no_log = False
        self.use_direction_classifier = False
        self.bev_only = True if model_cfg.MODE == "bev" else False

        # a shared convolution
        share_conv_channel = model_cfg.PARAMETERS.share_conv_channel
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.common_heads = model_cfg.PARAMETERS.common_heads
        self.init_bias = model_cfg.PARAMETERS.init_bias
        self.tasks = nn.ModuleList()

        self.use_dcn = model_cfg.USE_DCN
        for num_cls in self.num_classes:
            heads = copy.deepcopy(self.common_heads)
            if not self.use_dcn:
                heads.update(dict(hm=(num_cls, 2)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=self.init_bias, final_kernel=3, directional_classifier=False)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=self.init_bias, final_kernel=3,
                                directional_classifier=False)
                )
        self.target_assigner = CenterAssigner(
            model_cfg.TARGET_ASSIGNER_CONFIG,
            num_classes = sum(self.num_classes),
            no_log = self.no_log,
            grid_size = grid_size,
            pc_range = point_cloud_range,
            voxel_size = voxel_size
        )

        self.forward_ret_dict = {}
        self.build_losses()

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets_v2(
            gt_boxes
        )

        return targets_dict

    def forward(self, data_dict):
        multi_head_features = []
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv(spatial_features_2d)
        for task in self.tasks:
            multi_head_features.append(task(spatial_features_2d))

        self.forward_ret_dict['multi_head_features'] = multi_head_features

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            data_dict = self.generate_predicted_boxes(data_dict)

        return data_dict

    def build_losses(self):
        self.add_module(
            'crit',
            loss_utils.FocalLossCenterNet()
        )
        self.add_module(
            'crit_reg',
            loss_utils.RegLossCenterNet()
        )
        return

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        tb_dict = {}
        pred_dicts = self.forward_ret_dict['multi_head_features']
        center_loss = []
        self.forward_ret_dict['pred_box_encoding'] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self._sigmoid(pred_dict['hm'])
            hm_loss = self.crit(pred_dict['hm'], self.forward_ret_dict['heatmap'][task_id])

            target_box_encoding = self.forward_ret_dict['box_encoding'][task_id]
            # nuscense encoding format [x, y, z, w, l, h, sinr, cosr, vx, vy]

            if self.dataset == 'nuscenes':
                pred_box_encoding = torch.cat([
                    pred_dict['reg'],
                    pred_dict['height'],
                    pred_dict['dim'],
                    pred_dict['rot'],
                    pred_dict['vel']
                ], dim = 1).contiguous() # (B, 10, H, W)
            else:
                pred_box_encoding = torch.cat([
                    pred_dict['reg'],
                    pred_dict['height'],
                    pred_dict['dim'],
                    pred_dict['rot']
                ], dim = 1).contiguous() # (B, 8, H, W)

            self.forward_ret_dict['pred_box_encoding'][task_id] = pred_box_encoding

            box_loss = self.crit_reg(
                pred_box_encoding,
                self.forward_ret_dict['mask'][task_id],
                self.forward_ret_dict['ind'][task_id],
                target_box_encoding
            )

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            loss = hm_loss + self.weight * loc_loss

            tb_key = 'task_' + str(task_id) + '/'

            if self.dataset == 'nuscenes':
                tb_dict.update({
                    tb_key + 'loss': loss.item(), tb_key + 'hm_loss': hm_loss.item(), tb_key + 'loc_loss': loc_loss.item(),
                    tb_key + 'x_loss': box_loss[0].item(), tb_key + 'y_loss': box_loss[1].item(), tb_key + 'z_loss': box_loss[2].item(),
                    tb_key + 'w_loss': box_loss[3].item(), tb_key + 'l_loss': box_loss[4].item(), tb_key + 'h_loss': box_loss[5].item(),
                    tb_key + 'sin_r_loss': box_loss[6].item(), tb_key + 'cos_r_loss': box_loss[7].item(),
                    tb_key + 'vx_loss': box_loss[8].item(), tb_key + 'vy_loss': box_loss[9].item(),
                    tb_key + 'num_positive': self.forward_ret_dict['mask'][task_id].float().sum(),
                })
            else:
                tb_dict.update({
                    tb_key + 'loss': loss.item(), tb_key + 'hm_loss': hm_loss.item(),
                    tb_key + 'loc_loss': loc_loss.item(),
                    tb_key + 'x_loss': box_loss[0].item(), tb_key + 'y_loss': box_loss[1].item(),
                    tb_key + 'z_loss': box_loss[2].item(),
                    tb_key + 'w_loss': box_loss[3].item(), tb_key + 'l_loss': box_loss[4].item(),
                    tb_key + 'h_loss': box_loss[5].item(),
                    tb_key + 'sin_r_loss': box_loss[6].item(), tb_key + 'cos_r_loss': box_loss[7].item(),
                    tb_key + 'num_positive': self.forward_ret_dict['mask'][task_id].float().sum(),
                })
            center_loss.append(loss)

        return sum(center_loss), tb_dict

    def _double_flip_process(self, pred_dict, batch_size):
        for k in pred_dict.keys():
            # transform the prediction map back to their original coordinate befor flipping
            # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
            # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is
            # X and Y flip pointcloud(x=-x, y=-y).
            # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
            # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
            # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
            _, C, H, W = pred_dict[k].shape
            pred_dict[k] = pred_dict[k].reshape(int(batch_size), 4, C, H, W)
            pred_dict[k][:, 1] = torch.flip(pred_dict[k][:, 1], dims=[2])
            pred_dict[k][:, 2] = torch.flip(pred_dict[k][:, 2], dims=[3])
            pred_dict[k][:, 3] = torch.flip(pred_dict[k][:, 3], dims=[2, 3])

        # batch_hm = pred_dict['hm'].sigmoid_() inplace may cause errors
        batch_hm = pred_dict['hm'].sigmoid()

        batch_reg = pred_dict['reg']
        batch_hei = pred_dict['height']
        if not self.no_log:
            batch_dim = torch.exp(pred_dict['dim'])
        else:
            batch_dim = pred_dict['dim']

        batch_hm = batch_hm.mean(dim=1)
        batch_hei = batch_hei.mean(dim=1)
        batch_dim = batch_dim.mean(dim=1)

        # y = -y reg_y = 1-reg_y
        batch_reg[:, 1, 1] = 1 - batch_reg[:, 1, 1]
        batch_reg[:, 2, 0] = 1 - batch_reg[:, 2, 0]
        batch_reg[:, 3, 0] = 1 - batch_reg[:, 3, 0]
        batch_reg[:, 3, 1] = 1 - batch_reg[:, 3, 1]
        batch_reg = batch_reg.mean(dim=1)

        batch_rots = pred_dict['rot'][:, :, 0:1]
        batch_rotc = pred_dict['rot'][:, :, 1:2]

        # first yflip
        # y = -y theta = pi -theta ???
        # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta) ???
        # batch_rotc[:, 1] = -batch_rotc[:, 1] ???

        # y = -y; theta = -theta
        # sin(-theta) = -sin(theta); cos(-theta) = cos(theta)
        batch_rots[:, 1] = -batch_rots[:, 1]

        # then xflip x = -x theta = 2pi - theta ???
        # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta) ???
        # batch_rots[:, 2] = -batch_rots[:, 2] ???

        # x = -x; theta = pi - theta
        # sin(pi - theta) = sin(theta); cos(pi - theta) = -cos(theta)
        batch_rotc[:, 2] = -batch_rotc[:, 2]

        # double flip
        batch_rots[:, 3] = -batch_rots[:, 3]
        batch_rotc[:, 3] = -batch_rotc[:, 3]

        batch_rotc = batch_rotc.mean(dim=1)
        batch_rots = batch_rots.mean(dim=1)

        if self.dataset == 'nuscenes':
            batch_vel = pred_dict['vel']
            # flip vy
            batch_vel[:, 1, 1] = - batch_vel[:, 1, 1]
            # flip vx
            batch_vel[:, 2, 0] = - batch_vel[:, 2, 0]
            batch_vel[:, 3] = - batch_vel[:, 3]
            batch_vel = batch_vel.mean(dim=1)
        else:
            batch_vel = None

        batch_dir_preds = [None] * batch_size
        return batch_hm, batch_rots, batch_rotc, batch_hei, batch_dim, batch_dir_preds, batch_vel, batch_reg

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    @numba.jit(nopython=True)
    def circle_nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        scores = dets[:, 2]
        order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
        ndets = dets.shape[0]
        suppressed = np.zeros((ndets), dtype=np.int32)
        keep = []
        for _i in range(ndets):
            i = order[_i]  # start with highest score box
            if suppressed[i] == 1:  # if any box have enough iou with this, remove it
                continue
            keep.append(i)
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                # calculate center distance between i and j box
                dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

                # ovr = inter / areas[j]
                if dist <= thresh:
                    suppressed[j] = 1
        return keep

    def _circle_nms(self, boxes, min_radius, post_max_size=83):
        """
        NMS according to center distance
        """
        keep = np.array(self.circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
        keep = torch.from_numpy(keep).long().to(boxes.device)
        return keep

    def _rotate_nms(self, boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        """
        :param boxes: (N, 5) [x1, y1, x2, y2, ry]
        :param scores: (N)
        :param thresh:
        :return:
        """
        # areas = (x2 - x1) * (y2 - y1)
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()

        keep = torch.LongTensor(boxes.size(0))
        num_out = center_ops_cuda.center_rotate_nms_gpu(boxes, keep, thresh)
        selected = order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected

    def _nms_gpu_3d(self, boxes, scores, thresh, pre_maxsize=None, post_max_size = None):
        """
        :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert boxes.shape[1] == 7
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()
        keep = torch.LongTensor(boxes.size(0))
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
        selected =  order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected

    def _boxes3d_to_bevboxes_lidar_torch(self, boxes3d):
        """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
            boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
        """
        boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

        cu, cv = boxes3d[:, 0], boxes3d[:, 1]

        half_w, half_l = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
        boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
        boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
        boxes_bev[:, 4] = boxes3d[:, -1]
        return boxes_bev

    @torch.no_grad()
    def proposal_layer(self, heat, rots, rotc, hei, dim, vel, reg = None,
                       post_center_range=None, score_threshold=None, cfg=None, raw_rot=False, task_id=-1):

        assert self.encode_background_as_zeros is True
        assert self.use_sigmoid_score is True
        batch, cat, _, _ = heat.size()
        nms_cfg = cfg.nms.train if self.training else cfg.nms.test
        K = nms_cfg.nms_pre_max_size # topK selected
        maxpool = nms_cfg.get('max_pool_nms', False) or (nms_cfg.get('circle_nms', False) and (nms_cfg.min_radius[task_id] == -1))
        use_circle_nms = nms_cfg.get('circle_nms', False) and (nms_cfg.min_radius[task_id] != -1)
        if maxpool:
            heat = self._nms(heat)
        scores, inds, clses, ys, xs = self._topk(heat, K=K)
        assert reg is not None
        reg = self._transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        assert raw_rot is False
        rots = self._transpose_and_gather_feat(rots, inds)
        rots = rots.view(batch, K, 1)
        rotc = self._transpose_and_gather_feat(rotc, inds)
        rotc = rotc.view(batch, K, 1)
        rot = torch.atan2(rots, rotc)
        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, K, 1)
        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, K, 3)
        # class label
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)
        # center location
        pc_range = cfg.pc_range
        xs = xs.view(batch, K, 1) * cfg.out_size_factor * cfg.voxel_size[0] + pc_range[0]
        ys = ys.view(batch, K, 1) * cfg.out_size_factor * cfg.voxel_size[1] + pc_range[1]

        if self.dataset == 'nuscenes':
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, K, 2)
            # vel after rot
            final_box_preds = torch.cat(
                [xs, ys, hei, dim, rot, vel], dim=2
            )
        else:
            final_box_preds = torch.cat(
                [xs, ys, hei, dim, rot], dim=2
            )

        final_scores = scores
        final_preds = clses

        # restrict center range
        assert post_center_range is not None
        post_center_range = torch.tensor(post_center_range).to(final_box_preds.device)
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        # use score threshold
        assert score_threshold is not None
        thresh_mask = final_scores > score_threshold
        mask &= thresh_mask

        predictions_dicts = []
        for i in range(batch):
            cmask = mask[i, :]
            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]

            # circle nms
            if use_circle_nms:
                centers = boxes3d[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                keep = self._circle_nms(boxes, min_radius=nms_cfg.min_radius[task_id], post_max_size=nms_cfg.post_max_size)

                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]

            # rotate nms
            elif nms_cfg.get('use_rotate_nms', False):
                assert not maxpool
                top_scores = scores
                if top_scores.shape[0] != 0:
                    boxes_for_nms =self._boxes3d_to_bevboxes_lidar_torch(boxes3d)
                    selected = self._rotate_nms(boxes_for_nms, top_scores,
                                                thresh=nms_cfg.nms_iou_threshold,
                                                pre_maxsize=nms_cfg.nms_pre_max_size,
                                                post_max_size=nms_cfg.nms_post_max_size
                                               )
                else:
                    selected = []
                boxes3d = boxes3d[selected]
                labels = labels[selected]
                scores = scores[selected]

            # iou 3d nms
            elif nms_cfg.get('use_iou_3d_nms', False):
                assert not maxpool
                top_scores = scores
                if top_scores.shape[0] != 0:
                    selected = self._nms_gpu_3d(boxes3d[:, :7], top_scores,
                                                thresh=nms_cfg.nms_iou_threshold,
                                                pre_maxsize=nms_cfg.nms_pre_max_size,
                                                post_max_size=nms_cfg.nms_post_max_size
                                               )
                else:
                    selected = []
                boxes3d = boxes3d[selected]
                labels = labels[selected]
                scores = scores[selected]

            predictions_dict = {
                "boxes": boxes3d,
                "scores": scores,
                "labels": labels.long()
            }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    @torch.no_grad()
    def generate_predicted_boxes(self, data_dict):
        """
        Generate box predictions with decode, topk and circular_nms
        For single-stage-detector, another post-processing (nms) is needed
        For two-stage-detector, no need for proposal layer in roi_head
        Returns:
        """
        double_flip = not self.training and self.post_cfg.get('double_flip', False)
        post_center_range = self.post_cfg.post_center_limit_range
        pred_dicts = self.forward_ret_dict['multi_head_features']

        task_box_preds = {}
        task_score_preds = {}
        task_label_preds = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            batch_size = pred_dict['hm'].shape[0]
            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                batch_hm, batch_rots, batch_rotc, batch_hei, batch_dim, batch_dir_preds, batch_vel, batch_reg = \
                    self._double_flip_process(pred_dict, batch_size)
                # convert data_dict format
                data_dict['batch_size'] = batch_size
            else:
                # batch_hm = pred_dict['hm'].sigmoid_() inplace may cause errors
                batch_hm = pred_dict['hm'].sigmoid()
                batch_reg = pred_dict['reg']
                batch_hei = pred_dict['height']

                if not self.no_log:
                    batch_dim = torch.exp(pred_dict['dim'])
                    # add clamp for good init, otherwise we will get inf with exp
                    batch_dim = torch.clamp(batch_dim, min=0.001, max=30)
                else:
                    batch_dim = pred_dict['dim']
                batch_rots = pred_dict['rot'][:, 0].unsqueeze(1)
                batch_rotc = pred_dict['rot'][:, 1].unsqueeze(1)

                if self.dataset == 'nuscenes':
                    batch_vel = pred_dict['vel']
                else:
                    batch_vel = None

            #decode
            boxes = self.proposal_layer(
                batch_hm,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                post_center_range=post_center_range,
                score_threshold=self.post_cfg.score_threshold,
                cfg=self.post_cfg,
                task_id=task_id
            )
            task_box_preds[task_id] = [box['boxes'] for box in boxes]
            task_score_preds[task_id] = [box['scores'] for box in boxes]
            task_label_preds[task_id] = [box['labels'] for box in boxes] #labels are local here

        pred_dicts = []
        batch_size = len(task_box_preds[0])
        rois, roi_scores, roi_labels = [], [], []
        nms_cfg = self.post_cfg.nms.train if self.training else self.post_cfg.nms.test
        num_rois = nms_cfg.nms_post_max_size * len(self.class_names)
        for batch_idx in range(batch_size):
            offset = 1 # class label start from 1
            final_boxes, final_scores, final_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                final_boxes.append(task_box_preds[task_id][batch_idx])
                final_scores.append(task_score_preds[task_id][batch_idx])
                # convert to global labels
                final_global_label = task_label_preds[task_id][batch_idx] + offset
                offset += len(class_name)
                final_labels.append(final_global_label)

            final_boxes = torch.cat(final_boxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)

            roi = final_boxes.new_zeros(num_rois, final_boxes.shape[-1])
            roi_score = final_scores.new_zeros(num_rois)
            roi_label = final_labels.new_zeros(num_rois)
            num_boxes = final_boxes.shape[0]
            roi[:num_boxes] = final_boxes
            roi_score[:num_boxes] = final_scores
            roi_label[:num_boxes] = final_labels
            rois.append(roi)
            roi_scores.append(roi_score)
            roi_labels.append(roi_label)

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels
            }
            pred_dicts.append(record_dict)

        data_dict['pred_dicts'] = pred_dicts
        data_dict['rois'] = torch.stack(rois, dim = 0)
        data_dict['roi_scores'] = torch.stack(roi_scores, dim = 0)
        data_dict['roi_labels'] = torch.stack(roi_labels, dim = 0)
        data_dict['has_class_labels'] = True  # Force to be true
        data_dict.pop('batch_index', None)
        return data_dict


"""
BASIC BUILDING BLOCKS
"""
class Sequential(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SepHead(nn.Module):
    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            name="",
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            directional_classifier=False,
            **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                             kernel_size=final_kernel, stride=1,
                             padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

        assert directional_classifier is False, "Doesn't work well with nuScenes in my experiments, please open a pull request if you are able to get it work. We really appreciate contribution for this."

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def init_weights(self):
        pass

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x


class DCNSepHead(nn.Module):
    def __init__(
            self,
            in_channels,
            num_cls,
            heads,
            head_conv=64,
            name="",
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            directional_classifier=False,
            **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)

        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)

        # heatmap prediction head
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                      kernel_size=3, stride=1,
                      padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)

    def forward(self, x):
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret
