from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from typing import List, Tuple
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from mmseg.models.builder import MODELS

import torch.nn.functional as F
from mmengine.structures import InstanceData
from ...utils.dist_utils import reduce_mean
from ...utils.point_sample import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from ..losses.PCLoss import PCLoss
import numpy as np
from sklearn.cluster import KMeans
import os


@MODELS.register_module()
class ReinMask2FormerHead(Mask2FormerHead):
    def __init__(self, replace_query_feat=False, loss_pcl: ConfigType = dict(type='PCLoss', loss_weight=0.01),
                 **kwargs):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)
        self.index_test_img = 0
        self.save_path = './mask_pred/'
        self.loss_pcl = MODELS.build(loss_pcl)
        self.text_features = None
        self.learnable_token = None

    def forward(
            self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, query_embed, learnable_token, text_features = x
        self.text_features = text_features
        self.learnable_token = learnable_token
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        mb, mc, mh, mw = mask_pred.shape
        mask_pred_view = mask_pred.view(mb, mc, -1)
        global_learnable_token = learnable_token.mean(0)
        proto_att = torch.einsum("cm,bmn->bcn", global_learnable_token.t(), mask_pred_view)
        weighted_mask_pred = torch.einsum("mc,bcn->bmn", global_learnable_token, proto_att).view(mb, mc, mh,
                                                                                                 mw).contiguous()
        final_mask_pred = mask_pred + weighted_mask_pred
        final_mask_pred = final_mask_pred.cuda()

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(final_mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                -2:
                ],
            )

            mb, mc, mh, mw = mask_pred.shape
            mask_pred_view = mask_pred.view(mb, mc, -1)
            global_learnable_token = learnable_token.mean(0)
            proto_att = torch.einsum("cm,bmn->bcn", global_learnable_token.t(), mask_pred_view)
            weighted_mask_pred = torch.einsum("mc,bcn->bmn", global_learnable_token, proto_att).view(mb, mc, mh,
                                                                                                     mw).contiguous()
            final_mask_pred = mask_pred + weighted_mask_pred
            final_mask_pred = final_mask_pred.cuda()

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(final_mask_pred)
        return cls_pred_list, mask_pred_list

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)

        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        mask_point_preds = mask_point_preds.reshape(-1)
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:

        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        global_learnable_token = self.learnable_token.mean(0)

        loss_pcl = self.loss_pcl(self.text_features.to(torch.float32), global_learnable_token, all_mask_preds,
                                 all_cls_scores)

        losses['loss_pcl'] = loss_pcl

        return losses
