import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from sklearn.cluster import KMeans
import numpy as np


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)


@MODELS.register_module()
class PCLoss(nn.Module):
    def __init__(self,
                 loss_weight=0.01,
                 loss_name='loss_pcl'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.out_indices = [7, 11, 15, 23]

    def forward(self, text_features, global_learnable_token, all_mask_preds, all_cls_scores):
        assert not text_features.requires_grad
        assert global_learnable_token.requires_grad

        global_learnable_token_np = global_learnable_token.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=19, random_state=0).fit(global_learnable_token_np)
        centroids = kmeans.cluster_centers_
        centroids_tensor = torch.from_numpy(centroids).cuda()

        len_cls = len(all_cls_scores)
        len_mask = len(all_mask_preds)


        loss_ce = 0
        for i in range(len_mask):
            mask_pred_results = all_mask_preds[i]

            mask_pred_results = F.interpolate(
                mask_pred_results, size=(32, 32), mode='bilinear', align_corners=False)
            mb, mc, mh, mw = mask_pred_results.shape
            mask_pred_results = mask_pred_results.view(mb * mc, -1)

            token_logits = mask_pred_results.mm(
                centroids_tensor.permute(1, 0).contiguous())  
            text_logits = mask_pred_results.mm(text_features.permute(1, 0).contiguous())

            loss_ce += kl_categorical(token_logits, text_logits)

        loss_ce = loss_ce / len_mask

        loss = loss_ce * 0.01

        return loss

    @property
    def loss_name(self):
        return self._loss_name