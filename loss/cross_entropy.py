import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class EdgeCrossEntropy(nn.Module):
    def __init__(self):
        super(EdgeCrossEntropy, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y, dist, coef=0.5):
        edge_pre = self.maxpool(dist)
        edge_dist_1 = edge_pre[:, 1, :, :] - dist[:, 1, :, :]
        edge_dist_0 = dist[:, 0, :, :] - edge_pre[:, 0, :, :]

        edge_gt = self.maxpool(y)
        edge_gt_1 = edge_gt[:, 1, :, :] - y[:, 1, :, :]
        edge_gt_0 = y[:, 0, :, :] - edge_gt[:, 0, :, :]

        edge_dist = torch.stack((edge_dist_0, edge_dist_1), axis=1)
        edge_gt = torch.stack((edge_gt_0, edge_gt_1), axis=1)
        
        loss = self.bce_loss(y, dist) + coef * self.bce_loss(edge_gt, edge_dist)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, alpha=0.5, gamma=2, smooth=1):
        inputs = inputs.float()
        ce_loss = F.cross_entropy(inputs, targets, reduction="mean", weight=self.weight)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** gamma) * ce_loss).mean()
        
