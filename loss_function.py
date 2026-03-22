import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLoss(nn.Module):
    def __init__(self):
        super(PromptLoss, self).__init__()

    def forward(self, out1, out2, label):
        x_embed_norm1_low = out1['x_embed_norm1_low'].unsqueeze(1)
        x_embed_norm2_low = out2['x_embed_norm1_low'].unsqueeze(1)
        x_embed_norm1_high = out1['x_embed_norm1_high'].unsqueeze(1)
        x_embed_norm2_high = out2['x_embed_norm1_high'].unsqueeze(1)
        batched_key_norm1 = out1['batched_key_norm1']
        batched_key_norm2 = out2['batched_key_norm1']
        batched_key_norm1_high = out1['batched_key_norm1_high']
        batched_key_norm2_high = out2['batched_key_norm1_high']
        batch_size = out1['batch_size']

        sim_x1_pool_low = torch.bmm(batched_key_norm1, x_embed_norm1_low.transpose(-1, -2))
        loss_pool1 = torch.pow(1 - sim_x1_pool_low, 2)
        sim_x2_pool_low = torch.bmm(batched_key_norm2, x_embed_norm2_low.transpose(-1, -2))
        loss_pool2 = torch.pow(1 - sim_x2_pool_low, 2)
        sim_x1_pool_high = torch.bmm(batched_key_norm1_high, x_embed_norm1_high.transpose(-1, -2))
        loss_pool_high1 = torch.pow(1 - sim_x1_pool_high, 2)
        sim_x2_pool_high = torch.bmm(batched_key_norm2_high, x_embed_norm2_high.transpose(-1, -2))
        loss_pool_high2 = torch.pow(1 - sim_x2_pool_high, 2)
        loss_embed_pool = loss_pool1.sum() + loss_pool2.sum() + loss_pool_high1.sum() + loss_pool_high2.sum()

        sim_low_high_pool_x1 = torch.bmm(batched_key_norm1, batched_key_norm1_high.transpose(-1, -2))
        sim_low_high_pool_x2 = torch.bmm(batched_key_norm2, batched_key_norm2_high.transpose(-1, -2))
        loss_wave1 = torch.pow(1 + sim_low_high_pool_x1, 2)
        loss_wave2 = torch.pow(1 + sim_low_high_pool_x2, 2)
        loss_wave = loss_wave1.sum() + loss_wave2.sum()

        total_loss = (loss_embed_pool + loss_wave) / batch_size
        return total_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        x0 = F.normalize(x0)
        x1 = F.normalize(x1)
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + 1e-6)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss)
        return loss / 2.0 / x0.size()[0]
