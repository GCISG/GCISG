import torch
from torch import nn
import torch.nn.functional as F


class GuidanceLoss(nn.Module):
    def __init__(self, scale, *args, **kwargs):
        super().__init__()
        self.scale = scale

    def on_forward(self, *args, **kwargs):
        pass

    
    def self_attention_pool(self, f):
        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        batch_size = f.shape[0]
        gap = torch.flatten(avgpool(f), start_dim=1)
        attention = torch.einsum("nchw,nc->nhw", [f, gap])
        attention /= torch.einsum("nhw->n", [attention]).view(batch_size, 1, 1)
        features_with_attention = torch.einsum("nchw,nhw->nchw", [f, attention])
        return torch.einsum("nchw->nc", [features_with_attention])
    

    def forward(self, q_activation, k_activation, *args, **kwargs):
        q_pooled = self.self_attention_pool(q_activation)
        k_pooled = self.self_attention_pool(k_activation)
        q_n = F.normalize(q_pooled, dim=1, p=2)
        k_n = F.normalize(k_pooled, dim=1, p=2)
        loss = torch.mean(torch.frobenius_norm(q_n - k_n), dim=-1)
        return self.scale * loss
