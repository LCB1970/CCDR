import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def random_instance_normalization(content_feat, gamma_std=0.5, beta_std=0.5):
    device = content_feat.device
    size = content_feat.size()
    beta = torch.normal(mean=0., std=beta_std, size=(size[0], size[1], 1, 1)).to(device)
    gamma = torch.normal(mean=0., std=gamma_std, size=(size[0], size[1], 1, 1)).to(device)
    print(beta.shape, beta.shape)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * gamma.expand(size) + beta.expand(size)


class RandIN(nn.Module):
    def __init__(self, channel, gamma_std=0.5, beta_std=0.5):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channel, affine=False)
        self.gamma_std = gamma_std
        self.beta_std = beta_std
        self.channel = channel
        self.random()

    def random(self):
        self.register_buffer('gamma', torch.normal(mean=0., std=self.gamma_std, size=(self.channel, 1, 1)))
        self.register_buffer('beta', torch.normal(mean=0., std=self.beta_std, size=(self.channel, 1, 1)))

    def forward(self, input):
        x = self.norm(input)
        return x * self.gamma + self.beta
