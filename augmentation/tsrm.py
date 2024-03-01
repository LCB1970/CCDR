from augmentation.rand_conv import RandConv2d
from augmentation.randin import RandIN
import torch.nn as nn
import torch, random


class text_style_random_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 gamma_std, beta_std,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=False, range_up=None, range_low=None, **kwargs
                 ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(text_style_random_layer, self).__init__()

        self.rand_texture = RandConv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2,
                                       rand_bias=rand_bias, distribution=distribution,
                                       clamp_output=clamp_output,
                                       range_low=range_low, range_up=range_up,
                                       **kwargs)
        self.rand_style = RandIN(out_channels, gamma_std, beta_std)
        self.tahn = nn.Tanh()
        self.randomize()

    def randomize(self):
        self.rand_texture.randomize()
        self.rand_style.random()

    def forward(self, input):
        x = self.rand_texture(input)
        x = self.rand_style(x)
        output = self.tahn(x)
        return output


class text_random_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 gamma_std, beta_std,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=False, range_up=None, range_low=None, **kwargs
                 ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(text_random_layer, self).__init__()

        self.rand_texture = RandConv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2,
                                       rand_bias=rand_bias, distribution=distribution,
                                       clamp_output=clamp_output,
                                       range_low=range_low, range_up=range_up,
                                       **kwargs)
        # self.rand_style = RandIN(out_channels, gamma_std, beta_std)
        # self.tahn = nn.Tanh()
        self.randomize()

    def randomize(self):
        self.rand_texture.randomize()
        # self.rand_style.random()

    def forward(self, input):
        x = self.rand_texture(input)
        # x = self.rand_style(x)
        # output = self.tahn(x)
        return x


class style_random_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 gamma_std, beta_std,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=False, range_up=None, range_low=None, **kwargs
                 ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(style_random_layer, self).__init__()

        # self.rand_texture = RandConv2d(in_channels, out_channels, kernel_size, padding = kernel_size // 2,
        #                                   rand_bias=rand_bias, distribution=distribution,
        #                                   clamp_output=clamp_output,
        #                                   range_low=range_low, range_up=range_up,
        #                                   **kwargs)
        self.rand_style = RandIN(in_channels, gamma_std, beta_std)
        self.tahn = nn.Tanh()
        self.randomize()

    def randomize(self):
        # self.rand_texture.randomize()
        self.rand_style.random()

    def forward(self, input):
        # x = self.rand_texture(input)
        x = self.rand_style(input)
        output = self.tahn(x)
        return output


class TSRM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_list,
                 layer_max, gamma_std, beta_std,
                 mixing=False,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=True, data_mean=(0.5, 0.5, 0.5), data_std=(0.5, 0.5, 0.5), **kwargs
                 ):
        super(TSRM, self).__init__()
        self.layer_max = layer_max
        self.kernel_size_list = kernel_size_list

        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (
                    self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(3).reshape(3, 1,
                                                                                                   1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(3).reshape(3, 1,
                                                                                                     1) - self.data_mean) / self.data_std)

        self.layer = nn.ModuleDict(
            {str(kernel_size): text_style_random_layer(
                in_channels, out_channels, kernel_size,
                gamma_std, beta_std, rand_bias, distribution,
                self.clamp_output, self.range_up, self.range_low,
                **kwargs) for kernel_size in kernel_size_list}
        )
        self.randomize()
        self.mixing = mixing

    def randomize(self):
        self.layer_num = random.randint(1, self.layer_max)
        self.kernel_size = self.kernel_size_list[random.randint(0, len(self.kernel_size_list) - 1)]
        self.alpha = random.random()

    def forward(self, x):
        # print(self.layer_num, self.kernel_size, self.alpha)
        input = x
        device = x.device
        for _ in range(self.layer_num):
            self.layer[str(self.kernel_size)].randomize()
            self.layer[str(self.kernel_size)].to(device)
            x = self.layer[str(self.kernel_size)](x)
        output = x
        if self.mixing:
            output = (self.alpha * output + (1 - self.alpha) * input)
        return output


class TRM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_list,
                 layer_max, gamma_std, beta_std,
                 mixing=False,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=True, data_mean=(0.5, 0.5, 0.5), data_std=(0.5, 0.5, 0.5), **kwargs
                 ):
        super(TRM, self).__init__()
        self.layer_max = layer_max
        self.kernel_size_list = kernel_size_list

        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (
                    self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(3).reshape(3, 1,
                                                                                                   1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(3).reshape(3, 1,
                                                                                                     1) - self.data_mean) / self.data_std)

        self.layer = nn.ModuleDict(
            {str(kernel_size): text_random_layer(
                in_channels, out_channels, kernel_size,
                gamma_std, beta_std, rand_bias, distribution,
                self.clamp_output, self.range_up, self.range_low,
                **kwargs) for kernel_size in kernel_size_list}
        )
        self.randomize()
        self.mixing = mixing

    def randomize(self):
        self.layer_num = random.randint(1, self.layer_max)
        self.kernel_size = self.kernel_size_list[random.randint(0, len(self.kernel_size_list) - 1)]
        self.alpha = random.random()

    def forward(self, x):
        # print(self.layer_num, self.kernel_size, self.alpha)
        input = x
        device = x.device
        for _ in range(self.layer_num):
            self.layer[str(self.kernel_size)].randomize()
            self.layer[str(self.kernel_size)].to(device)
            x = self.layer[str(self.kernel_size)](x)
        output = x
        if self.mixing:
            output = (self.alpha * output + (1 - self.alpha) * input)
        return output


class SRM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_list,
                 layer_max, gamma_std, beta_std,
                 mixing=False,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=True, data_mean=(0.5, 0.5, 0.5), data_std=(0.5, 0.5, 0.5), **kwargs
                 ):
        super(SRM, self).__init__()
        self.layer_max = layer_max
        self.kernel_size_list = kernel_size_list

        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (
                    self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(3).reshape(3, 1,
                                                                                                   1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(3).reshape(3, 1,
                                                                                                     1) - self.data_mean) / self.data_std)

        self.layer = nn.ModuleDict(
            {str(kernel_size): text_random_layer(
                in_channels, out_channels, kernel_size,
                gamma_std, beta_std, rand_bias, distribution,
                self.clamp_output, self.range_up, self.range_low,
                **kwargs) for kernel_size in kernel_size_list}
        )
        self.randomize()
        self.mixing = mixing

    def randomize(self):
        self.layer_num = random.randint(1, self.layer_max)
        self.kernel_size = self.kernel_size_list[random.randint(0, len(self.kernel_size_list) - 1)]
        self.alpha = random.random()

    def forward(self, x):
        # print(self.layer_num, self.kernel_size, self.alpha)
        input = x
        device = x.device
        for _ in range(self.layer_num):
            self.layer[str(self.kernel_size)].randomize()
            self.layer[str(self.kernel_size)].to(device)
            x = self.layer[str(self.kernel_size)](x)
        output = x
        if self.mixing:
            output = (self.alpha * output + (1 - self.alpha) * input)
        return output
