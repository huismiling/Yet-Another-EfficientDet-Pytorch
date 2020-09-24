# Author: Zylo117

import math

from torch import nn
import torch.nn.functional as F


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        self.padd_inited = False

    def forward(self, x):
        if not self.padd_inited:
            # h, w = x.shape[-2:]
            h, w = x.cpu().detach().numpy().shape[-2:]
            h_step = math.ceil(w / self.stride[1])
            v_step = math.ceil(h / self.stride[0])
            h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
            v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

            extra_h = h_cover_len - w
            extra_v = v_cover_len - h

            self.left = extra_h // 2
            self.right = extra_h - self.left
            self.top = extra_v // 2
            self.bottom = extra_v - self.top

        x = F.pad(x, [self.left, self.right, self.top, self.bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2
        
        self.padd_inited = False

    def forward(self, x):
        if not self.padd_inited:
            # h, w = x.shape[-2:]
            h, w = x.cpu().detach().numpy().shape[-2:]

            h_step = math.ceil(w / self.stride[1])
            v_step = math.ceil(h / self.stride[0])
            h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
            v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

            extra_h = h_cover_len - w
            extra_v = v_cover_len - h

            self.left = extra_h // 2
            self.right = extra_h - self.left
            self.top = extra_v // 2
            self.bottom = extra_v - self.top

        x = F.pad(x, [self.left, self.right, self.top, self.bottom])

        x = self.pool(x)
        return x
