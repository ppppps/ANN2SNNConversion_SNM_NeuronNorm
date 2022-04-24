import torch
import torch.nn as nn
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SPIKE_PosNeg_layer(nn.Module):
    def __init__(self, PosThresh, NegThresh, Conv2d):
        super().__init__()
        self.posThresh = PosThresh
        self.negThresh = NegThresh
        self.ops = Conv2d
        self.mem = 0
        self.transmitted = 0

    def init_mem(self):
        self.mem = 0

    def forward(self, input, t):
        x = self.ops(input)
        if t == 0:
            self.transmitted = torch.zeros_like(x)

        self.mem += x
        posSpike = self.mem.ge(self.posThresh).float() * self.posThresh
        negSpike = self.mem.le(self.negThresh).float() * self.negThresh
        compare = torch.where(self.transmitted > 0, torch.ones_like(self.transmitted),
                              torch.zeros_like(self.transmitted))
        negSpike = negSpike * compare
        spike = posSpike + negSpike

        self.mem -= spike
        self.transmitted += spike

        return spike, self.mem


class SPIKE_PosNeg_layer_BN(nn.Module):
    def __init__(self, PosThresh, NegThresh, Conv2d, BatchNorm2d):
        super().__init__()
        self.posThresh = PosThresh
        self.negThresh = NegThresh
        self.ops = Conv2d
        self.bn = BatchNorm2d
        self.mem = 0
        self.transmitted = 0

    def compute_Conv_weight(self):
        bn_weight = copy.deepcopy(self.bn.weight.reshape(self.ops.out_channels, 1, 1, 1))
        bn_weight = bn_weight.expand_as(self.ops.weight)
        running_var = copy.deepcopy(self.bn.running_var)

        safe_std = torch.sqrt(running_var + self.bn.eps)
        std = safe_std.reshape(self.ops.out_channels, 1, 1, 1)
        std = std.expand_as(self.ops.weight)
        self.ops.weight = nn.Parameter(self.ops.weight * bn_weight / std).to('cuda')
        self.ops.bias = nn.Parameter(
            self.bn.weight / safe_std * (self.ops.bias - self.bn.running_mean) + self.bn.bias).to('cuda')

    def init_mem(self):
        self.mem = 0

    def forward(self, input, t):
        x = self.ops(input)
        if t == 0:
            self.transmitted = torch.zeros_like(x)

        self.mem += x
        posSpike = self.mem.ge(self.posThresh).float() * self.posThresh
        negSpike = self.mem.le(self.negThresh).float() * self.negThresh
        compare = torch.where(self.transmitted > 0, torch.ones_like(self.transmitted),
                              torch.zeros_like(self.transmitted))
        negSpike = negSpike * compare
        spike = posSpike + negSpike

        self.mem -= spike
        self.transmitted += spike

        return spike, self.mem
