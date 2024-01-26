import torch
import torch.nn as nn
from torch import Tensor
from .initializers import he_normalizer
from models.cotnet import CotLayer
from einops import rearrange


class ConvBlock(nn.Module):
  def __init__(self, in_channels, cotlayer_channels, cotlayer_kernel_size):
    super(ConvBlock, self).__init__()

    self.net = nn.Sequential(
            nn.Conv2d(in_channels, cotlayer_channels, 1, bias=False),
            CotLayer(cotlayer_channels, cotlayer_kernel_size),
            nn.BatchNorm2d(cotlayer_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
    )

    # self.net[0].weight = self.net[0].apply(he_normalizer)


  def forward(self, x):
    """
    accepts a 2D-Tensor and returns two 2D-Tensors
    """
    return self.net(x)


class Conv2dWithHeNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same", padding_mode="zeros", bias=False, use_bn=False, use_activation=True):
        super(Conv2dWithHeNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode)
        self.conv.apply(he_normalizer)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99, affine=False) if use_bn else None
        self.act_func = nn.ReLU(inplace=True) if use_activation else None

    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        
        if self.act_func:
            output = self.act_func(output)

        return output


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, maxpool_kernel_size, maxpool_stride, padding="same", padding_mode="zeros", bias: bool = False, dropout: float = 0.0):
        super(CustomConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99),
            nn.ReLU(inplace=True),
            # kernel equal with stride in size is the padding 'same'
            nn.MaxPool2d(maxpool_kernel_size, maxpool_stride),
        )
        self.conv[0] = self.conv[0].apply(he_normalizer)

        if dropout != 0.0:
            self.conv.add_module("dropout", nn.Dropout(0.5))

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class BackBone(nn.Module):
    def __init__(self, in_channels, dropout_val = 0.5):
        super(BackBone, self).__init__()

        cotlayer_kernel_size = 5

        self.nets = nn.ModuleDict({
            "net64": ConvBlock(in_channels, 64, cotlayer_kernel_size),
            "net128": ConvBlock(64, 128, cotlayer_kernel_size),
            "net256": ConvBlock(128, 256, cotlayer_kernel_size),
            "net512": ConvBlock(256, 512, cotlayer_kernel_size),
        })

        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        out64 = self.nets["net64"](x)
        out128 = self.nets["net128"](out64)
        out256 = self.nets["net256"](out128)
        out512 = self.nets["net512"](out256)
        out512 = self.dropout(out512)
        return [out64, out128, out256, out512]


class UpsamplingBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        translation_in_out_channels_1: tuple, 
        translation_in_out_channels_2: tuple, 
        merge_upsample=True
    ):
        super(UpsamplingBlock, self).__init__()

        self.merge_upsample = merge_upsample

        self.conv_bn2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if merge_upsample:
            self.conv_bn3 = nn.Sequential(
                nn.Conv2d(translation_in_out_channels_1[0]*2, translation_in_out_channels_1[1], 3, padding=1),
                nn.BatchNorm2d(translation_in_out_channels_1[1]),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_bn3 = nn.Sequential(
                nn.Conv2d(translation_in_out_channels_1[0], translation_in_out_channels_1[1], 3, padding=1),
                nn.BatchNorm2d(translation_in_out_channels_1[1]),
                nn.ReLU(inplace=True)
            )

        self.conv_bn4 = nn.Sequential(
            nn.Conv2d(translation_in_out_channels_2[0], translation_in_out_channels_2[1], 3, padding=1),
            nn.BatchNorm2d(translation_in_out_channels_2[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, left_cot_x = None, right_cot_x = None):
        """
        receives a merged or not 4D-Tensor
        """
        out = self.conv_bn2(x)
        # print("--------\nhow ever the entry shape is --> ", x.shape)
        # print("out shape from upsampling block --> ", out.shape)
        
        if self.merge_upsample:
            merge = torch.cat([left_cot_x, right_cot_x], axis=1)
            # print("concatenated out shape from left and right --> ", merge.shape)
            out = torch.cat([out, merge], axis=1)
        
        return self.conv_bn4(self.conv_bn3(out))


class ChunkGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1*x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class EncoderSpaChAttBlock(nn.Module):
    def __init__(self, in_channels, num_heads=3, chn_expanded=2, dropout_rate=0.0):
        super(EncoderSpaChAttBlock, self).__init__()

        self.num_heads = num_heads

        self.qkv = nn.Conv2d(in_channels, in_channels*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(in_channels*3, in_channels*3, kernel_size=3, padding=1, stride=1, groups=in_channels*3)

        self.prj_outc = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.tempc = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.prj_outs = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.temps = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.gate = ChunkGate()

        ffn_channel = chn_expanded * in_channels
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()

        self.betac = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.betas = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        
        qs = q.clone().permute(0, 1, 3, 2)
        ks = k.clone().permute(0, 1, 3, 2)
        vs = v.clone().permute(0, 1, 3, 2)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # # Channel attention
        attn = (q @ k.transpose(-2, -1)) * self.tempc
        attn = self.relu(attn)
        attn = self.softmax(attn)

        outc = (attn @ v)

        # Spatial Attention
        qs = torch.nn.functional.normalize(qs, dim=-1)
        ks = torch.nn.functional.normalize(ks, dim=-1)

        attns = (qs @ ks.transpose(-2, -1)) * self.temps
        attns = self.relu(attns)
        attns = self.softmax(attns)
        outs = (attns @ vs)
        outs = outs.permute(0, 1, 3, 2)

        outc = rearrange(outc, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outs = rearrange(outs, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        xc = self.prj_outc(outc)
        xc = self.dropout1(xc)
        xs = self.prj_outs(outs)
        xs = self.dropout1(xs)

        y = inp + xc * self.betac + xs * self.betas

        x = self.conv4(self.norm2(y))
        x = self.gate(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

