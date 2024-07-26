# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import timm.models.vision_transformer
from Morgnet import gnconv


def custom_build_activation_layer(cfg):
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    if cfg is None:
        return nn.Identity()
    if cfg['type'] == 'SiLU':
        return nn.SiLU()
    else:
        return build_activation_layer(cfg)


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_activation_layer(act_cfg)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.
    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, ],
                 channel_split=[1, 3, 4, ],
                 init_cfg=None):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        # assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=3,
            padding=1,
            groups=self.embed_dims,
            stride=1,
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=3,
            padding=1,
            groups=self.embed_dims_1,
            stride=1,
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=3,
            padding=1 ,
            groups=self.embed_dims_2,
            stride=1,
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.
    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=dict(type="SiLU"),
                 attn_force_fp32=False,
                 init_cfg=None):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = custom_build_activation_layer(attn_act_cfg)
        self.act_gate = custom_build_activation_layer(attn_act_cfg)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x


    def forward_gating(self, g, v):
        g = g.to(torch.float32)
        v = v.to(torch.float32)
        return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class Res_1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=None, pool_size=None):
        super(Res_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel,kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm1d(out_channel)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.down_pool = nn.Conv1d(in_channel,out_channel,kernel_size=3,stride=2,padding=1)
        self.BN2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.BN1(self.conv1(x)))  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.pool(x1))  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.BN2(self.down_pool(x))  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x2 + x3)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
            # x = self.spec_features(x)
            # x = self.res_net1D(x)
        return x.numel()

    def __init__(self, input_channels, n_classes,patch_size, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels
        self.patch_size = patch_size
        #
        # cfg = {
        #     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        #     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        #     'D': [32, 'M', 64,'M'],
        #     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
        #           'M'],
        # }
        # layers = []
        #
        # in_channels = 1
        # for v in cfg['D']:
        #
        #     if v == 'M':
        #         layers += [nn.MaxPool1d(kernel_size=3)]
        #     else:
        #         conv1d = nn.Conv1d(in_channels, v, kernel_size=5)
        #         layers += [conv1d, nn.BatchNorm1d(v), nn.LeakyReLU(inplace=True)]
        #         in_channels = v
        #
        # self.spec_features = nn.Sequential(*layers)
        # self.block1 = Res_1D(1,20)
        # self.block2 = Res_1D(20,40)
        # self.block3 = Res_1D(40,60)
        # self.res_net1D = nn.Sequential(self.block1,self.block2,self.block3)

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.bn_conv = nn.BatchNorm1d(20)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # self.norm_conv = nn.BatchNorm1d(self.features_size)
        # [n4 is set to be 100]
        hidden = 192
        self.fc1 = nn.Linear(self.features_size, hidden)
        self.norm_conv = nn.LayerNorm(self.features_size)
        self.bn_fc = nn.BatchNorm1d(hidden)
        self.norm_fc = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, n_classes)


    def forward(self, x1):
        center_pos = (self.patch_size) // 2
        center_X = x1[:, :, center_pos, center_pos]
        x = torch.flatten(center_X, 1)
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        # x = self.spec_features(x)
        # x = self.res_net1D(x)
        x = self.conv(x)
        x = self.bn_conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        # x = self.norm_conv(x)
        x = torch.tanh(self.fc1(x))
        x = self.norm_fc(x)
        # x = self.bn_fc(x)
        # x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_norm = nn.LayerNorm(dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x ,spe_agg):
        B, N, C = x.shape
        scale = (torch.ones((B,N,C))).cuda()
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = (scale*(spe_agg.unsqueeze(1))).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_norm = nn.LayerNorm(dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x ,x2):
        B, N, C = x.shape
        _, N2, _ = x2.shape

        kv = self.kv(x).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        q = self.q(x2).reshape(B, N, 1, C).permute(2, 0, 1, 3)

        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class attn_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x ,spe_agg):
        x = x + self.drop_path(self.attn(self.norm1(x),spe_agg))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,alpha=0.9, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.img_size = kwargs['img_size']
        embed_dim = kwargs['embed_dim']
        norm_layer = kwargs['norm_layer']
        self.spectral_net = HuEtAl(kwargs['in_chans'],kwargs['num_classes'],kwargs['img_size'])
        self.BN_fc = nn.BatchNorm1d(kwargs['num_classes'])
        self.multiorder_agg = gnconv(embed_dim)  #MultiOrderGatedAggregation(embed_dim ,attn_dw_dilation=[0,0,0])

        self.spec_agg = cross_Attention(dim=embed_dim,num_heads=kwargs['num_heads'])
        self.head = nn.Linear(self.embed_dim, kwargs['num_classes']) if kwargs['num_classes'] > 0 else nn.Identity()
        self.agg_norm = norm_layer(embed_dim)
        self.lambda1 = alpha


        if self.global_pool:


            self.fc_norm = norm_layer(embed_dim)
            self.spa_norm = norm_layer(embed_dim)
            self.fus_norm = nn.BatchNorm1d(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        f_spec = self.spectral_net(x)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)


        x_spa = self.spec_agg(x, f_spec.unsqueeze(1).repeat(1, x.shape[1], 1))

        x_agg = x_spa[:, 1:, :].transpose(1, 2)
        _, C, N = x_agg.shape
        x_shotcut = x_agg.view(B,C,self.img_size//2,self.img_size//2)

        x_agg = x_shotcut+self.multiorder_agg(x_shotcut)
        x_agg = x_agg.flatten(2).transpose(1,2)

        if self.global_pool:
            x_agg = x_agg.mean(dim=1)
            # x_agg = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x_agg)

        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # f_spa = self.spa_norm(outcome)
        # weight_ss =torch.sigmoid(self.lambda1)
        fus_out = self.lambda1 * f_spec + (1-self.lambda1) * outcome

        # fus_out = torch.cat((f_spec,outcome),1)
        # fus_out = self.fus_norm(fus_out)

        return fus_out

    def forward_head(self, x ,pre_logits: bool = False):
        x = self.head(x)

        return  x

    def forward(self,x, training=False):
        f = self.forward_features(x)
        out = self.forward_head(f)

        return f,out


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=192, depth=4, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model