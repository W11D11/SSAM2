

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# 实现梯度检查点（用于：训练时内存优化以时间为代价，减少训练占的内存）
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_




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


def window_partition(x, window_size):

    """
    将输入张量分割成多个窗口。
    参数：
    x: 输入张量，形状为（b,h,w,c),其中b是批次大小，h是高度，w是宽度，c是通道数
    window_size：窗口的大小
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns（返回）:
        windows: 分割后的窗口张量，形状为(num_windows*B, window_size, window_size, C)
    """
    # 获取输入张量的形状
    B, H, W, C = x.shape
    # 重新调整输入张量的形状以便分割成多个窗口
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 调整张量的维度顺序，并合并窗口维度
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将分割的窗口恢复成原始图像。
    参数:
        windows (Tensor): 分割的窗口张量，形状为 (num_windows*b, window_size, window_size, c)。
        window_size (int): 窗口大小。
        h (int): 原始图像的高度。
        w (int): 原始图像的宽度。
    返回:
        Tensor: 恢复后的图像张量，形状为 (b, h, w, c)。

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # 计算批次大小
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 重新调整窗口张量的形状
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 调整张量的维度顺序，并合并窗口维度，恢复成原始图像的形状
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# nn.Module 是所有自定义神经网络模型的基类
class WindowAttention(nn.Module):
    r"""
      基于窗口的多头自注意力（W-MSA）模块，具有相对位置偏置。
    支持移位和非移位窗口。
    参数:
        dim (int): 输入通道的数量。
        window_size (tuple[int]): 窗口的高度和宽度。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional): 如果为 True，则为查询、键、值添加可学习的偏置。默认值为 True。
        qk_scale (float, optional): 如果设置，则覆盖默认的 qk 缩放值 head_dim ** -0.5。
        attn_drop (float, optional): 注意力权重的 dropout 率。默认值为 0.0。
        proj_drop (float, optional): 输出的 dropout 率。默认值为 0.0。
    输入:
        x: 输入特征，形状为 (num_windows*b, n, c)，其中 num_windows 是窗口的数量，b 是批次大小，n 是窗口内的令牌数量，c 是通道数。
        mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None。
    输出:
        Tensor: 输出特征，形状为 (num_windows*b, n, c)。

     Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww  窗口的高度和宽度
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # define a parameter table of relative position bias 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window 获取窗口内每个令牌的相对位置索引3
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        #  定义查询、键和值的线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)   # 注意力 dropout
        self.proj = nn.Linear(dim, dim) # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)   # 输出投影 dropout

        # 初始化相对位置偏置参数表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)    # Softmax 层，用于注意力计算

    def forward(self, x, mask=None):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入特征，形状为 (num_windows*b, n, c)。
            mask (Tensor, optional): 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None。
        返回:
            Tensor: 输出特征，形状为 (num_windows*b, n, c)
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape          # 获取输入的形状
        # 将输入特征转换为查询、键和值
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)# 分别获取查询、键和值

        q = q * self.scale       # 缩放查询向量
        attn = (q @ k.transpose(-2, -1))          # 计算注意力分数


        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果提供了掩码，则应用掩码
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)        # 应用注意力 dropout
        # 将注意力应用到值向量上
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)     # 输出投影
        x = self.proj_drop(x)    # 输出投影 dropout
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """
        计算 FLOPs（浮点运算次数）。

        参数:
           N(int): 窗口内的令牌数量。
        返回:
            int: FLOPs 数量。
        """
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)  查询、键和值的线性变换
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))  注意力计算 (q @ k.transpose)
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)  应用注意力到值向量 (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)  输出投影
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
     这个类定义了一个Swin Transformer块，包含窗口多头自注意力和MLP块。
     参数:
        dim (int): 输入通道的数量。
        input_resolution (tuple[int]): 输入的分辨率 (高度, 宽度)。
        num_heads (int): 注意力头的数量。
        window_size (int): 窗口大小。
        shift_size (int): SW-MSA 的移位大小。
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。
        qkv_bias (bool, optional): 如果为True，则为查询、键和值添加可学习的偏置。默认值为True。
        qk_scale (float, optional): 如果设置，则覆盖默认的 qk 缩放值 head_dim ** -0.5。
        drop (float, optional): Dropout率。默认值为0.0。
        attn_drop (float, optional): 注意力权重的Dropout率。默认值为0.0。
        drop_path (float, optional): 随机深度的比率。默认值为0.0。
        act_layer (nn.Module, optional): 激活层。默认值为 nn.GELU。
        norm_layer (nn.Module, optional): 归一化层。默认值为 nn.LayerNorm。
    输入:
        x: 输入特征，形状为 (batch_size, seq_len, dim)。
        x_size: 输入的空间维度 (高度, 宽度)。
    输出:
        Tensor: 输出特征，形状为 (batch_size, seq_len, dim)
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim        # 输入通道数
        self.input_resolution = input_resolution   # 输入分辨率
        self.num_heads = num_heads    # 注意力头数量
        self.window_size = window_size    # 窗口大小
        self.shift_size = shift_size      # 移位大小
        self.mlp_ratio = mlp_ratio        # MLP比率

        # 如果窗口大小大于输入分辨率，则不进行窗口划分
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)      # 第一层归一化
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)        # 窗口注意力层

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()       # 随机深度
        self.norm2 = norm_layer(dim)              # 第二层归一化
        mlp_hidden_dim = int(dim * mlp_ratio)     # MLP隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     # MLP层

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)    # 计算注意力掩码
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)           # 注册注意力掩码

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        """
                计算 SW-MSA 的注意力掩码。
                参数:
                    x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
                返回:
                    Tensor: 注意力掩码，形状为 (num_windows, window_size*window_size, window_size*window_size)。
                """
        h,w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 H W 1 创建掩码张量，形状为 (1, 高度, 宽度, 1)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  #  设置掩码值
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1 划分窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        """
       前向传播函数。
        参数:
            x (Tensor): 输入特征，形状为 (batch_size, seq_len, dim)。
            x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
        返回:
            Tensor: 输出特征，形状为 (batch_size, seq_len, dim)

        """
        H, W = x_size       # 获取输入的高度和宽度
        B, L, C = x.shape   # 获取输入的批次大小和通道数
        # assert L == H * W, "input feature has wrong size"

        shortcut = x            # 残差连接
        x = self.norm1(x)       # 第一层归一化
        x = x.view(B, H, W, C)  # 重塑张量为图像形状

        # cyclic shift  循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows  划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (兼容测试图像的形状是窗口大小的倍数）
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows  合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift  逆循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)         # 恢复为 (batch_size, seq_len, dim) 形状

        # FFN
        x = shortcut + self.drop_path(x)   # 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # MLP 和 归一化

        return x

    def extra_repr(self) -> str:  # 返回类的额外表示信息
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        """
            计算 FLOPs（浮点运算次数）。

            返回:
                   int: FLOPs 数量。
         """
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
     这个类定义了一个Patch合并层，用于将输入的Patch合并，减少空间维度并增加通道维度。

    参数:
        input_resolution (tuple[int]): 输入特征的分辨率 (高度, 宽度)。
        dim (int): 输入通道的数量。
        norm_layer (nn.Module, optional): 归一化层。默认是 nn.LayerNorm。
    输入:
        x: 输入特征，形状为 (batch_size, h*w, dim)。
    输出:
        Tensor: 合并后的特征，形状为 (batch_size, h/2*w/2, 2*dim)
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution      # 输入分辨率
        self.dim = dim       # 输入通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)     # 线性层，用于减少空间维度并增加通道维度
        self.norm = norm_layer(4 * dim)    # 归一化层

    def forward(self, x):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入特征，形状为 (batch_size, h*w, dim)。
        返回:
            Tensor: 合并后的特征，形状为 (batch_size, h/2*w/2, 2*dim)。
        x: B, H*W, C
        """
        H, W = self.input_resolution     # 获取输入分辨率
        B, L, C = x.shape                # 获取输入形状
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)             # 重新调整输入张量的形状
        # 按照窗口大小进行划分，并在通道维度上拼接
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)         # 归一化
        x = self.reduction(x)    # 线性层降维

        return x

    def extra_repr(self) -> str:       # 返回类的额外表示信息。
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """
        计算 FLOPs（浮点运算次数）。
        返回:
            int: FLOPs 数量。
        """
        H, W = self.input_resolution
        flops = H * W * self.dim           # 归一化 FLOPs
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim    # 线性层 FLOPs
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    这个类定义了一个基本的Swin Transformer层，包括多个Swin Transformer块和一个可选的下采样层。
    参数:
        dim (int): 输入通道的数量。
        input_resolution (tuple[int]): 输入的分辨率 (高度, 宽度)。
        depth (int): 块的数量。
        num_heads (int): 注意力头的数量。
        window_size (int): 本地窗口大小。
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。
        qkv_bias (bool, optional): 如果为True，则为查询、键和值添加可学习的偏置。默认值为True。
        qk_scale (float, optional): 如果设置，则覆盖默认的 qk 缩放值 head_dim ** -0.5。
        drop (float, optional): Dropout率。默认值为0.0。
        attn_drop (float, optional): 注意力权重的Dropout率。默认值为0.0。
        drop_path (float, optional): 随机深度的比率。默认值为0.0。
        norm_layer (nn.Module, optional): 归一化层。默认值为 nn.LayerNorm。
        downsample (nn.Module | None, optional): 层结束时的下采样层。默认值为None。
        use_checkpoint (bool): 是否使用检查点以节省内存。默认值为False。
    输入:
        x: 输入特征，形状为 (batch_size, seq_len, dim)。
        x_size: 输入的空间维度 (高度, 宽度)。
    输出:
    Tensor: 输出特征，形状为 (batch_size, seq_len, dim).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim      # 输入通道数
        self.input_resolution = input_resolution     # 输入分辨率
        self.depth = depth    # 块的数量
        self.use_checkpoint = use_checkpoint            # 是否使用检查点

        # build blocks 构建Swin Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer   下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入特征，形状为 (batch_size, seq_len, dim)。
            x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
        返回:
            Tensor: 输出特征，形状为 (batch_size, seq_len, dim)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)    # 使用检查点
            else:
                x = blk(x, x_size)                          # 不使用检查点
        if self.downsample is not None:               # 下采样
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:     # 返回类的额外表示信息
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """
        计算 FLOPs（浮点运算次数）。

        返回:
            int: FLOPs 数量。
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
这个类定义了一个残差Swin Transformer块，包含一个基础层和一个卷积层，用于特征提取和残差连接。
    参数:
        dim (int): 输入通道的数量。
        input_resolution (tuple[int]): 输入分辨率 (高度, 宽度)。
        depth (int): 块的数量。
        num_heads (int): 注意力头的数量。
        window_size (int): 本地窗口大小。
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。
        qkv_bias (bool, optional): 如果为True，则为查询、键和值添加可学习的偏置。默认值为True。
        qk_scale (float, optional): 如果设置，则覆盖默认的 qk 缩放值 head_dim ** -0.5。
        drop (float, optional): Dropout率。默认值为0.0。
        attn_drop (float, optional): 注意力权重的Dropout率。默认值为0.0。
        drop_path (float, optional): 随机深度的比率。默认值为0.0。
        norm_layer (nn.Module, optional): 归一化层。默认值为 nn.LayerNorm。
        downsample (nn.Module | None, optional): 层结束时的下采样层。默认值为None。
        use_checkpoint (bool): 是否使用检查点以节省内存。默认值为False。
        img_size (int): 输入图像的大小。默认值为224。
        patch_size (int): Patch大小。默认值为4。
        resi_connection (str): 残差连接的卷积块类型。默认值为'1conv'。
    输入:
        x (Tensor): 输入特征，形状为 (batch_size, seq_len, dim)。
        x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
    输出:
        Tensor: 输出特征，形状为 (batch_size, seq_len, dim)。

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim     # 输入通道数
        self.input_resolution = input_resolution         # 输入分辨率
        # 定义基础层
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)
        # 定义卷积层，根据 resi_connection 参数选择不同的卷积块
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))
        # 定义Patch嵌入层和反嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        """
            前向传播函数。
            参数:
                    x (Tensor): 输入特征，形状为 (batch_size, seq_len, dim)。
                    x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
            返回:
                    Tensor: 输出特征，形状为 (batch_size, seq_len, dim)。
        """
        # 通过基础层，反嵌入，卷积和嵌入过程，最后加上输入实现残差连接
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):     # 计算浮点运算次数
        flops = 0
        flops += self.residual_group.flops()   # 计算基础层的FLOPs
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9    # 卷积层的FLOPs
        flops += self.patch_embed.flops()           # Patch嵌入层的FLOPs
        flops += self.patch_unembed.flops()         # Patch反嵌入层的FLOPs

        return flops

# PatchEmbed 类的详细注释
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    这个类定义了一个图像到Patch嵌入的层，将输入图像分割成多个Patch，并将其展平和嵌入到更高维度的空间中。
    参数:
        img_size (int): 图像大小。默认值为224。
        patch_size (int): Patch大小。默认值为4。
        in_chans (int): 输入图像的通道数。默认值为3。
        embed_dim (int): 线性投影的输出通道数。默认值为96。
        norm_layer (nn.Module, optional): 归一化层。默认值为None。
    输入:
        x (Tensor): 输入图像，形状为 (batch_size, in_chans, img_size, img_size)。
    输出:
        Tensor: 嵌入后的特征，形状为 (batch_size, num_patches, embed_dim)。
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)       # 将图像大小转换为元组
        patch_size = to_2tuple(patch_size)   # 将Patch大小转换为元组
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution             # 计算Patch的分辨率
        self.num_patches = patches_resolution[0] * patches_resolution[1]    # 计算Patch的数量

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)    # 定义归一化层
        else:
            self.norm = None

    def forward(self, x):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入图像，形状为 (batch_size, in_chans, img_size, img_size)。
        返回:
            Tensor: 嵌入后的特征，形状为 (batch_size, num_patches, embed_dim)。
        """
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C  展平并转换维度
        if self.norm is not None:
            x = self.norm(x)          # 应用归一化
        return x

    def flops(self):    # 计算浮点运算次数
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim   # 归一化层的FLOPs
        return flops

# PatchUnEmbed 类的详细注释
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    这个类定义了一个Patch反嵌入的层，将嵌入的特征恢复为图像。
    参数:
        img_size (int): 图像大小。默认值为224。
        patch_size (int): Patch大小。默认值为4。
        in_chans (int): 输入图像的通道数。默认值为3。
        embed_dim (int): 线性投影的输出通道数。默认值为96。
        norm_layer (nn.Module, optional): 归一化层。默认值为None。
    输入:
        x (Tensor): 嵌入的特征，形状为 (batch_size, num_patches, embed_dim)。
        x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
    输出:
        Tensor: 恢复后的图像，形状为 (batch_size, in_chans, img_size, img_size)。

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)          # 将图像大小转换为元组
        patch_size = to_2tuple(patch_size)      # 将Patch大小转换为元组
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """
                前向传播函数。
                参数:
                    x (Tensor): 嵌入的特征，形状为 (batch_size, num_patches, embed_dim)。
                    x_size (tuple[int]): 输入的空间维度 (高度, 宽度)。
                返回:
                    Tensor: 恢复后的图像，形状为 (batch_size, in_chans, img_size, img_size)。
                """
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):        # 计算 FLOPs（浮点运算次数）
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
    SwinIR: 使用Swin Transformer进行图像恢复的PyTorch实现。
    参数:
        img_size (int | tuple(int)): 输入图像大小。默认值为 64。
        patch_size (int | tuple(int)): Patch 大小。默认值为 1。
        in_chans (int): 输入图像通道数。默认值为 3。
        embed_dim (int): Patch 嵌入维度。默认值为 96。
        depths (tuple(int)): 每个Swin Transformer层的深度。
        num_heads (tuple(int)): 不同层中的注意力头数量。
        window_size (int): 窗口大小。默认值为 7。
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。默认值为 4。
        qkv_bias (bool): 如果为True，则为查询、键和值添加可学习的偏置。默认值为 True。
        qk_scale (float): 如果设置，则覆盖默认的 qk 缩放值 head_dim ** -0.5。默认值为 None。
        drop_rate (float): Dropout率。默认值为 0。
        attn_drop_rate (float): 注意力权重的Dropout率。默认值为 0。
        drop_path_rate (float): 随机深度的比率。默认值为 0.1。
        norm_layer (nn.Module): 归一化层。默认值为 nn.LayerNorm。
        ape (bool): 如果为True，则为patch嵌入添加绝对位置嵌入。默认值为 False。
        patch_norm (bool): 如果为True，则在patch嵌入后添加归一化。默认值为 True。
        use_checkpoint (bool): 是否使用检查点以节省内存。默认值为 False。
        upscale: 放大因子。用于图像超分辨率的值为 2/3/4/8，用于去噪和压缩伪影消除的值为 1。
        img_range: 图像范围。1. 或 255。
        upsampler: 重建模块。'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None。
        resi_connection: 残差连接前的卷积块。'1conv'/'3conv'。
    输入:
        x (Tensor): 输入图像，形状为 (batch_size, in_chans, height, width)。
    输出:
        Tensor: 恢复后的图像，形状为 (batch_size, in_chans, height*upscale, width*upscale)。
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches 将图像分割成不重叠的patch
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image 将不重叠的patch合并成图像
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding  绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 随机深度衰减规则

        # build Residual Swin Transformer blocks (RSTB) 构建残差Swin Transformer块（RSTB）
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction 构建深层特征提取中的最后一个卷积层
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
