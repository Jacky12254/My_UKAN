import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .kan import KANLinear, KAN


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=8, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
def swish(x):
    
    return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):#下采样
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):#上采样
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x
    
class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=Swish
        grid_eps=0.02
        grid_range=[-1, 1]

        self.fc1 = KANLinear(
                    in_features,
                    hidden_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()

        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,drop_path=0.,norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, dim),
        )

        self.kan = kan(in_features=dim, hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, temb):

        temb = self.temb_proj(temb)
        x = self.drop_path(self.kan(self.norm2(x), H, W))
        x = x + temb.unsqueeze(1)

        return x

class DWConv(nn.Module): #深度可分离卷积
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module): #深度可分离卷积
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.GroupNorm(32, dim)#归一化，参数为均值和标准差
        # self.relu = Swish()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = swish(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class SingleConv(nn.Module):#单卷积
    def __init__(self, in_ch, h_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Conv2d(h_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]


class D_SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32,h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input,temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, h_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(h_ch, h_ch, 3, stride=1, padding=1),
        )
        if in_ch != h_ch:
            self.shortcut = nn.Conv2d(in_ch, h_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(h_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UKan_Hybrid(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):#T是迭代次数，ch是通道数，ch_mult是通道数的倍数，attn是注意力机制，num_res_blocks是残差块的数量，dropout是丢弃率
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index h of bound'#这里是判断attn的索引是否超出范围
        tdim = ch * 4#tdim是时间维度的维度
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        attn = []
        self.head = nn.Conv2d(8, ch, kernel_size=3, stride=1, padding=1)#这里是卷积层，输入通道数是8，输出通道数是ch，卷积核大小是3*3，步长是1，填充是1

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record hput channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):#enumerate是枚举，返回的是索引和值
            h_ch = ch * mult#h_ch是h的通道数，ch是输入通道数，mult是倍数
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:#这里是判断是否是最后一个，如果不是最后一个，就添加下采样
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):#reversed是反转，enumerate是枚举，返回的是索引和值
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch#h_ch是h的通道数，h是输入通道数，mult是倍数
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),#这里是归一化
            Swish(),
            nn.Conv2d(now_ch, 8, 3, stride=1, padding=1)
        )#这里是尾部，输入通道数是now_ch，输出通道数是3，卷积核大小是3*3，步长是1，填充是1

        # 
        # embed_dims = [256, 320, 512]#这是嵌入维度
        embed_dims = [512, 640, 1024]
        norm_layer = nn.LayerNorm#这是归一化层
        dpr = [0.0, 0.0, 0.0]

        # self.patch_embed3 = nn.Conv2d(512, embed_dims[0], kernel_size=1, stride=1, padding=0)

        self.patch_embed3 = OverlapPatchEmbed(img_size=64 // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])#patch_size是指patch的大小，stride是指步长，in_chans是指输入通道数，embed_dim是指嵌入维度
        self.patch_embed4 = OverlapPatchEmbed(img_size=64 // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        
        

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        self.kan_block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1],  mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer)])

        self.kan_block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2],  mlp_ratio=1, drop_path=dpr[1], norm_layer=norm_layer)])

        self.kan_dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer)])

        self.decoder1 = D_SingleConv(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_SingleConv(embed_dims[1], embed_dims[0])  

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
    
        t3 = h

        B = x.shape[0]
        result = self.patch_embed3(h)
        h, H, W = result[:3]
        # h, H, W, _ = self.patch_embed3(h)#h

        for i, blk in enumerate(self.kan_block1):
            h = blk(h, H, W, temb)
        h = self.norm3(h)#这里是归一化
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#这里是reshape，将h的维度变为[B, H, W, -1]，然后转置为[B, -1, H, W]
        t4 = h

        h, H, W= self.patch_embed4(h)
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, H, W, temb)
        h = self.norm4(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        h = swish(F.interpolate(self.decoder1(h, temb), scale_factor=(2,2), mode ='bilinear'))

        h = torch.add(h, t4)

        _, _, H, W = h.shape
        h = h.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.kan_dblock1):
            h = blk(h, H, W, temb)#blk是shiftedBlock

            
        ### Stage 3
        h = self.dnorm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        h = swish(F.interpolate(self.decoder2(h, temb),scale_factor=(2,2),mode ='bilinear'))

        h = F.interpolate(h, size=t3.shape[2:], mode='bilinear', align_corners=False)#插值
        h = torch.add(h,t3)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)#这里是尾部，

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UKan_Hybrid(
        T=1000, ch=64, ch_mult=[1, 2, 2, 2], attn=[],
        num_res_blocks=4, dropout=0.1)
    print(model)

