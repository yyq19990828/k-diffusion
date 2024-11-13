import torch
import torch.nn as nn
import torch.nn.functional as F

class time_embedding(nn.Module):
    def __init__(self, dim_emd, dim_map, max_period=10000):
        super().__init__()
        self.dim = dim_emd
        self.max_period = max_period
        self.label_map = nn.Linear(1, dim_map, bias=False)
        self.time_map = nn.Sequential(nn.Linear(dim_emd, dim_map), 
                                      nn.SiLU(),
                                      nn.Linear(dim_map, dim_map))
        self.last_norm = nn.SiLU()

    def forward(self, t, label=None):
        freqs = torch.pow(10000, torch.linspace(0., 1., self.dim // 2))
        args = t[:, None] / freqs[None]
        args = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        args = self.time_map(args)
        # print(args.shape)
        # print(self.label_map(label.float()).shape)
        return self.last_norm(args) if label is None else self.last_norm(args + self.label_map(label.float()))

class AdaptiveGrpNorm(nn.Module):
    def __init__(self, C, emd_dim, num_groups=32, eps=1e-5):
        super(AdaptiveGrpNorm, self).__init__()
        self.groupnorm = nn.GroupNorm(num_groups, C, eps, affine=False)
        self.WB_map = nn.Linear(emd_dim, C * 2)
    
    def forward(self, x, emd):
        W, B = self.WB_map(emd).chunk(2, dim=-1)
        W = W[:, :, None, None]
        B = B[:, :, None, None]
        # print(W.shape)
        # print(B.shape)
        # print(x.shape)
        return W * self.groupnorm(x) + B

class Downsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Downsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)

class MHSA_2D(nn.Module):
    """
    Multi-Head Self-Attention Module for Image Feature Maps.
    
    Args:
        in_channels (int): Number of input channels.
        num_heads (int): Number of attention heads.
        reduction (int): Reduction ratio for intermediate channels. Default: 8.
    """
    def __init__(self, in_channels, num_heads=8, reduction=8):
        super(MHSA_2D, self).__init__()
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # Dimension per head
        
        # self.reduction = reduction
        # self.inter_channels = (in_channels // reduction) // num_heads
        
        # Single convolution to generate Q, K, V
        # Output channels = 3 * in_channels for Q, K, V
        self.qkv_conv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False)
        
        # Output projection
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Optional: LayerNorm for stabilization
        # self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, x):
        """
        Forward pass of the Multi-Head Self-Attention module.
        
        Args:
            x (Tensor): Input feature maps of shape (B, C, H, W)
        
        Returns:
            out (Tensor): Self-attended feature maps.
        """
        B, C, H, W = x.size()
        
        # Generate Q, K, V
        qkv = self.qkv_conv(x)  # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)  # Each of shape (B, C, H, W)
        
        # Reshape and split into heads
        # New shape: (B, num_heads, C_per_head, H*W)
        def reshape_to_heads(tensor):
            return tensor.view(B, self.num_heads, self.head_dim, H * W)
        
        q = reshape_to_heads(q)  # (B, num_heads, C_per_head, H*W)
        k = reshape_to_heads(k)  # (B, num_heads, C_per_head, H*W)
        v = reshape_to_heads(v)  # (B, num_heads, C_per_head, H*W)
        
        # Compute attention scores
        # Attention scores: (B, num_heads, H*W, H*W)
        attn_scores = torch.einsum('bnch,bnck->bnhk', q, k)  # Alternative to bmm for multiple heads
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, H*W, H*W)
        
        # Weighted sum of V
        # (B, num_heads, C_per_head, H*W)
        attn_output = torch.einsum('bnhw,bnch->bncw', attn_weights, v)
        
        # Reshape back to (B, C, H, W)
        attn_output = attn_output.contiguous().view(B, C, H, W)
        
        # Output projection
        attn_output = self.out_conv(attn_output)
        
        # Apply scaling and residual connection
        out = self.gamma * attn_output + x
        
        return out

class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, emd_dim, downsample=False, attn=None) -> None:
        super().__init__()
        self.attn = attn(ch_out) if attn else nn.Identity()
        self.adaptive_norm = AdaptiveGrpNorm(ch_out, emd_dim)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.SiLU(),
            Downsample(2) if downsample else nn.Identity(),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1))
        
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1),
        )
        self.shortcut = nn.Sequential(
            Downsample(2) if downsample else nn.Identity(),
            nn.Conv2d(ch_in, ch_out, 1, 1, 0)
        ) 

    def forward(self, x, emd):
        # print(x.shape)
        # print(self.block2(self.adaptive_norm(self.block1(x), emd)).shape)
        # print(self.shortcut(x).shape)
        return self.attn(self.shortcut(x) + self.block2(self.adaptive_norm(self.block1(x), emd)))

class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, emd_dim, skip=False, upsample=False, attn=None) -> None:
        super().__init__()
        if skip:
            ch_in *= 2
        self.skip = skip
        self.attn = attn(ch_out) if attn else nn.Identity()
        self.adaptive_norm = AdaptiveGrpNorm(ch_out, emd_dim)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.SiLU(),
            nn.Upsample(scale_factor=2) if upsample else nn.Identity(),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1))
        
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2) if upsample else nn.Identity(),
            nn.Conv2d(ch_in, ch_out, 1, 1, 0)
        ) 
    def forward(self, x, emd):
        return self.attn(self.shortcut(x) + self.block2(self.adaptive_norm(self.block1(x), emd)))

class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x, emd):
        return super().forward(x)
    
class Unet(nn.Module):
    def __init__(self, c_ori=3, c_begin=192, n_blocks=4, stacks=2, attn=MHSA_2D):
        super().__init__()
        channel_list = [c_ori] + [(c_begin * (1+i)) for i in range(stacks)]
        channel_list1 = list(reversed(channel_list[1:] + [channel_list[-1]]))
        # print(channel_list)
        # print(channel_list1)
        self.n_blocks = n_blocks
        self.stacks = stacks
        self.in_channels = self.out_channels = c_ori
        # self.num_layers = num_layers
        # self.num_filters = num_filters
        self.attn = attn
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.out = nn.Sequential(nn.GroupNorm(32, c_begin), 
                                nn.SiLU(), 
                                nn.Conv2d(c_begin, c_ori, 3, 1, 1))

        for i in range(stacks):
            if i==0:
                self.encoder.append(self.make_enc_stack1(channel_list[i], channel_list[i+1], attn=None))
            else:
                self.encoder.append(self.make_enc_stack2(channel_list[i], channel_list[i+1], attn=self.attn))
        
        for i in range(stacks+1):
            if i==0:
                self.decoder.append(nn.ModuleList([DecoderBlock(channel_list1[i], channel_list1[i], 768)]))
            elif i==1:
                self.decoder.append(self.make_dec_stack(channel_list1[i-1], channel_list1[i], attn=self.attn))
            elif i==stacks:
                self.decoder.append(self.make_dec_stack(channel_list1[i-1], channel_list1[i], upsample=True ,attn=None))
            else:
                self.decoder.append(self.make_dec_stack(channel_list1[i-1], channel_list1[i], upsample=True, attn=self.attn))
        # print(self.decoder)
    
    def make_enc_stack1(self, c_in, c_out, attn):
        '''1+n'''
        # EncoderBlock(c_in, c_out, 768, downsample=False, attn=attn)
            
        return nn.ModuleList([Conv(c_in, c_out, 3,1,1)] +
            [EncoderBlock(c_out, c_out, 768, attn=attn) for _ in range(self.n_blocks-1)] 
            )

    def make_enc_stack2(self, c_in, c_out, attn):
        '''down+降维A+2A'''
        # EncoderBlock(c_in, c_out, 768, downsample=False, attn=attn)
            
        return nn.ModuleList([EncoderBlock(c_in, c_in, 768, downsample=True)] +
            [EncoderBlock(c_in, c_out, 768, attn=attn)] +
            [EncoderBlock(c_out, c_out, 768, attn=attn) for _ in range(self.n_blocks-2)] 
            )
    
    def make_dec_stack(self, c_in, c_out, upsample=False, attn=None):
        # DecoderBlock(c_in, c_in, 768, skip=True, upsample=False, attn=attn)

        return nn.ModuleList(
            [DecoderBlock(c_in, c_in, 768, skip=False, upsample=upsample)] + 
            [DecoderBlock(c_in, c_out, 768, skip=True, attn=attn) ] + 
            [DecoderBlock(c_out, c_out, 768, skip=True, attn=attn) for _ in range(self.n_blocks-1)]
        )

    def test(self, x, emd):
        skip = []
        skipsize = []
        for stack in self.encoder:
            sub_skip = []
            for enc in stack:
                x = enc(x, emd)
                skip.append(x)
                skipsize.append(x.size())
        print(skipsize)
        # print(skip)
        for stack in self.decoder:
            for dec in stack:
                if dec.skip:
                    print(x.shape)
                    print(skip[-1].shape)
                    x_skip = skip.pop()
                    print()
                    if x_skip.size(1) != x.size(1):
                        conv1X1 = nn.Conv2d(x_skip.size(1), x.size(1), 1, 1, 0)
                        x_skip = conv1X1(x_skip)
                    x = torch.cat([x, x_skip], dim=1)
                    x = dec(x, emd)
                    print(x.shape)
                else:
                    print(x.shape)
                    x = dec(x, emd)
        
        print('-----------------')
        print(x.shape)
        print('-----------------')

    def forward(self, x, emd):
        skip = []
        for stack in self.encoder:
            sub_skip = []
            for enc in stack:
                x = enc(x, emd)
                skip.append(x)
        for stack in self.decoder:
            for dec in stack:
                if dec.skip:
                    x_skip = skip.pop()
                    if x_skip.size(1) != x.size(1):
                        conv1X1 = nn.Conv2d(x_skip.size(1), x.size(1), 1, 1, 0)
                        x_skip = conv1X1(x_skip)
                    x = torch.cat([x, x_skip], dim=1)
                    x = dec(x, emd)
                else:
                    x = dec(x, emd)
        return self.out(x)
    

if __name__ == '__main__':
    emd_layer = time_embedding(512, 768)
    t = torch.arange(0, 5, 1)
    emd = emd_layer(t)
    x = torch.randn(5, 3, 28, 28)
    # enc = EncoderBlock(64, 128, 768, downsample=True,attn=MHSA_2D)
    # y = enc(x, emd)
    # print(y.shape)
    # print(out)
    unet = Unet()
    x = unet(x, emd)
    print(x.shape)