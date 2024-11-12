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

class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, emd_dim, downsample=False, attn=None) -> None:
        super().__init__()
        self.attn = attn if attn else nn.Identity()
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
        self.attn = attn if attn else nn.Identity()
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
    def __init__(self, c_ori=3, c_begin=192, n_blocks=3, stacks=2, attn=None):
        super().__init__()
        channel_list = [c_ori] + [(c_begin * (1+i)) for i in range(stacks)]
        channel_list1 = channel_list[1:] + [channel_list[-1]]
        print(channel_list)
        self.n_blocks = n_blocks
        self.stacks = stacks
        self.in_channels = self.out_channels = c_ori
        # self.num_layers = num_layers
        # self.num_filters = num_filters
        self.attn = nn.Identity()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        for i in range(stacks):
            self.encoder.append(self.make_enc_stack(channel_list[i], channel_list[i+1], attn))
            # print(self.encoder)
            self.decoder.append(self.make_dec_stack(channel_list1[i+1], channel_list1[i], attn))
        
        # print(self.encoder)
    
    def make_enc_stack(self, c_in, c_out, attn):
            
        return nn.ModuleList([EncoderBlock(c_in, c_out, 768) if c_in > 3 else Conv(c_in, c_out, 3,1,1)] +
            [EncoderBlock(c_out, c_out, 768, attn=attn) for _ in range(self.n_blocks-2)] + 
            [EncoderBlock(c_out, c_out, 768, downsample=True, attn=attn)])
    
    def make_dec_stack(self, c_in, c_out, attn):

        return nn.ModuleList([DecoderBlock(c_in, c_in, 768, skip=True, attn=attn) for _ in range(self.n_blocks -1)] + 
            [DecoderBlock(c_in, c_in, 768, skip=True, attn=attn) ] + 
            [DecoderBlock(c_out, c_in, 768, upsample=True)])

    def test(self, x, emd):
        skip = []
        skipsize = []
        for stack in self.encoder:
            sub_skip = []
            for enc in stack:
                x = enc(x, emd)
                sub_skip.append(x)
                skipsize.append(x.size())
            sub_skip.append(None)
            skip.append(sub_skip)
        print(skipsize)
        for sub_skip, stack in zip(reversed(skip), reversed(self.decoder)):
            for x_skip, dec in zip(reversed(sub_skip), reversed(stack)):
                if x_skip is not None:
                    print(dec)
                    print(x.shape)
                    print(x_skip.shape)
                    print()
                    x = torch.cat([x, x_skip], dim=1)
                    x = dec(x, emd)
                    print(x.shape)
                else:
                    print(x.shape)
                    x = dec(x, emd)
        
        print('-----------------')
        print(x.shape)
        print('-----------------')
 
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return self.out(x)
    

if __name__ == '__main__':
    emd_layer = time_embedding(512, 768)
    t = torch.arange(0, 5, 1)
    emd = emd_layer(t)
    x = torch.randn(5, 3, 64, 64)
    # enc = EncoderBlock(64, 128, 768, downsample=True)
    # y = enc(x, emd)
    # print(y.shape)
    # print(out)
    unet = Unet()
    unet.test(x, emd)