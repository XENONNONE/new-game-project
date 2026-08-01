import torch.nn as nn
from models.vq.resnet import Resnet1D, CausalResnet1D


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)         
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,                        
            stride=stride, 
            padding=0,                          # no padding here
            dilation=dilation
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 causal=False):
        super().__init__()
        self.causal = causal

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # First convolution layer
        if causal:
            blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        else:
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            # Downsampling convolution
            if causal:
                down_conv = CausalConv1d(input_dim, width, filter_t, stride_t, 1)
            else:
                down_conv = nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t)
            
            block = nn.Sequential(
                down_conv,
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm) if causal else Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        
        # Final convolution layer
        if causal:
            blocks.append(CausalConv1d(width, output_emb_width, 3, 1, 1))
        else:
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 causal=False):
        super().__init__()
        self.causal = causal
        blocks = []

        # First convolution layer
        if causal:
            blocks.append(CausalConv1d(output_emb_width, width, 3, 1, 1))
        else:
            blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            out_dim = width
            # Upsampling convolution
            if causal:
                up_conv = CausalConv1d(width, out_dim, 3, 1, 1)
            else:
                up_conv = nn.Conv1d(width, out_dim, 3, 1, 1)
                
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm) if causal else Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                up_conv
            )
            blocks.append(block)
        
        # Final convolution layers
        if causal:
            blocks.append(CausalConv1d(width, width, 3, 1, 1))
        else:
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        if causal:
            blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))
        else:
            blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)