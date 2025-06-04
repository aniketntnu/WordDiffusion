from abc import abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

from inspect import isfunction
import math
import random
import pickle

from config import *

#from configNor import *
#torch.autograd.set_detect_anomaly(True)

print("\n\t loading fraom unet.py")

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        
        ctx.input_tensors = [x.float().detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding




# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttention(nn.Module):
    def __init__(self, args,query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.args = args
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        
        #attn = torch.nn.functional.softmax(sim, dim=-1)
        #print("\n\t debug attn.shape:",attn.shape," \t sim.shape:",sim.shape)

        if attn.shape[-1]==10 and self.args.attentionMaps==1:
            attnCopy = attn #deepcopy(attn) 
            
            #print("\n\t attention modulation!!")
            
            if attn.shape[1]==256:
                attnCopy = rearrange(attnCopy, 'b (h w) n -> b h w n', h=8, w=32)
            elif attn.shape[1]==64:
                attnCopy = rearrange(attnCopy, 'b (h w) n -> b h w n', h=4, w=16)
            
            #attnCopy[:, :-10, :, :] = attnCopy[:, 10:, :, :]

            #print("\n\t 11.attnCopy =",attnCopy.shape)

            #attnCopyTemp = deepcopy(attnCopy)

            #attnCopy[...,0] =  attnCopy[...,0]
            #attnCopy[...,1] =  attnCopy[...,1]
            #attn[...,1] = 1.5 * attn[...,1]
            #attn = torch.cat([attn[..., :1], 5 * attn[..., 1:2], attn[..., 2:]], dim=-1)

            if 0:
                #print("\n\t ROI")
                # Define the region of interest (ROI) to increase attention values
                # For this example, let's say we want to increase attention in the middle part
                roi_start_h = 4  # Start height index
                roi_end_h = 6    # End height index
                roi_start_w = 10  # Start width index
                roi_end_w = 20   # End width index

                # Define the increment value for the attention
                increment_factor = 1.5
                # Increase the attention values in the specified region
                attnCopy[:, :, roi_start_h:roi_end_h, roi_start_w:roi_end_w] *= increment_factor

                # Ensure the attention values do not exceed 1 (if attention values are normalized between 0 and 1)
                attnCopy = torch.clamp(attnCopy, 0, 1)



            if attnCopy.shape[1]==8:
                ##print("*")
                attn = rearrange(attnCopy, 'b h w n-> b (h w) n', h=8, w=32)
            elif attnCopy.shape[1]==4:
                ##print("#")
                attn = rearrange(attnCopy, 'b h w n-> b (h w) n', h=4, w=16)

        #attn =1 + 0 * attn
        
        if 0:#attn.shape[-1]==10:
            
            #max_neg_value = -torch.finfo(attn.dtype).max
            
            ##print("\n\t setting 0")
            attn[:, :, 2] = -1 #max_neg_value
            #attn[:, :, 1] = max_neg_value
            
            context[:, 2, :] = -1
            #context[:, 1, :] = 0
            #attn[:, :, 2] = 0
            #attn[:, :, 3] = 0
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        """
            adding 1 extra dimension for the head
        """
        attn = attn.unsqueeze(1)
        attn = rearrange(attn,'(b h) 1 n d -> b h n d', h =h)
        
        return self.to_out(out),attn
        
        
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    #'seq shape', seq.shape)
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class BasicTransformerBlock(nn.Module):
    def __init__(self,args, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.args = args
        self.attn1 = CrossAttention(self.args,query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the image
        self.attnc = CrossAttention(self.args,query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the context
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(self.args,query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        #return self._forward, (x, context), self.parameters(), self.checkpoint
        #return self._forward(x, context)

    
    def _forward(self, x, context=None):
        """
        
        batch head (height width) noChars
        
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=None) + x
        x = self.ff(self.norm3(x)) + x
        return x
        """
        #x = self.attn1(self.norm1(x)) + x

        x1,attn = self.attn1(self.norm2(x), context=context, mask=None) 
        x = x1+x
           
        x1,attn = self.attn2(self.norm2(x), context=context, mask=None) 
        x = x1+x
                
        #x,attn = self.attn2(self.norm2(x), context=context, mask=None) + x
        x = self.ff(self.norm3(x)) + x
        return x,attn

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self,args, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, part='encoder', vocab_size=None):
        super().__init__()
        
        self.args = args
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(args,inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        self.part = part
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        
        #print('\n\t\t\t x spatial trans in', x.shape," context.shape=",context.shape)
        
        
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.part != 'sca':
            x = rearrange(x, 'b c h w -> b (h w) c')
    
        for block in self.transformer_blocks:
            x,attn = block(x, context=context)
        
        #print("\n\t\t 1. MAX_CHARS:",MAX_CHARS)
        
        """
            this bring back height and width of the attention
        """
        if self.part != 'sca':
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            attn = rearrange(attn,'batch head (height width) noChars -> batch head height width noChars', height=h, width=w, noChars =10)

        x = self.proj_out(x)
        
        if self.args.attentionMaps ==1:
            return x + x_in,attn
        elif self.args.attentionMaps ==0:
            return x + x_in

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, context):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        
        #print("\n\t\t 360. x.shape:",x.shape)
        
        for layer in self:
            
            #print("\n\t\t\t TimestepEmbedSequential layer:",layer.__class__.__name__)
            
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
                
            elif isinstance(layer, SpatialTransformer):
                #print("\n\t\t calling SpatialTransformer:::",x.shape)
                x = layer(x, context)
                
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        
        #print("\n\t use_conv:",use_conv)
        
        print("\n\t dims:",dims," stride:",stride," use_conv:",use_conv)
        
        if use_conv:
            
            #print("#############")
            
            self.op = nn.Conv2d(#dims,
                 self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            #print("\n\t dims:",dims," stride:",stride)
            #print("========")
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        # if context is None:
        #     context= torch.zeros(emb.shape).to(emb.device)
        
        # emb = torch.cat([emb, context], dim=-1)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h





class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


##################################################################################

    
class Word_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Word_Attention, self).__init__()
        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        #print("\n\t 0.x.shape:",x.shape," x.device:",x.device)
        # x shape: (batch_size, seq_len, input_size)
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)
        
        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)
        scores = self.softmax(scores)
        
        # Calculate weighted sum of the values
        word_embedding = scores @ value
        return word_embedding


class CharacterEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len,args=None):
        super(CharacterEncoder, self).__init__()
        
        self.args = args
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Word_Attention(hidden_size, hidden_size)

        self.embedding_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        #print("\n\t 0.x.shape:",x.shape) # x.shape: torch.Size([2, 16, 320])

        if self.args is not None and getattr(self.args, 'charLevelEmb', False):
            x = x.unsqueeze(1)         
            x = torch.flatten(x, start_dim=0)  
            x = x.unsqueeze(1) # Shape: (BS, 10)       
        
        x = self.embedding(x) # Shape: (10*BS, 320)
        
        if self.args is not None and getattr(self.args, 'charLevelEmb', False):
            BS = x.size(0) // 10  # Calculate BS: 80 // 10 = 8
            x = x.view(BS, 10, 320)  # Shape: (8, 10, 320)

        #Remove positional encoding for ablation study
        
        #print("\n\t 1.x.shape:",x.shape) # x.shape: torch.Size([2, 16, 320])
        #input("check!!!")
        
        #exit()
        x += self.positional_encoding[:x.size(1), :].to(x.device)
        word_embedding = self.attention(x)
        return word_embedding
    
    def get_positional_encoding(self):
        positional_encoding = torch.zeros(self.max_seq_len, self.embedding_dim)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / self.embedding_dim)))
        return positional_encoding



class ResBlockConditional(TimestepBlock):
    
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        #print("\n\t channels:",channels)
        
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, True, dims)
            self.x_upd = Downsample(channels, True, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    """
    
	 x_t.shape: torch.Size([2, 32, 8, 32])

	 
	 4.x.shape: torch.Size([2, 32, 4, 16])

	 5.h.shape: torch.Size([2, 320, 2, 8])
    
    """

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            
            #print("\n\t 1.x.shape:",x.shape)    #    1.x.shape: torch.Size([2, 32, 8, 32])
            h = in_rest(x)
            #print("\n\t 2.h.shape:",h.shape)    # 	 2.h.shape: torch.Size([2, 32, 8, 32])

            h = self.h_upd(h)                
            
            #print("\n\t 3.h.shape:",h.shape)    # 	 3.h.shape: torch.Size([2, 32, 4, 16])

            h = self.h_upd(h)

            #print("\n\t 33.h.shape:",h.shape)   # 	 33.h.shape: torch.Size([2, 32, 2, 8])

            x = self.x_upd(x)
            #print("\n\t 4.x.shape:",x.shape)    # 	 4.x.shape: torch.Size([2, 32, 4, 16])

            #x = self.x_upd(x)
            #print("\n\t 44.x.shape:",x.shape)    # 	 4.x.shape: torch.Size([2, 32, 4, 16])

            #x = self.x_upd(x)
            
            h = in_conv(h)
            #print("\n\t 5.h.shape:",h.shape) 	#    5.h.shape: torch.Size([2, 320, 2, 8])

        else:
            h = self.in_layers(x)
            
        # if context is None:
        #     context= torch.zeros(emb.shape).to(emb.device)
        
        # emb = torch.cat([emb, context], dim=-1)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        #print("\n\t h.shape:",h.shape)
        
        if h.shape[-1] <=10:
            h = F.pad(h,(0,10-h.shape[-1]))
            h = F.max_pool2d(h, kernel_size=(h.shape[2], 1))
            
            #print("\n\t 1.columnwise max pool:",h.shape)
            
            h = h.squeeze(2)
            
            #print("\n\t 2.columnwise max pool:",h.shape)

        return  h


##################################################################################

class CTCtopC(nn.Module):
    def __init__(self, input_size, head_cfg, nclasses):
        super(CTCtopC, self).__init__()

        hidden_size, num_layers = head_cfg

        self.temporal_i = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
        )

        list = [
            nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
                nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
            ) for _ in range(num_layers)
        ]

        self.temporal_m = nn.ModuleList(list)

        self.temporal_o = nn.Conv2d(hidden_size, nclasses, kernel_size=(1, 5), stride=1, padding=(0, 2))

        self.lin1 = nn.Linear(32,128)
        self.lin2 = nn.Linear(128,256)

    def forward(self, x):

        y = self.temporal_i(x)

        for f in self.temporal_m:
            y = f(y)

        y = self.temporal_o(y)
        
        y = self.lin1(y)
        y = self.lin2(y)

        y = y.permute(2, 3, 0, 1)[0]
        return y



class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=768,                 # custom transformer support
        vocab_size=256,                  # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=False,
        args=None, 
        max_seq_len=20,
        #mix_rate=0.5
    ):
        super().__init__()
        
        with open("/cluster/datastore/aniketag/allData/wordStylist/writerStyle/cropStyleDict_Numpy.pkl", 'rb') as f:
            # Load the object from the pickle file
            self.cropStyleDict = pickle.load(f)

        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.args = args
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.args = args

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # 
        self.max_seq_len = max_seq_len
        
        if self.args.charLevelEmb:
            self.word_emb = CharacterEncoder(vocab_size, context_dim, max_seq_len,args).to(args.device)

        else:
            self.word_emb = CharacterEncoder(vocab_size, context_dim, max_seq_len).to(args.device)
        
        ############################### FOR HANDLING CHARACTER CONTEXT IMAGES ############################### 
        
        if self.args.charImages == 1:
            
            self.conv_layer1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(4, 16))  # Reduce height and width by half

            # Define the second convolutional layer
            self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=160, kernel_size=(4, 12))  # Further reduce to 1x1
            self.conv_layer3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(2, 6))  # Further reduce to 1x1

            # Apply the convolutions
            """
            img1 = conv_layer1(img)

            print("\n\t img1.shape:",img1.shape)
            output = conv_layer2(img1)
            print("\n\t img2.shape:",output.shape)

            output = conv_layer3(output)
            # Check the output shape
            print(output.shape)  # Output: torch.Size([1, 320, 1, 1])
            """
                
        ############################################################################################################################

        
        #==================== INPUT BLOCK ====================
        
        self.wrd_proj = nn.Linear(4096, 320)
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [   
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            self.args,ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        #==================== MIDDLE BLOCK ====================
        

        self._feature_size += ch


        if self.args.attentionMaps == 1:
            self.middle_block1 = nn.ModuleList([])
            
            layers= [   
                    ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,),
                    ]
            
            layers.append(    
                    SpatialTransformer(self.args,ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
                    )
            
            self.middle_block1.append(TimestepEmbedSequential(*layers))

            self.middle_block1.append(TimestepEmbedSequential(

                    ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,)
            ))        
            
        else:
            
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                ) if not use_spatial_transformer else SpatialTransformer(
                                self.args,ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
            
        #==================== OUTPUT BLOCK ====================
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            self.args,ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            nn.Conv2d(model_channels, n_embed, 1),
            nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )
        
        self.interpolation = args.interpolation
        
        if self.args.ocrTraining==1:
            self.auxhead = CTCtopC(4,(256,3),vocab_size-2)
        
        
        self.res = ResBlockConditional(32,1280,0.2,320,use_conv=True,use_scale_shift_norm=False,dims=2,use_checkpoint=False,up=False,down=True)
        #res = self.res.to(device)

        """
        self.res = ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,
        dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True)
        """
                
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
  
    #           model(x_t,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)

    def forward(self, x,wrdChrWrStyl, original_images=None, timesteps=None, context=None, y=None, charContextImages=None,original_context=None, or_images=None, mix_rate=None, **kwargs):
        
        # model(x,"no_wrdChrWrStyl",original_images=None,timesteps=t,context=text_features,y=labels,charContextImages=charJoinImg)

        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #print('y', y.shape)
        
        #print("\n\t 1.charContextImages=",charContextImages.shape)

        #print("&&&&&&&&&&&&&&&&")
        
        if self.args.charImages == 1:

            BS = charContextImages.size(0) 
            
            #print("\n\t BS =",BS," self.max_seq_len:",self.max_seq_len," charContextImages.shape:",charContextImages.shape)
            # image.shape: torch.Size([500, 4, 8, 32])
            charContextImages = charContextImages.reshape(self.max_seq_len*BS,4,8,32)
            #print("\n\t 2.charContextImages=",charContextImages.shape)
            
            img1 = self.conv_layer1(charContextImages)

            #print("\n\t img1.shape:",img1.shape)
            output = self.conv_layer2(img1)
            #print("\n\t img2.shape:",output.shape)

            output = self.conv_layer3(output)
            
            output = output.reshape(BS,self.max_seq_len,320,1,1)
            
            # Check the output shape
            #print("\n\t output.shape:",output.shape)  # Output: torch.Size([BS, 10, 320, 1, 1])

            output = output.squeeze()

            #print("\n\t output.shape:",output.shape)  
        
        #print("\n\t self.num_classes:",self.num_classes)
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        """
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        #print( "\n\t shape:",y.shape ,x.shape)
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
        
        #if you want to explore interpolation between 2 random styles you can go to the --interpolation argument in the train.py file
        if self.interpolation:
            if mix_rate is not None:
                print('interpolation')
                s1 = random.randint(0, 338)
                s2 = random.randint(0, 338)
                while s1 == s2:
                    s2 = random.randint(0, 338)
                y1 = torch.tensor([s1]).long().to(x.device)
                y2 = torch.tensor([s2]).long().to(x.device)
                y1 = self.label_emb(y1).to(x.device)
                y2 = self.label_emb(y2).to(x.device)
            
                y = (1-mix_rate)*y1 + mix_rate*y2
                
                y = y.to(x.device)
                emb = emb + y #self.label_emb(y)
            else:
                emb = emb + self.label_emb(y) 
        else:
            
            if self.args.imgConditioned ==1:
                emb = emb #+  self.label_emb(y)  
            else:
                emb = emb  +  self.label_emb(y)
                
        """
            wrdChrWrStyl = torch.from_numpy(self.cropStyleDict["b06-019-00-07.png"])
            wrdChrWrStyl = wrdChrWrStyl.squeeze()
            wrdChrWrStyl = wrdChrWrStyl.to("cuda:0")
            print("\n\t 11.wrdChrWrStyl.shape:",wrdChrWrStyl.shape)#," context.shape:",context.shape,"\t original_images=",original_images.shape)
        """
        
        if self.args.wrdChrWrStyl == 1:
            wrdChrWrStyl = self.wrd_proj(wrdChrWrStyl) 

        if 0:#self.args.imgConditioned ==1:
            original_images = original_images.permute([0,3,1,2])
        
            condImgEmb = self.res(original_images,emb)
            condImgEmb = condImgEmb.permute([0,2,1])
            #print("\n\t original_images.shape:",original_images.shape,"\t emb.shape:",emb.shape,"\t condImgEmb =",condImgEmb.shape)
            
            #print("\n\t condImgEmb =",condImgEmb.shape)
        else:
            pass
            #print("\n\t not adding!!!")
    
        if context is not None:
            #Word embedding
            #print("\n\t context.shape:",context.shape)
            #context[:,0] = 0
            context = self.word_emb(context)
            #output
            #context = context + wrdChrWrStyl
            #print("\n\t context.shape =",context.shape) # [BS,10,320]
            
            #print("context:",context)
            
    
            if self.args.imgConditioned ==0 and self.args.wrdChrWrStyl==1:
                context = wrdChrWrStyl
            elif self.args.imgConditioned ==1 and self.args.charImages == 0:
                
                context = context #+ condImgEmb
            elif self.args.charImages == 1:
                #print("\n\t output.shape:",output.shape)
                context =   context #+ output 
                
                
                # Assume 'output' and 'context' are your input tensors of shape (BS, 10, 320)

                # Extract even elements from 'output' tensor
                #output_even = output[:, ::2, :]

                # Extract odd elements from 'context' tensor
                #context_odd = context[:, 1::2, :]

                # Concatenate the even elements from 'output' and odd elements from 'context'
                #context = torch.cat((output_even, context_odd), dim=1)
                        
        h = x.type(self.dtype)

        #print("\n\t 2.wrdChrWrStyl.shape:",wrdChrWrStyl," context.shape:",context.shape," condImgEmb.shape:",condImgEmb.shape," h.shape:",h.shape," \t emb.shape:",emb.shape)
        #INPUT BLOCKS
        
        #print("\n\t h.shape:",h.shape,"\t emb.shape:",emb.shape,"\t context.shape:",context.shape)
        
        if self.args.attentionMaps == 1:
        
            #INPUT BLOCKS
            for ind, module in enumerate(self.input_blocks):
                
                #print("\n\t INPUT BLOCKS module:",module.__class__.__name__," \t hs.shap:",len(hs))
                #print("\n\t h.shape:",h.shape," emb.shape:",emb.shape," context.shape:",context.shape," \t ind:",ind)
                
                layerName = ""                
                #print(f"\t\t layer: {module.__class__.__name__.split('.')[-1]}")

                if len(module) ==2:
                    inner_module = module[1]
                    #print(f"\t\t layer: {inner_module.__class__.__name__.split('.')[-1]}")
                    layerName = inner_module.__class__.__name__.split('.')[-1]
                
                if layerName == "SpatialTransformer":
                    h,attn1 = module(h, emb, context)
                    #print("\n\t 11.attn =",attn.shape," h.shape:",h.shape)
                    # 	 11.attn = torch.Size([500, 4, 8, 32, 10])  h.shape: torch.Size([500, 320, 8, 32])

                else:
                    h = module(h, emb, context)
                
                hs.append(h)
        
        else:
            for ind,module in enumerate(self.input_blocks):
                
                #print("\n\t INPUT BLOCKS module:",module.__class__.__name__," \t hs.shap:",len(hs))
                #print("\n\t 1.h.shape:",h.shape," emb.shape:",emb.shape," context.shape:",context.shape," \t ind:",ind)
                
                h = module(h, emb, context)
                
                #print("\n\t h.type:",type(h))
                #print("\n\t 2.h.shape:",h.shape," emb.shape:",emb.shape," context.shape:",context.shape," \t ind:",ind)

                hs.append(h)
        
        #MIDDLE BLOCK
        
        #print("\n\t MIDDLE BLOCK layers:",self.args.attentionMaps == 0)
        if self.args.attentionMaps == 1:
            
            for ind,module in enumerate(self.middle_block1):
                
                #print("\n\t 1#####")
                #h = module(h, emb, context)

                layerName = "" #module.__class__.__name__.split('.')[-1]
                
                #print(f"\t\t layer: {module.__class__.__name__.split('.')[-1]}")
    
                if len(module) ==2:
                    
                    inner_module = module[1]

                    #print(f"\t\t layer: {inner_module.__class__.__name__.split('.')[-1]}")
                    layerName = inner_module.__class__.__name__.split('.')[-1]
                
                #print("\n\t *layerName =",layerName)
                        
                if layerName == "SpatialTransformer":
                    h,attn2 = module(h, emb, context)
                    #print("\n\t 2.attn =",attn.shape," h.shape:",h.shape)    
                    # 2.attn = torch.Size([2000, 64, 10])                    
                else:
                    h = module(h, emb, context)            
            
        elif self.args.attentionMaps == 0:
            #print("\n\t 2*************")

            h = self.middle_block(h, emb, context)

        #print("\n\t middle block complete!!!")
        
        if self.args.attentionMaps == 1:

            #OUTPUT BLOCKS
            for ind,module in enumerate(self.output_blocks):
                h = torch.cat([h, hs.pop()], dim=1)
                #print("\n\t h.shape:",h.shape," emb.shape:",emb.shape," context.shape:",context.shape," \t ind:",18+ind)
                #input("check!!")
                            
                layerName = ""
                if len(module) ==2:
                    
                    inner_module = module[1]

                    #print(f"\t\t layer: {inner_module.__class__.__name__.split('.')[-1]}")
                    layerName = inner_module.__class__.__name__.split('.')[-1]

                #print("\n\t *layerName =",layerName)
                
                if layerName == "SpatialTransformer":
                    h,attn3 = module(h, emb, context)
                    #print("\n\t 3.attn =",attn.shape," h.shape:",h.shape)
                    # 3.attn = torch.Size([500, 4, 8, 32, 10])  h.shape: torch.Size([500, 320, 8, 32])
                else:
                    h = module(h, emb, context)


        elif self.args.attentionMaps == 0:
            #OUTPUT BLOCKS
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
            
        h = h.type(x.dtype)
        
        #att1,att2,att3 = None,None,None attentionMaps
        if  self.args.attentionMaps==1:
            attn1 =  attn1.sum(dim =1)
            #print("\n\t 11. sum Attention shape:",attn1.shape)
            # 1. sum Attention shape: torch.Size([500, 8, 32, 10])
            
            attn1 = F.interpolate(attn1.permute(0,3,1,2),scale_factor = (8,8),mode = "nearest")
            attn1 = attn1.permute(0,2,3,1)
            #print("\n\t 12.  Attention reshape:",attn1.shape)

            attn2 =  attn2.sum(dim =1)
            #print("\n\t 21. sum Attention shape:",attn2.shape)
            # 1. sum Attention shape: torch.Size([500, 8, 32, 10])
            
            attn2 = F.interpolate(attn2.permute(0,3,1,2),scale_factor = (16,16),mode = "nearest")
            attn2 = attn2.permute(0,2,3,1)
            #print("\n\t 22.  Attention reshape:",attn2.shape)

            attn3 =  attn3.sum(dim =1)
            #print("\n\t 31. sum Attention shape:",attn3.shape)
            # 1. sum Attention shape: torch.Size([500, 8, 32, 10])
                        
            attn3 = F.interpolate(attn3.permute(0,3,1,2),scale_factor = (8,8),mode = "nearest")
            attn3 = attn3.permute(0,2,3,1)
            #print("\n\t 32.  Attention reshape:",attn3.shape)
            

            """
            BS = attn.shape[0] 
            attn = attn.reshape(BS,1, 64, 256, MAX_CHARS)
            print("\n\t 2. Attention shape:",attn.shape)
            """
            
        #print("\n\t attn1.shape:",attn1.shape,"\n\t att2.shape:",attn2.shape,"\n\t attn3.shape:",attn3.shape)

        if self.predict_codebook_ids:

            if self.args.attentionMaps == 0 and self.args.ocrTraining == 0:
                
                #print("\n\t 1.returning ")
                return self.id_predictor(h)
            
            elif self.args.attentionMaps == 1 and self.args.ocrTraining == 0:
                #print("\n\t 1.2 returning ")

                return self.id_predictor(h),attn1,attn2,attn3
            elif self.args.attentionMaps == 1 and self.args.ocrTraining == 1:

                h = self.id_predictor(h)
                tdec = self.auxhead(h)
                #print("\n\t 1.3 returning ")

                return h,attn1,attn2,attn3,tdec

            else:
                #print("\n\t 1.4 returning ")

                return self.id_predictor(h)#,attn1,attn2,attn3
        else:

            h = self.out(h)

            if self.args.attentionMaps == 0 and self.args.ocrTraining == 0:
                #return self.id_predictor(h)
                
                #print("\n\t 2.returning ")
                return h
                            
            elif self.args.attentionMaps == 1 and self.args.ocrTraining == 0:
                #print("\n\t 2.1.returning ")
                #print("\n\t attn1.shape:",attn1.shape,"\t att2.shape:",attn2.shape,"\t attn3.shape:",attn3.shape)

                return h,attn1,attn2,attn3,context
            elif self.args.attentionMaps == 1 and self.args.ocrTraining == 1:
                tdec = self.auxhead(h)
                #print("\n\t 2.2.returning ")

                return h,attn1,attn2,attn3,tdec                
            else:
                #print("\n\t 2.3.returning ")

                return h #,attn1,attn2,attn3
                

import argparse

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam or other dataset') 
    parser.add_argument('--iam_path', type=str, default='/cluster/datastore/aniketag/allData/wordStylist/washingtondb-v1.0/data/preprocess_words_gw/', help='path to iam dataset (images 64x256)')
    parser.add_argument('--gt_train', type=str, default='./gt/gan.iam.tr_va.gt.filter27')
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./save_path/')
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default="/cluster/datastore/aniketag/WordStylist/stableDiffusion//", help='path to stable diffusion')
    parser.add_argument('--partialLoad', type=int, default=0.001) 
    parser.add_argument('--wrdChrWrStyl', type=int, default=0)
    parser.add_argument('--charImages', type=int, default=0)
    parser.add_argument('--imgConditioned', type=int, default=0)
    parser.add_argument('--attentionMaps', type=int, default=1,help= "return attention maps")
    parser.add_argument('--ocrTraining', type=int, default=1) 

    parser.add_argument('--charLevelEmb', type=int, default=1,help = "the word level embeddings are calculated by concatenating char level embeddings")
    parser.add_argument('--boxLoss', type=int, default=0) 
    parser.add_argument('--textEmb', type=int, default=0) 
    parser.add_argument('--noiseEmb', type=int, default=0) 

    args = parser.parse_args()
    device= args.device

    ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /args.batch_size

    
    maxLen = 10
    x_t_shape = torch.Size([2, 4, 8, 32])
    original_images_shape = torch.Size([2, 3, 64, 256])
    text_features = torch.Size([2, maxLen])
    s_id = torch.Size([2])
    timesteps = torch.Size([2])

    x_t = torch.randn(x_t_shape).to(device).to(torch.float)
    original_images = torch.randn(original_images_shape).to(device).to(torch.int)
    text_features = torch.randn(text_features).to(device).to(torch.int)
    s_id = torch.randn(s_id).to(device).to(torch.int)
    timesteps = torch.randn(timesteps).to(device).to(torch.int)
    
    unet = UNetModel(image_size = (64, 256), in_channels=4, model_channels=320, out_channels=4, num_res_blocks=1, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=4, 
                     num_classes=339, context_dim=320, vocab_size=53, args=args, max_seq_len=maxLen).to(device)    

    
    #print("\n\t unet =\n",unet)

    import pickle # tempData["word"]
    with open('/cluster/datastore/aniketag/newWordStylist/WordStylist/gt/temp.pkl', 'rb') as f:
        tempData = pickle.load(f)

    #print("\n\t rrrr:",tempData["word"][:,:10])
    #input("ccc")
    text_features = tempData["word"][:,:10]
    text_features = torch.from_numpy(text_features)
    text_features = text_features.to(device).to(torch.int)

    print("\n\t text_features.shape:",text_features.shape)

    print("\n\t text_features:",text_features)

    # UNetModel( x,wrdChrWrStyl, original_images=None, timesteps=None, context=None, y=None, charContextImages=None,original_context=None, or_images=None, mix_rate=None, **kwargs):

    if 0:
        state_dict = torch.load("/cluster/datastore/aniketag/allData/wordStylist/models/IAM/authorsModel/models/models/ema_ckpt.pt",map_location=args.device)
        

        if isinstance(unet, torch.nn.DataParallel) or isinstance(unet, torch.nn.parallel.DistributedDataParallel):
            unet.module.load_state_dict(state_dict)
        else:
            unet.load_state_dict(state_dict)

        #unet.load_state_dict(state_dict)


        print("\n\t loaded!!!")
        
    #print("\n\t unet =\n",unet)

    if args.ocrTraining == 0:    

        predicted_noise,attn1,attn2,attn3 = unet(x_t,0, original_images=original_images, timesteps=timesteps, context=text_features, y=s_id)
        print("\n\t predicted_noise =",predicted_noise.shape)
        #                     (x,wrdChrWrStyl, original_images=None, timesteps=None, context=None, y=None, charContextImages=None,original_context=None, or_images=None, mix_rate=None, **kwargs):

    else:
        
        target_sequence_length = 10

        predicted_noise,attn1,attn2,attn3,tdec = unet(x_t,0, original_images=original_images, timesteps=timesteps, context=text_features, y=s_id)
        print("predicted_noise =",predicted_noise.shape," tdec.shape:",tdec.shape)
        
        #tdec =  torch.randn(10, 2, 54).log_softmax(2).detach().requires_grad_() #tdec.log_softmax(2).detach()#.requires_grad()
        transcr = torch.randint(0,54,(args.batch_size,target_sequence_length),dtype = torch.long)
        act_lens = torch.full((args.batch_size,),tdec.shape[0],dtype= torch.long)
        label_lens = torch.full((args.batch_size,),10,dtype= torch.long)
        
        print("\n tdec.shape:",tdec.shape," transcr.shape:",transcr.shape," act_lens:",act_lens," label_lens.shape:",label_lens)
        
        loss_val = ctc_loss(tdec, transcr, act_lens, label_lens)

        print("\n\t loss_val =",loss_val)

        

    #cropStyleDict
    

    """
	 image_name = b06-019-00-07.png

	 image_name = a03-034-00-03.png

	 image_name = a03-034-06-07.png

	 image_name = b06-019-07-00.png

	 image_name = a03-034-03-05.png

	 image_name = a03-034-04-01.png

	 image_name = b06-019-03-01.png

	 image_name = b06-019-02-01.png

	 image_name = b06-019-02-03.png

	 image_name = b06-019-08-02.png    
    
    """