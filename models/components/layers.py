import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np


class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion models with ConditionalTimestepTensor support"""
    
    def __init__(self, channel, time_embed_dim, act_fn="silu", out_dim=None, post_act_fn=None, cond_proj_dim=None):
        super().__init__()
        
        self.channel = channel
        self.time_embed_dim = time_embed_dim
        
        # Import TimestepTensor if available
        try:
            from ..tensors import TimestepTensor
            self.TimestepTensor = TimestepTensor
        except ImportError:
            self.TimestepTensor = None
        
        self.linear_1 = nn.Linear(channel, time_embed_dim)
        
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, time_embed_dim, bias=False)
        else:
            self.cond_proj = None
            
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()
            
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)
        
        if post_act_fn is None:
            self.post_act = None
        elif post_act_fn == "silu":
            self.post_act = nn.SiLU()
        elif post_act_fn == "mish":
            self.post_act = nn.Mish()
        else:
            self.post_act = None
            
    def forward(self, sample, condition=None):
        # Handle TimestepTensor - it automatically converts to float for linear operations
        if self.TimestepTensor and isinstance(sample, self.TimestepTensor):
            # TimestepTensor automatically handles dtype conversion
            if sample.shape[-1] != self.channel:
                # Generate sinusoidal embeddings if input is 1D timesteps
                sample = sample.get_sinusoidal_embeddings(self.channel)
            # TimestepTensor will auto-convert to float in linear operations
        else:
            # Fallback for regular tensors
            if not torch.is_tensor(sample):
                sample = torch.tensor(sample, dtype=torch.float32)
                
            if sample.ndim == 0:
                sample = sample.unsqueeze(0)
                
            # Convert to float if needed
            if sample.dtype in [torch.int32, torch.int64, torch.long]:
                sample = sample.float()
            elif sample.dtype != torch.float32:
                sample = sample.to(torch.float32)
                
            # Generate embeddings if needed
            if sample.shape[-1] != self.channel:
                if sample.ndim == 1:
                    # Generate sinusoidal embeddings
                    sample = self._generate_timestep_embedding(sample)
                else:
                    raise ValueError(f"Expected input dimension {self.channel}, got {sample.shape[-1]}")
        
        # Linear operations will automatically handle TimestepTensor conversion
        sample = self.linear_1(sample)
        
        if condition is not None:
            condition = condition.to(sample.dtype)
            sample = sample + self.cond_proj(condition)
            
        sample = self.act(sample)
        sample = self.linear_2(sample)
        
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
    
    def _generate_timestep_embedding(self, timesteps):
        """Generate sinusoidal timestep embeddings"""
        embedding_dim = self.channel
        half_dim = embedding_dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
            
        return emb


def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1, max_period=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    """
    # Ensure timesteps is a tensor
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, dtype=torch.float32)
    
    # Ensure proper dtype and shape
    if timesteps.dtype in [torch.int32, torch.int64, torch.long]:
        timesteps = timesteps.float()
    
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
        
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    
    emb = torch.exp(exponent)
    emb = timesteps[:, None] * emb[None, :]
    
    # Scale embeddings
    emb = scale * emb
    
    # Concat sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    else:
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
    # Zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ResnetBlock(nn.Module):
    """ResNet block with time embedding and optional conditioning"""
    
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, 
                 groups=32, groups_out=None, pre_norm=True, eps=1e-6, non_linearity="swish", 
                 time_embedding_norm="default", kernel=None, output_scale_factor=1.0, use_in_shortcut=None,
                 up=False, down=False):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        
        if groups_out is None:
            groups_out = groups
            
        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
            
        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
            
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()
        else:
            self.nonlinearity = nn.SiLU()
            
        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample(in_channels, use_conv=False)
                
        self.use_in_shortcut = self.in_channels != out_channels if use_in_shortcut is None else use_in_shortcut
        
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, input_tensor, temb, *args, **kwargs):
        hidden_states = input_tensor
        
        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)
            
        hidden_states = self.nonlinearity(hidden_states)
        
        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
            
        hidden_states = self.conv1(hidden_states)
        
        if self.time_emb_proj is not None and temb is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            
        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)
            
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
            
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        
        return output_tensor
    
    @property
    def skip_time_act(self):
        return (
            hasattr(self, "time_embedding_norm") and self.time_embedding_norm == "scale_shift"
        )


class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention"""
    
    def __init__(self, channels, num_head_channels=None, norm_num_groups=32, rescale_output_factor=1.0, eps=1e-5):
        super().__init__()
        self.channels = channels
        
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)
        
        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, bias=True)
        
    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection
        
    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        
        # norm
        hidden_states = self.group_norm(hidden_states)
        
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        
        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)
        
        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)
        
        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
        
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)
        
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)
        
        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        
        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class Downsample(nn.Module):
    """Downsampling layer"""
    
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        
        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
            
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv
            
    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
            
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        
        return hidden_states


class Upsample(nn.Module):
    """Upsampling layer"""
    
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        
        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
            
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv
            
    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels
        
        if self.use_conv_transpose:
            return self.conv(hidden_states)
            
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
            
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
            
        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
            
        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
            
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
                
        return hidden_states


class AdaGroupNorm(nn.Module):
    """Adaptive Group Normalization"""
    
    def __init__(self, embedding_dim, out_dim, num_groups, act_fn=None, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        
        if act_fn is None:
            self.act = None
        elif act_fn == "swish":
            self.act = lambda x: F.silu(x)
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            self.act = None
            
        self.linear = nn.Linear(embedding_dim, out_dim * 2)
        
    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class SpatialNorm(nn.Module):
    """Spatially Adaptive Normalization"""
    
    def __init__(self, f_channels, zq_channels):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, f, zq):
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1, max_period=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    """
    # Handle TimestepTensor
    try:
        from ..tensors import TimestepTensor
        if isinstance(timesteps, TimestepTensor):
            return timesteps.get_sinusoidal_embeddings(
                embedding_dim=embedding_dim,
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=downscale_freq_shift,
                scale=scale,
                max_period=max_period
            )
    except ImportError:
        pass
    
    # Fallback for regular tensors
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, dtype=torch.float32)
    
    # Ensure float for computation
    if timesteps.dtype in [torch.int32, torch.int64, torch.long]:
        timesteps = timesteps.float()
    
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    
    emb = torch.exp(exponent)
    emb = timesteps[:, None] * emb[None, :]
    
    # Scale embeddings
    emb = scale * emb
    
    # Concat sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    else:
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
    # Zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb