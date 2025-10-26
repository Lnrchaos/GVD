import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from .attention import SpatialTransformer
from .layers import ResnetBlock, Downsample, Upsample, TimestepEmbedding


class NoiseScheduler:
    """DDPM noise scheduler with improved sampling"""
    
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 beta_schedule="linear", prediction_type="epsilon"):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)
    
    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self, model_output, timestep, sample, generator=None):
        t = timestep
        
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = self.sqrt_alphas_cumprod[t] * sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        
        # Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = self.posterior_mean_coef1[t]
        current_sample_coeff = self.posterior_mean_coef2[t]
        
        # Compute predicted previous sample μ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise
        variance = 0
        if t > 0:
            noise = torch.randn_like(sample, generator=generator)
            variance = (self.posterior_variance[t] ** 0.5) * noise
            
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample


class DiffusionUNet(nn.Module):
    """Advanced U-Net with attention and conditioning"""
    
    def __init__(
        self,
        sample_size=64,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention=False,
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=1280,
        attention_head_dim=8,
        dual_cross_attention=False,
        use_linear_projection=False,
        class_embed_type=None,
        num_class_embeds=None,
        upcast_attention=False,
        resnet_time_scale_shift="default",
    ):
        super().__init__()
        
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center_input_sample = center_input_sample
        self.num_unet_blocks = len(block_out_channels)
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        
        # Time embedding
        time_embed_dim = block_out_channels[0] * 4
        
        # Simple time projection that works
        self.time_proj = TimestepEmbedding(block_out_channels[0], block_out_channels[0])
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)
        
        # Class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(1, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None
            
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            if down_block_type == "CrossAttnDownBlock2D":
                down_block = CrossAttnDownBlock2D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    downsample_padding=downsample_padding,
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            else:
                down_block = DownBlock2D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            self.down_blocks.append(down_block)
            
        # Mid block
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        else:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            
        # Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            
            if not is_final_block:
                add_upsample = True
            else:
                add_upsample = False
                
            if up_block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            else:
                up_block = UpBlock2D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
            
        # Output
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        
    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, return_dict=True):
        # 0. center input if necessary
        if self.center_input_sample:
            sample = 2 * sample - 1.0
            
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
            
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            emb = emb + class_emb
            
        # 2. pre-process
        sample = self.conv_in(sample)
        
        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
            down_block_res_samples += res_samples
            
        # 4. mid
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = self.mid_block(sample, emb)
            
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            if not is_final_block and hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples
                )
                
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        if not return_dict:
            return (sample,)
            
        return {"sample": sample}
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Additional block implementations
class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, temb_channels, add_downsample=True, **kwargs):
        super().__init__()
        self.has_cross_attention = True
        self.num_layers = num_layers
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, temb_channels=temb_channels, dropout=0.0)
            for i in range(num_layers)
        ])
        
        self.attentions = nn.ModuleList([
            SpatialTransformer(out_channels, 8, 64) for _ in range(num_layers)
        ])
        
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample(out_channels, use_conv=True)])
        else:
            self.downsamplers = None
            
    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()
        
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states += (hidden_states,)
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
            
        return hidden_states, output_states
        
class DownBlock2D(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, temb_channels, add_downsample=True, **kwargs):
        super().__init__()
        self.has_cross_attention = False
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, temb_channels=temb_channels, dropout=0.0)
            for i in range(num_layers)
        ])
        
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample(out_channels, use_conv=True)])
        else:
            self.downsamplers = None
            
    def forward(self, hidden_states, temb=None):
        output_states = ()
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
            
        return hidden_states, output_states
        
class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_channels, temb_channels, **kwargs):
        super().__init__()
        self.has_cross_attention = True
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, dropout=0.0),
            ResnetBlock(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, dropout=0.0)
        ])
        
        self.attentions = nn.ModuleList([
            SpatialTransformer(in_channels, 8, 64)
        ])
        
    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states
        
class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels, temb_channels, **kwargs):
        super().__init__()
        self.has_cross_attention = False
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, dropout=0.0),
            ResnetBlock(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, dropout=0.0)
        ])
        
    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states
        
class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, prev_output_channel, temb_channels, add_upsample=True, **kwargs):
        super().__init__()
        self.has_cross_attention = True
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels=in_channels + prev_output_channel if i == 0 else out_channels + prev_output_channel, 
                       out_channels=out_channels, temb_channels=temb_channels, dropout=0.0)
            for i in range(num_layers)
        ])
        
        self.attentions = nn.ModuleList([
            SpatialTransformer(out_channels, 8, 64) for _ in range(num_layers)
        ])
        
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, use_conv=True)])
        else:
            self.upsamplers = None
            
    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None):
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            res_hidden_states = res_hidden_states_tuple[i]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
                
        return hidden_states
        
class UpBlock2D(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, prev_output_channel, temb_channels, add_upsample=True, **kwargs):
        super().__init__()
        self.has_cross_attention = False
        
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels=in_channels + prev_output_channel if i == 0 else out_channels + prev_output_channel, 
                       out_channels=out_channels, temb_channels=temb_channels, dropout=0.0)
            for i in range(num_layers)
        ])
        
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, use_conv=True)])
        else:
            self.upsamplers = None
            
    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for i, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[i]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
                
        return hidden_states