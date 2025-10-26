import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

# Import TimestepTensor for robust timestep handling
try:
    from .tensors import TimestepTensor, random_timesteps, fixed_timesteps
except ImportError:
    # Fallback for testing
    TimestepTensor = torch.Tensor
    def random_timesteps(batch_size, max_timesteps=1000, device=None):
        timesteps = torch.randint(0, max_timesteps, (batch_size,), dtype=torch.long)
        return timesteps.to(device) if device else timesteps
    def fixed_timesteps(timestep_value, batch_size, device=None):
        timesteps = torch.full((batch_size,), timestep_value, dtype=torch.long)
        return timesteps.to(device) if device else timesteps

from .components import (
    VAEEncoder, VAEDecoder, DiagonalGaussianDistribution,
    DiffusionUNet, NoiseScheduler,
    StyleGAN3Generator, ProjectedDiscriminator,
    MultiHeadCrossAttention, SpatialTransformer,
    ResnetBlock, TimestepEmbedding
)


class HybridGenerativeModel(nn.Module):
    """
    Hybrid architecture combining VAE, GAN, and Diffusion models
    Supports multiple generation modes and cross-modal training
    """
    
    def __init__(
        self,
        # Image parameters
        img_resolution: int = 512,
        img_channels: int = 3,
        latent_channels: int = 4,
        
        # VAE parameters
        vae_config: Dict[str, Any] = None,
        
        # Diffusion parameters  
        diffusion_config: Dict[str, Any] = None,
        
        # GAN parameters
        gan_config: Dict[str, Any] = None,
        
        # Hybrid parameters
        cross_attention_dim: int = 1024,
        conditioning_dim: int = 512,
        enable_cross_modal: bool = True,
        
        # Training modes
        training_mode: str = "hybrid"  # "vae", "gan", "diffusion", "hybrid"
    ):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.latent_channels = latent_channels
        self.cross_attention_dim = cross_attention_dim
        self.conditioning_dim = conditioning_dim
        self.enable_cross_modal = enable_cross_modal
        self.training_mode = training_mode
        
        # Initialize VAE components
        self._init_vae(vae_config or {})
        
        # Initialize Diffusion components
        self._init_diffusion(diffusion_config or {})
        
        # Initialize GAN components  
        self._init_gan(gan_config or {})
        
        # Initialize cross-modal components
        if enable_cross_modal:
            self._init_cross_modal()
            
        # Initialize hybrid loss weights
        self.loss_weights = {
            'vae_recon': 1.0,
            'vae_kl': 0.1,
            'diffusion': 1.0,
            'gan_g': 1.0,
            'gan_d': 1.0,
            'cross_modal': 0.5
        }
        
    def _init_vae(self, config):
        """Initialize VAE encoder and decoder"""
        default_config = {
            'ch': 128,
            'out_ch': 3,
            'ch_mult': (1, 2, 4, 8),
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
            'resamp_with_conv': True,
            'in_channels': self.img_channels,
            'resolution': self.img_resolution,
            'z_channels': self.latent_channels,
            'double_z': True,
            'use_linear_attn': False,
            'attn_type': "vanilla"
        }
        default_config.update(config)
        
        self.vae_encoder = VAEEncoder(**default_config)
        self.vae_decoder = VAEDecoder(**default_config)
        
    def _init_diffusion(self, config):
        """Initialize diffusion U-Net and noise scheduler"""
        default_config = {
            'sample_size': self.img_resolution // 8,  # Latent space resolution
            'in_channels': self.latent_channels,
            'out_channels': self.latent_channels,
            'down_block_types': ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            'up_block_types': ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            'block_out_channels': (320, 640, 1280, 1280),
            'layers_per_block': 2,
            'cross_attention_dim': self.cross_attention_dim,
            'attention_head_dim': 8
        }
        default_config.update(config)
        
        self.diffusion_unet = DiffusionUNet(**default_config)
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        
    def _init_gan(self, config):
        """Initialize StyleGAN3 generator and discriminator"""
        default_config = {
            'z_dim': 512,
            'c_dim': self.conditioning_dim,
            'w_dim': 512,
            'img_resolution': self.img_resolution,
            'img_channels': self.img_channels
        }
        default_config.update(config)
        
        self.gan_generator = StyleGAN3Generator(**default_config)
        
        disc_config = {
            'c_dim': self.conditioning_dim,
            'img_resolution': self.img_resolution,
            'img_channels': self.img_channels
        }
        self.gan_discriminator = ProjectedDiscriminator(**disc_config)
        
    def _init_cross_modal(self):
        """Initialize cross-modal attention and fusion layers"""
        # Cross-modal attention between VAE and diffusion latents
        self.vae_to_diffusion_attn = MultiHeadCrossAttention(
            query_dim=self.latent_channels,
            context_dim=self.latent_channels,
            heads=8,
            dim_head=64
        )
        
        # Cross-modal attention between GAN and VAE features
        self.gan_to_vae_attn = MultiHeadCrossAttention(
            query_dim=512,  # GAN w-space
            context_dim=self.latent_channels,
            heads=8,
            dim_head=64
        )
        
        # Fusion layers
        self.latent_fusion = nn.Sequential(
            nn.Linear(self.latent_channels * 2, self.latent_channels),
            nn.SiLU(),
            nn.Linear(self.latent_channels, self.latent_channels)
        )
        
        # Conditioning projections
        self.text_projection = nn.Linear(768, self.cross_attention_dim)  # For text conditioning
        self.image_projection = nn.Linear(self.latent_channels, self.cross_attention_dim)
        
    def encode_vae(self, x: torch.Tensor) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """Encode image to VAE latent space"""
        h = self.vae_encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        return z, posterior
        
    def decode_vae(self, z: torch.Tensor) -> torch.Tensor:
        """Decode VAE latent to image"""
        return self.vae_decoder(z)
        
    def generate_diffusion(
        self, 
        batch_size: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        conditioning: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generate images using diffusion model"""
        device = next(self.parameters()).device
        
        # Sample initial noise
        shape = (batch_size, self.latent_channels, self.img_resolution // 8, self.img_resolution // 8)
        latents = torch.randn(shape, generator=generator, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0, num_inference_steps, device=device).long()
        
        # Prepare conditioning
        if conditioning is not None:
            conditioning = self.text_projection(conditioning)
        
        # Denoising loop
        for t in timesteps:
            # Predict noise
            noise_pred = self.diffusion_unet(
                latents, 
                t.expand(batch_size), 
                encoder_hidden_states=conditioning
            )["sample"]
            
            # Apply guidance
            if guidance_scale > 1.0 and conditioning is not None:
                noise_pred_uncond = self.diffusion_unet(
                    latents, 
                    t.expand(batch_size), 
                    encoder_hidden_states=None
                )["sample"]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # Denoise
            latents = self.noise_scheduler.step(noise_pred, t, latents)
            
        return latents
        
    def generate_gan(
        self, 
        z: torch.Tensor, 
        c: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0
    ) -> torch.Tensor:
        """Generate images using GAN"""
        return self.gan_generator(z, c, truncation_psi=truncation_psi)
        
    def forward(
        self, 
        x: Optional[torch.Tensor] = None,
        mode: str = "auto",
        conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting multiple generation modes
        
        Args:
            x: Input images (for VAE/reconstruction modes)
            mode: Generation mode ("vae", "diffusion", "gan", "hybrid", "auto")
            conditioning: Optional conditioning (text embeddings, etc.)
        """
        outputs = {}
        
        if mode == "auto":
            mode = self.training_mode
            
        # VAE forward pass
        if mode in ["vae", "hybrid"] and x is not None:
            vae_latents, vae_posterior = self.encode_vae(x)
            vae_recon = self.decode_vae(vae_latents)
            
            outputs.update({
                'vae_latents': vae_latents,
                'vae_posterior': vae_posterior,
                'vae_reconstruction': vae_recon
            })
            
        # Diffusion forward pass
        if mode in ["diffusion", "hybrid"]:
            if self.training:
                # Training: add noise and predict
                if 'vae_latents' in outputs:
                    clean_latents = outputs['vae_latents']
                else:
                    # Generate random latents for training
                    batch_size = x.shape[0] if x is not None else 1
                    clean_latents = torch.randn(
                        batch_size, self.latent_channels, 
                        self.img_resolution // 8, self.img_resolution // 8,
                        device=next(self.parameters()).device
                    )
                
                # Sample timesteps and add noise using TimestepTensor
                batch_size = clean_latents.shape[0]
                device = clean_latents.device
                
                # Create TimestepTensor for automatic dtype handling
                timesteps = random_timesteps(
                    batch_size=batch_size,
                    max_timesteps=self.noise_scheduler.num_train_timesteps,
                    device=device
                )
                
                noise = torch.randn_like(clean_latents)
                noisy_latents = self.noise_scheduler.add_noise(clean_latents, noise, timesteps)
                
                # Predict noise
                noise_pred = self.diffusion_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=conditioning
                )["sample"]
                
                outputs.update({
                    'diffusion_noise_pred': noise_pred,
                    'diffusion_noise_target': noise,
                    'diffusion_timesteps': timesteps
                })
            else:
                # Inference: generate from noise
                batch_size = x.shape[0] if x is not None else kwargs.get('batch_size', 1)
                diffusion_latents = self.generate_diffusion(
                    batch_size=batch_size,
                    conditioning=conditioning,
                    **kwargs
                )
                diffusion_images = self.decode_vae(diffusion_latents)
                
                outputs.update({
                    'diffusion_latents': diffusion_latents,
                    'diffusion_images': diffusion_images
                })
                
        # GAN forward pass
        if mode in ["gan", "hybrid"]:
            batch_size = x.shape[0] if x is not None else kwargs.get('batch_size', 1)
            z = torch.randn(batch_size, 512, device=next(self.parameters()).device)
            
            gan_images = self.generate_gan(z, conditioning)
            
            outputs.update({
                'gan_z': z,
                'gan_images': gan_images
            })
            
            # Discriminator forward pass (training only)
            if self.training:
                if x is not None:
                    real_logits = self.gan_discriminator(x, conditioning)
                    outputs['gan_real_logits'] = real_logits
                    
                fake_logits = self.gan_discriminator(gan_images.detach(), conditioning)
                outputs['gan_fake_logits'] = fake_logits
                
        # Cross-modal fusion (hybrid mode)
        if mode == "hybrid" and self.enable_cross_modal:
            self._apply_cross_modal_fusion(outputs)
            
        return outputs
        
    def _apply_cross_modal_fusion(self, outputs: Dict[str, torch.Tensor]):
        """Apply cross-modal attention and fusion"""
        if 'vae_latents' in outputs and 'diffusion_latents' in outputs:
            # Fuse VAE and diffusion latents
            vae_flat = rearrange(outputs['vae_latents'], 'b c h w -> b (h w) c')
            diff_flat = rearrange(outputs['diffusion_latents'], 'b c h w -> b (h w) c')
            
            # Cross-attention
            fused_vae = self.vae_to_diffusion_attn(vae_flat, diff_flat)
            fused_diff = self.vae_to_diffusion_attn(diff_flat, vae_flat)
            
            # Combine and project back
            combined = torch.cat([fused_vae, fused_diff], dim=-1)
            fused_latents = self.latent_fusion(combined)
            
            # Reshape back to spatial dimensions
            h, w = outputs['vae_latents'].shape[-2:]
            fused_latents = rearrange(fused_latents, 'b (h w) c -> b c h w', h=h, w=w)
            
            # Generate fused reconstruction
            fused_images = self.decode_vae(fused_latents)
            
            outputs.update({
                'fused_latents': fused_latents,
                'fused_images': fused_images
            })
            
    def compute_losses(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute losses for all active components"""
        losses = {}
        
        # VAE losses
        if 'vae_reconstruction' in outputs:
            # Reconstruction loss
            recon_loss = F.mse_loss(outputs['vae_reconstruction'], targets)
            losses['vae_recon'] = recon_loss * self.loss_weights['vae_recon']
            
            # KL divergence loss
            if 'vae_posterior' in outputs:
                kl_loss = outputs['vae_posterior'].kl().mean()
                losses['vae_kl'] = kl_loss * self.loss_weights['vae_kl']
                
        # Diffusion loss
        if 'diffusion_noise_pred' in outputs and 'diffusion_noise_target' in outputs:
            diffusion_loss = F.mse_loss(
                outputs['diffusion_noise_pred'], 
                outputs['diffusion_noise_target']
            )
            losses['diffusion'] = diffusion_loss * self.loss_weights['diffusion']
            
        # GAN losses
        if 'gan_real_logits' in outputs and 'gan_fake_logits' in outputs:
            # Discriminator loss
            d_real_loss = F.softplus(-outputs['gan_real_logits']).mean()
            d_fake_loss = F.softplus(outputs['gan_fake_logits']).mean()
            losses['gan_d'] = (d_real_loss + d_fake_loss) * self.loss_weights['gan_d']
            
        if 'gan_images' in outputs:
            # Generator loss (computed separately to avoid gradients through discriminator)
            fake_logits = self.gan_discriminator(outputs['gan_images'])
            g_loss = F.softplus(-fake_logits).mean()
            losses['gan_g'] = g_loss * self.loss_weights['gan_g']
            
        # Cross-modal consistency loss
        if 'fused_images' in outputs:
            cross_modal_loss = F.mse_loss(outputs['fused_images'], targets)
            losses['cross_modal'] = cross_modal_loss * self.loss_weights['cross_modal']
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
        
    def set_training_mode(self, mode: str):
        """Set training mode for different components"""
        self.training_mode = mode
        
        # Freeze/unfreeze components based on mode
        if mode == "vae":
            self._set_requires_grad(self.vae_encoder, True)
            self._set_requires_grad(self.vae_decoder, True)
            self._set_requires_grad(self.diffusion_unet, False)
            self._set_requires_grad(self.gan_generator, False)
            self._set_requires_grad(self.gan_discriminator, False)
        elif mode == "diffusion":
            self._set_requires_grad(self.vae_encoder, False)
            self._set_requires_grad(self.vae_decoder, False)
            self._set_requires_grad(self.diffusion_unet, True)
            self._set_requires_grad(self.gan_generator, False)
            self._set_requires_grad(self.gan_discriminator, False)
        elif mode == "gan":
            self._set_requires_grad(self.vae_encoder, False)
            self._set_requires_grad(self.vae_decoder, False)
            self._set_requires_grad(self.diffusion_unet, False)
            self._set_requires_grad(self.gan_generator, True)
            self._set_requires_grad(self.gan_discriminator, True)
        elif mode == "hybrid":
            # All components active
            self._set_requires_grad(self.vae_encoder, True)
            self._set_requires_grad(self.vae_decoder, True)
            self._set_requires_grad(self.diffusion_unet, True)
            self._set_requires_grad(self.gan_generator, True)
            self._set_requires_grad(self.gan_discriminator, True)
            
    def _set_requires_grad(self, module: nn.Module, requires_grad: bool):
        """Set requires_grad for all parameters in a module"""
        for param in module.parameters():
            param.requires_grad = requires_grad
            
    def get_optimizers(self, lr: float = 1e-4) -> Dict[str, torch.optim.Optimizer]:
        """Get optimizers for different components"""
        optimizers = {}
        
        if self.training_mode in ["vae", "hybrid"]:
            vae_params = list(self.vae_encoder.parameters()) + list(self.vae_decoder.parameters())
            optimizers['vae'] = torch.optim.AdamW(vae_params, lr=lr, betas=(0.9, 0.999))
            
        if self.training_mode in ["diffusion", "hybrid"]:
            optimizers['diffusion'] = torch.optim.AdamW(self.diffusion_unet.parameters(), lr=lr, betas=(0.9, 0.999))
            
        if self.training_mode in ["gan", "hybrid"]:
            optimizers['gan_g'] = torch.optim.Adam(self.gan_generator.parameters(), lr=lr, betas=(0.0, 0.99))
            optimizers['gan_d'] = torch.optim.Adam(self.gan_discriminator.parameters(), lr=lr, betas=(0.0, 0.99))
            
        if self.training_mode == "hybrid" and self.enable_cross_modal:
            cross_modal_params = (
                list(self.vae_to_diffusion_attn.parameters()) +
                list(self.gan_to_vae_attn.parameters()) +
                list(self.latent_fusion.parameters()) +
                list(self.text_projection.parameters()) +
                list(self.image_projection.parameters())
            )
            optimizers['cross_modal'] = torch.optim.AdamW(cross_modal_params, lr=lr, betas=(0.9, 0.999))
            
        return optimizers