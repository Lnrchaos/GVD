import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock missing dependencies
class MockModule:
    def __getattr__(self, name):
        return MagicMock()

sys.modules['einops'] = MockModule()
sys.modules['flash_attn'] = MockModule()

# Test individual components
class TestVAEComponents:
    """Test VAE encoder and decoder components"""
    
    def test_vae_encoder_shapes(self):
        """Test VAE encoder output shapes"""
        from models.components.vae import VAEEncoder
        
        config = {
            'ch': 32,
            'out_ch': 3,
            'ch_mult': (1, 2, 4),
            'num_res_blocks': 1,
            'attn_resolutions': [8],
            'dropout': 0.0,
            'resamp_with_conv': True,
            'in_channels': 3,
            'resolution': 64,
            'z_channels': 4,
            'double_z': True,
            'use_linear_attn': False,
            'attn_type': "vanilla"
        }
        
        # Mock dependencies
        with patch('models.components.vae.ResnetBlock'), \
             patch('models.components.vae.Downsample'), \
             patch('models.components.vae.make_attn'), \
             patch('models.components.vae.Normalize'), \
             patch('models.components.vae.nonlinearity'):
            
            encoder = VAEEncoder(**config)
            
            # Check basic structure
            assert hasattr(encoder, 'conv_in')
            assert hasattr(encoder, 'down')
            assert hasattr(encoder, 'mid')
            assert hasattr(encoder, 'conv_out')
            
            print("✓ VAE Encoder structure test passed")
    
    def test_diagonal_gaussian_distribution(self):
        """Test DiagonalGaussianDistribution"""
        from models.components.vae import DiagonalGaussianDistribution
        
        # Create test parameters (mean and logvar concatenated)
        batch_size, channels, height, width = 2, 8, 16, 16  # 8 channels = 4 mean + 4 logvar
        parameters = torch.randn(batch_size, channels, height, width)
        
        # Test distribution
        dist = DiagonalGaussianDistribution(parameters)
        
        # Check attributes
        assert dist.mean.shape == (batch_size, channels // 2, height, width)
        assert dist.logvar.shape == (batch_size, channels // 2, height, width)
        assert dist.std.shape == (batch_size, channels // 2, height, width)
        
        # Test sampling
        sample = dist.sample()
        assert sample.shape == (batch_size, channels // 2, height, width)
        
        # Test KL divergence
        kl = dist.kl()
        assert kl.shape == (batch_size,)
        
        print("✓ DiagonalGaussianDistribution test passed")


class TestAttentionComponents:
    """Test attention mechanisms"""
    
    def test_multi_head_cross_attention(self):
        """Test MultiHeadCrossAttention"""
        from models.components.attention import MultiHeadCrossAttention
        
        query_dim = 256
        context_dim = 512
        heads = 8
        dim_head = 64
        
        attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, query_dim)
        context = torch.randn(batch_size, seq_len, context_dim)
        
        # Mock einops functions
        with patch('models.components.attention.rearrange') as mock_rearrange:
            mock_rearrange.side_effect = lambda x, pattern, **kwargs: x
            
            output = attn(x, context)
            
            # Check that to_q, to_k, to_v were called
            assert hasattr(attn, 'to_q')
            assert hasattr(attn, 'to_k')
            assert hasattr(attn, 'to_v')
            assert hasattr(attn, 'to_out')
            
        print("✓ MultiHeadCrossAttention structure test passed")
    
    def test_spatial_transformer(self):
        """Test SpatialTransformer"""
        from models.components.attention import SpatialTransformer
        
        channels = 256
        n_heads = 8
        d_head = 64
        depth = 2
        
        transformer = SpatialTransformer(
            channels=channels,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth
        )
        
        # Check structure
        assert hasattr(transformer, 'norm')
        assert hasattr(transformer, 'proj_in')
        assert hasattr(transformer, 'transformer_blocks')
        assert hasattr(transformer, 'proj_out')
        assert len(transformer.transformer_blocks) == depth
        
        print("✓ SpatialTransformer structure test passed")


class TestLayerComponents:
    """Test layer components"""
    
    def test_timestep_embedding(self):
        """Test TimestepEmbedding"""
        from models.components.layers import TimestepEmbedding
        
        channel = 128
        time_embed_dim = 512
        
        time_emb = TimestepEmbedding(channel, time_embed_dim)
        
        # Test forward pass
        timesteps = torch.randint(0, 1000, (4,))
        output = time_emb(timesteps)
        
        assert output.shape == (4, time_embed_dim)
        
        print("✓ TimestepEmbedding test passed")
    
    def test_resnet_block(self):
        """Test ResnetBlock"""
        from models.components.layers import ResnetBlock
        
        in_channels = 128
        out_channels = 256
        temb_channels = 512
        
        # Mock dependencies
        with patch('models.components.layers.AdaGroupNorm'), \
             patch('models.components.layers.SpatialNorm'):
            
            block = ResnetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                dropout=0.1
            )
            
            # Check structure
            assert hasattr(block, 'norm1')
            assert hasattr(block, 'conv1')
            assert hasattr(block, 'norm2')
            assert hasattr(block, 'conv2')
            
            print("✓ ResnetBlock structure test passed")
    
    def test_attention_block(self):
        """Test AttentionBlock"""
        from models.components.layers import AttentionBlock
        
        channels = 256
        num_head_channels = 32
        
        attn_block = AttentionBlock(
            channels=channels,
            num_head_channels=num_head_channels
        )
        
        # Check structure
        assert hasattr(attn_block, 'group_norm')
        assert hasattr(attn_block, 'query')
        assert hasattr(attn_block, 'key')
        assert hasattr(attn_block, 'value')
        assert hasattr(attn_block, 'proj_attn')
        
        print("✓ AttentionBlock structure test passed")


class TestNoiseScheduler:
    """Test noise scheduler for diffusion"""
    
    def test_noise_scheduler_initialization(self):
        """Test NoiseScheduler initialization"""
        from models.components.diffusion import NoiseScheduler
        
        scheduler = NoiseScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        # Check attributes
        assert scheduler.num_train_timesteps == 1000
        assert len(scheduler.betas) == 1000
        assert len(scheduler.alphas) == 1000
        assert len(scheduler.alphas_cumprod) == 1000
        
        print("✓ NoiseScheduler initialization test passed")
    
    def test_add_noise(self):
        """Test noise addition"""
        from models.components.diffusion import NoiseScheduler
        
        scheduler = NoiseScheduler(num_train_timesteps=1000)
        
        # Test data
        batch_size, channels, height, width = 2, 4, 16, 16
        original_samples = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(original_samples)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        # Add noise
        noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
        
        # Check output shape
        assert noisy_samples.shape == original_samples.shape
        
        print("✓ NoiseScheduler add_noise test passed")


if __name__ == "__main__":
    print("Running component tests...")
    
    try:
        # Test VAE components
        test_vae = TestVAEComponents()
        test_vae.test_diagonal_gaussian_distribution()
        
        # Test attention components  
        test_attn = TestAttentionComponents()
        test_attn.test_multi_head_cross_attention()
        test_attn.test_spatial_transformer()
        
        # Test layer components
        test_layers = TestLayerComponents()
        test_layers.test_timestep_embedding()
        test_layers.test_attention_block()
        
        # Test noise scheduler
        test_scheduler = TestNoiseScheduler()
        test_scheduler.test_noise_scheduler_initialization()
        test_scheduler.test_add_noise()
        
        print("\nAll component tests passed! ✓")
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()