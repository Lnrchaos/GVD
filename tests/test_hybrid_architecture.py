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

# Mock einops and other dependencies that might not be available
sys.modules['einops'] = MockModule()
sys.modules['flash_attn'] = MockModule()

# Import after mocking
from models.hybrid_architecture import HybridGenerativeModel


class TestHybridArchitecture:
    """Test suite for the hybrid generative architecture"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def small_config(self):
        """Small configuration for fast testing"""
        return {
            'img_resolution': 64,
            'img_channels': 3,
            'latent_channels': 4,
            'cross_attention_dim': 256,
            'conditioning_dim': 128,
            'vae_config': {
                'ch': 32,
                'ch_mult': (1, 2, 4),
                'num_res_blocks': 1,
                'attn_resolutions': [8]
            },
            'diffusion_config': {
                'block_out_channels': (64, 128, 256),
                'layers_per_block': 1
            },
            'gan_config': {
                'z_dim': 128,
                'w_dim': 128
            }
        }
    
    @pytest.fixture
    def model(self, small_config, device):
        """Create a small model for testing"""
        with patch('models.components.VAEEncoder'), \
             patch('models.components.VAEDecoder'), \
             patch('models.components.DiffusionUNet'), \
             patch('models.components.NoiseScheduler'), \
             patch('models.components.StyleGAN3Generator'), \
             patch('models.components.ProjectedDiscriminator'), \
             patch('models.components.MultiHeadCrossAttention'):
            
            model = HybridGenerativeModel(**small_config)
            return model.to(device)
    
    def test_model_initialization(self, small_config):
        """Test that the model initializes correctly"""
        with patch('models.components.VAEEncoder'), \
             patch('models.components.VAEDecoder'), \
             patch('models.components.DiffusionUNet'), \
             patch('models.components.NoiseScheduler'), \
             patch('models.components.StyleGAN3Generator'), \
             patch('models.components.ProjectedDiscriminator'), \
             patch('models.components.MultiHeadCrossAttention'):
            
            model = HybridGenerativeModel(**small_config)
            
            # Check basic attributes
            assert model.img_resolution == 64
            assert model.img_channels == 3
            assert model.latent_channels == 4
            assert model.training_mode == "hybrid"
            
            # Check components exist
            assert hasattr(model, 'vae_encoder')
            assert hasattr(model, 'vae_decoder')
            assert hasattr(model, 'diffusion_unet')
            assert hasattr(model, 'gan_generator')
            assert hasattr(model, 'gan_discriminator')
    
    def test_training_mode_switching(self, model):
        """Test switching between training modes"""
        # Test VAE mode
        model.set_training_mode("vae")
        assert model.training_mode == "vae"
        
        # Test diffusion mode
        model.set_training_mode("diffusion")
        assert model.training_mode == "diffusion"
        
        # Test GAN mode
        model.set_training_mode("gan")
        assert model.training_mode == "gan"
        
        # Test hybrid mode
        model.set_training_mode("hybrid")
        assert model.training_mode == "hybrid"
    
    def test_loss_weights(self, model):
        """Test loss weight configuration"""
        expected_weights = {
            'vae_recon': 1.0,
            'vae_kl': 0.1,
            'diffusion': 1.0,
            'gan_g': 1.0,
            'gan_d': 1.0,
            'cross_modal': 0.5
        }
        
        for key, expected_value in expected_weights.items():
            assert model.loss_weights[key] == expected_value
    
    def test_optimizer_creation(self, model):
        """Test optimizer creation for different modes"""
        # Test VAE optimizers
        model.set_training_mode("vae")
        optimizers = model.get_optimizers(lr=1e-4)
        assert 'vae' in optimizers
        assert isinstance(optimizers['vae'], torch.optim.AdamW)
        
        # Test diffusion optimizers
        model.set_training_mode("diffusion")
        optimizers = model.get_optimizers(lr=1e-4)
        assert 'diffusion' in optimizers
        assert isinstance(optimizers['diffusion'], torch.optim.AdamW)
        
        # Test GAN optimizers
        model.set_training_mode("gan")
        optimizers = model.get_optimizers(lr=1e-4)
        assert 'gan_g' in optimizers
        assert 'gan_d' in optimizers
        assert isinstance(optimizers['gan_g'], torch.optim.Adam)
        assert isinstance(optimizers['gan_d'], torch.optim.Adam)
        
        # Test hybrid optimizers
        model.set_training_mode("hybrid")
        optimizers = model.get_optimizers(lr=1e-4)
        expected_keys = ['vae', 'diffusion', 'gan_g', 'gan_d', 'cross_modal']
        for key in expected_keys:
            assert key in optimizers
    
    @patch('models.hybrid_architecture.rearrange')
    def test_forward_pass_shapes(self, mock_rearrange, model, device):
        """Test forward pass with different modes"""
        batch_size = 2
        img_size = 64
        
        # Mock input
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # Mock component outputs
        mock_latents = torch.randn(batch_size, 4, img_size // 8, img_size // 8, device=device)
        mock_posterior = MagicMock()
        mock_posterior.kl.return_value = torch.tensor([0.1, 0.1], device=device)
        
        # Mock VAE components
        model.vae_encoder.return_value = torch.randn(batch_size, 8, img_size // 8, img_size // 8, device=device)
        model.vae_decoder.return_value = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # Mock diffusion components
        model.diffusion_unet.return_value = {"sample": mock_latents}
        model.noise_scheduler.add_noise.return_value = mock_latents
        
        # Mock GAN components
        model.gan_generator.return_value = torch.randn(batch_size, 3, img_size, img_size, device=device)
        model.gan_discriminator.return_value = torch.randn(batch_size, 1, device=device)
        
        # Mock DiagonalGaussianDistribution
        with patch('models.hybrid_architecture.DiagonalGaussianDistribution') as mock_dist:
            mock_dist.return_value = mock_posterior
            mock_posterior.sample.return_value = mock_latents
            
            # Test VAE mode
            model.set_training_mode("vae")
            model.train()
            outputs = model(x, mode="vae")
            
            assert 'vae_latents' in outputs
            assert 'vae_posterior' in outputs
            assert 'vae_reconstruction' in outputs
            
            # Test diffusion mode
            model.set_training_mode("diffusion")
            outputs = model(x, mode="diffusion")
            
            assert 'diffusion_noise_pred' in outputs
            assert 'diffusion_noise_target' in outputs
            
            # Test GAN mode
            model.set_training_mode("gan")
            outputs = model(x, mode="gan")
            
            assert 'gan_z' in outputs
            assert 'gan_images' in outputs
            assert 'gan_real_logits' in outputs
            assert 'gan_fake_logits' in outputs
    
    def test_loss_computation(self, model, device):
        """Test loss computation for different components"""
        batch_size = 2
        img_size = 64
        
        # Mock targets
        targets = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # Mock outputs
        outputs = {
            'vae_reconstruction': torch.randn(batch_size, 3, img_size, img_size, device=device),
            'vae_posterior': MagicMock(),
            'diffusion_noise_pred': torch.randn(batch_size, 4, img_size // 8, img_size // 8, device=device),
            'diffusion_noise_target': torch.randn(batch_size, 4, img_size // 8, img_size // 8, device=device),
            'gan_real_logits': torch.randn(batch_size, 1, device=device),
            'gan_fake_logits': torch.randn(batch_size, 1, device=device),
            'gan_images': torch.randn(batch_size, 3, img_size, img_size, device=device),
            'fused_images': torch.randn(batch_size, 3, img_size, img_size, device=device)
        }
        
        # Mock KL divergence
        outputs['vae_posterior'].kl.return_value = torch.tensor([0.1, 0.1], device=device)
        
        # Mock discriminator for generator loss
        model.gan_discriminator.return_value = torch.randn(batch_size, 1, device=device)
        
        # Compute losses
        losses = model.compute_losses(outputs, targets)
        
        # Check that all expected losses are computed
        expected_losses = ['vae_recon', 'vae_kl', 'diffusion', 'gan_d', 'gan_g', 'cross_modal', 'total']
        for loss_name in expected_losses:
            assert loss_name in losses
            assert isinstance(losses[loss_name], torch.Tensor)
            assert losses[loss_name].requires_grad
    
    def test_cross_modal_disabled(self, small_config, device):
        """Test model with cross-modal fusion disabled"""
        small_config['enable_cross_modal'] = False
        
        with patch('models.components.VAEEncoder'), \
             patch('models.components.VAEDecoder'), \
             patch('models.components.DiffusionUNet'), \
             patch('models.components.NoiseScheduler'), \
             patch('models.components.StyleGAN3Generator'), \
             patch('models.components.ProjectedDiscriminator'):
            
            model = HybridGenerativeModel(**small_config)
            
            # Check cross-modal components don't exist
            assert not hasattr(model, 'vae_to_diffusion_attn')
            assert not hasattr(model, 'gan_to_vae_attn')
            assert not hasattr(model, 'latent_fusion')
    
    def test_inference_mode(self, model, device):
        """Test model in inference mode"""
        batch_size = 1
        
        # Set to eval mode
        model.eval()
        
        # Mock components for inference
        mock_latents = torch.randn(batch_size, 4, 8, 8, device=device)
        model.generate_diffusion = MagicMock(return_value=mock_latents)
        model.decode_vae = MagicMock(return_value=torch.randn(batch_size, 3, 64, 64, device=device))
        model.gan_generator = MagicMock(return_value=torch.randn(batch_size, 3, 64, 64, device=device))
        
        # Test diffusion inference
        with torch.no_grad():
            outputs = model(mode="diffusion", batch_size=batch_size)
            assert 'diffusion_latents' in outputs
            assert 'diffusion_images' in outputs
            
        # Test GAN inference
        with torch.no_grad():
            outputs = model(mode="gan", batch_size=batch_size)
            assert 'gan_z' in outputs
            assert 'gan_images' in outputs


if __name__ == "__main__":
    # Run basic tests
    print("Running basic hybrid architecture tests...")
    
    # Test model creation
    try:
        config = {
            'img_resolution': 64,
            'img_channels': 3,
            'latent_channels': 4,
            'cross_attention_dim': 256,
            'conditioning_dim': 128
        }
        
        with patch('models.components.VAEEncoder'), \
             patch('models.components.VAEDecoder'), \
             patch('models.components.DiffusionUNet'), \
             patch('models.components.NoiseScheduler'), \
             patch('models.components.StyleGAN3Generator'), \
             patch('models.components.ProjectedDiscriminator'), \
             patch('models.components.MultiHeadCrossAttention'):
            
            model = HybridGenerativeModel(**config)
            print("✓ Model initialization successful")
            
            # Test training mode switching
            model.set_training_mode("vae")
            assert model.training_mode == "vae"
            print("✓ Training mode switching works")
            
            # Test optimizer creation
            optimizers = model.get_optimizers()
            assert 'vae' in optimizers
            print("✓ Optimizer creation works")
            
            print("\nAll basic tests passed! ✓")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()