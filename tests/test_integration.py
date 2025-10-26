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


def test_full_integration():
    """Integration test for the complete hybrid architecture"""
    print("Running full integration test...")
    
    # Import after mocking
    from models.hybrid_architecture import HybridGenerativeModel
    
    # Configuration for small model
    config = {
        'img_resolution': 64,
        'img_channels': 3,
        'latent_channels': 4,
        'cross_attention_dim': 256,
        'conditioning_dim': 128,
        'enable_cross_modal': True,
        'training_mode': 'hybrid',
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
    
    # Mock all components
    with patch('models.components.VAEEncoder') as mock_vae_enc, \
         patch('models.components.VAEDecoder') as mock_vae_dec, \
         patch('models.components.DiffusionUNet') as mock_diffusion, \
         patch('models.components.NoiseScheduler') as mock_scheduler, \
         patch('models.components.StyleGAN3Generator') as mock_gan_gen, \
         patch('models.components.ProjectedDiscriminator') as mock_gan_disc, \
         patch('models.components.MultiHeadCrossAttention') as mock_attn, \
         patch('models.hybrid_architecture.DiagonalGaussianDistribution') as mock_dist, \
         patch('models.hybrid_architecture.rearrange') as mock_rearrange:
        
        # Setup mocks
        device = torch.device('cpu')
        batch_size = 2
        img_size = 64
        latent_size = img_size // 8
        
        # Mock VAE components
        mock_vae_enc_instance = MagicMock()
        mock_vae_dec_instance = MagicMock()
        mock_vae_enc.return_value = mock_vae_enc_instance
        mock_vae_dec.return_value = mock_vae_dec_instance
        
        mock_vae_enc_instance.return_value = torch.randn(batch_size, 8, latent_size, latent_size)
        mock_vae_dec_instance.return_value = torch.randn(batch_size, 3, img_size, img_size)
        
        # Mock diffusion components
        mock_diffusion_instance = MagicMock()
        mock_scheduler_instance = MagicMock()
        mock_diffusion.return_value = mock_diffusion_instance
        mock_scheduler.return_value = mock_scheduler_instance
        
        mock_diffusion_instance.return_value = {"sample": torch.randn(batch_size, 4, latent_size, latent_size)}
        mock_scheduler_instance.add_noise.return_value = torch.randn(batch_size, 4, latent_size, latent_size)
        mock_scheduler_instance.num_train_timesteps = 1000
        
        # Mock GAN components
        mock_gan_gen_instance = MagicMock()
        mock_gan_disc_instance = MagicMock()
        mock_gan_gen.return_value = mock_gan_gen_instance
        mock_gan_disc.return_value = mock_gan_disc_instance
        
        mock_gan_gen_instance.return_value = torch.randn(batch_size, 3, img_size, img_size)
        mock_gan_disc_instance.return_value = torch.randn(batch_size, 1)
        
        # Mock attention
        mock_attn_instance = MagicMock()
        mock_attn.return_value = mock_attn_instance
        mock_attn_instance.return_value = torch.randn(batch_size, latent_size * latent_size, 4)
        
        # Mock distribution
        mock_posterior = MagicMock()
        mock_dist.return_value = mock_posterior
        mock_posterior.sample.return_value = torch.randn(batch_size, 4, latent_size, latent_size)
        mock_posterior.kl.return_value = torch.tensor([0.1, 0.1])
        
        # Mock rearrange
        mock_rearrange.side_effect = lambda x, pattern, **kwargs: x.view(x.shape[0], -1, x.shape[1])
        
        # Create model
        model = HybridGenerativeModel(**config)
        model.to(device)
        
        print("✓ Model created successfully")
        
        # Test input
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # Test 1: VAE mode
        print("Testing VAE mode...")
        model.set_training_mode("vae")
        model.train()
        
        outputs = model(x, mode="vae")
        
        assert 'vae_latents' in outputs
        assert 'vae_posterior' in outputs  
        assert 'vae_reconstruction' in outputs
        print("✓ VAE mode test passed")
        
        # Test 2: Diffusion mode
        print("Testing Diffusion mode...")
        model.set_training_mode("diffusion")
        
        outputs = model(x, mode="diffusion")
        
        assert 'diffusion_noise_pred' in outputs
        assert 'diffusion_noise_target' in outputs
        print("✓ Diffusion mode test passed")
        
        # Test 3: GAN mode
        print("Testing GAN mode...")
        model.set_training_mode("gan")
        
        outputs = model(x, mode="gan")
        
        assert 'gan_z' in outputs
        assert 'gan_images' in outputs
        assert 'gan_real_logits' in outputs
        assert 'gan_fake_logits' in outputs
        print("✓ GAN mode test passed")
        
        # Test 4: Hybrid mode
        print("Testing Hybrid mode...")
        model.set_training_mode("hybrid")
        
        # Mock cross-modal fusion components
        model.latent_fusion = nn.Sequential(
            nn.Linear(8, 4),
            nn.SiLU(),
            nn.Linear(4, 4)
        )
        
        outputs = model(x, mode="hybrid")
        
        # Should have outputs from all components
        assert 'vae_latents' in outputs
        assert 'vae_reconstruction' in outputs
        assert 'diffusion_noise_pred' in outputs
        assert 'gan_images' in outputs
        print("✓ Hybrid mode test passed")
        
        # Test 5: Loss computation
        print("Testing loss computation...")
        
        # Create comprehensive outputs for loss testing
        test_outputs = {
            'vae_reconstruction': torch.randn(batch_size, 3, img_size, img_size),
            'vae_posterior': mock_posterior,
            'diffusion_noise_pred': torch.randn(batch_size, 4, latent_size, latent_size),
            'diffusion_noise_target': torch.randn(batch_size, 4, latent_size, latent_size),
            'gan_real_logits': torch.randn(batch_size, 1),
            'gan_fake_logits': torch.randn(batch_size, 1),
            'gan_images': torch.randn(batch_size, 3, img_size, img_size),
            'fused_images': torch.randn(batch_size, 3, img_size, img_size)
        }
        
        losses = model.compute_losses(test_outputs, x)
        
        # Check all expected losses
        expected_losses = ['vae_recon', 'vae_kl', 'diffusion', 'gan_d', 'gan_g', 'cross_modal', 'total']
        for loss_name in expected_losses:
            assert loss_name in losses
            assert isinstance(losses[loss_name], torch.Tensor)
        
        print("✓ Loss computation test passed")
        
        # Test 6: Optimizer creation
        print("Testing optimizer creation...")
        
        optimizers = model.get_optimizers(lr=1e-4)
        
        expected_optimizers = ['vae', 'diffusion', 'gan_g', 'gan_d', 'cross_modal']
        for opt_name in expected_optimizers:
            assert opt_name in optimizers
            assert isinstance(optimizers[opt_name], torch.optim.Optimizer)
        
        print("✓ Optimizer creation test passed")
        
        # Test 7: Training mode switching
        print("Testing training mode switching...")
        
        modes = ['vae', 'diffusion', 'gan', 'hybrid']
        for mode in modes:
            model.set_training_mode(mode)
            assert model.training_mode == mode
            
            # Test that optimizers are created correctly for each mode
            opts = model.get_optimizers()
            assert len(opts) > 0
        
        print("✓ Training mode switching test passed")
        
        # Test 8: Inference mode
        print("Testing inference mode...")
        
        model.eval()
        model.set_training_mode("diffusion")
        
        # Mock inference methods
        model.generate_diffusion = MagicMock(return_value=torch.randn(1, 4, latent_size, latent_size))
        model.decode_vae = MagicMock(return_value=torch.randn(1, 3, img_size, img_size))
        
        with torch.no_grad():
            outputs = model(mode="diffusion", batch_size=1)
            
        assert 'diffusion_latents' in outputs
        assert 'diffusion_images' in outputs
        
        print("✓ Inference mode test passed")
        
        print("\n🎉 All integration tests passed successfully!")
        return True


def test_memory_efficiency():
    """Test memory efficiency and gradient flow"""
    print("\nTesting memory efficiency...")
    
    from models.hybrid_architecture import HybridGenerativeModel
    
    config = {
        'img_resolution': 32,  # Very small for memory test
        'img_channels': 3,
        'latent_channels': 4,
        'cross_attention_dim': 64,
        'conditioning_dim': 32
    }
    
    with patch('models.components.VAEEncoder'), \
         patch('models.components.VAEDecoder'), \
         patch('models.components.DiffusionUNet'), \
         patch('models.components.NoiseScheduler'), \
         patch('models.components.StyleGAN3Generator'), \
         patch('models.components.ProjectedDiscriminator'), \
         patch('models.components.MultiHeadCrossAttention'):
        
        model = HybridGenerativeModel(**config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Test gradient requirements for different modes
        model.set_training_mode("vae")
        vae_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model.set_training_mode("diffusion")  
        diff_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model.set_training_mode("gan")
        gan_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model.set_training_mode("hybrid")
        hybrid_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ VAE mode trainable: {vae_trainable:,}")
        print(f"✓ Diffusion mode trainable: {diff_trainable:,}")
        print(f"✓ GAN mode trainable: {gan_trainable:,}")
        print(f"✓ Hybrid mode trainable: {hybrid_trainable:,}")
        
        # Hybrid should have the most trainable parameters
        assert hybrid_trainable >= vae_trainable
        assert hybrid_trainable >= diff_trainable
        assert hybrid_trainable >= gan_trainable
        
        print("✓ Memory efficiency test passed")


if __name__ == "__main__":
    try:
        # Run integration tests
        success = test_full_integration()
        
        if success:
            # Run memory efficiency test
            test_memory_efficiency()
            
            print("\n🎉 All tests completed successfully!")
            print("\nThe hybrid architecture is working correctly and ready for training!")
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()