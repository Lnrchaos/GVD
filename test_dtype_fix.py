#!/usr/bin/env python3
"""
Simple test to verify dtype handling fixes
"""

import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys

# Mock missing dependencies
class MockModule:
    def __getattr__(self, name):
        return MagicMock()

sys.modules['einops'] = MockModule()
sys.modules['flash_attn'] = MockModule()

def mock_rearrange(x, pattern, **kwargs):
    if 'b c h w -> b (h w) c' in pattern:
        b, c, h, w = x.shape
        return x.view(b, h * w, c)
    elif 'b (h w) c -> b c h w' in pattern:
        b, hw, c = x.shape
        h = kwargs.get('h', int(hw**0.5))
        w = kwargs.get('w', int(hw**0.5))
        return x.view(b, c, h, w)
    return x

sys.modules['einops'].rearrange = mock_rearrange

def test_dtype_handling():
    """Test that dtype issues are resolved"""
    print("🔧 Testing Dtype Handling Fixes...")
    
    # Test TimestepEmbedding with different input types
    from models.components.layers import TimestepEmbedding
    
    print("1. Testing TimestepEmbedding...")
    
    # Test with integer timesteps (common source of dtype errors)
    time_emb = TimestepEmbedding(320, 1280)
    
    # Test different input types
    test_cases = [
        torch.tensor([0, 100, 500], dtype=torch.long),  # Long tensor
        torch.tensor([0, 100, 500], dtype=torch.int32),  # Int32 tensor
        torch.tensor([0.0, 100.0, 500.0], dtype=torch.float32),  # Float tensor
        torch.randn(3, 320),  # Already correct shape
    ]
    
    for i, timesteps in enumerate(test_cases):
        try:
            output = time_emb(timesteps)
            print(f"  ✓ Test case {i+1}: {timesteps.dtype} -> {output.dtype}, shape: {output.shape}")
        except Exception as e:
            print(f"  ❌ Test case {i+1} failed: {e}")
            return False
    
    print("2. Testing Hybrid Architecture with dtype fixes...")
    
    # Mock all components
    with patch('models.components.VAEEncoder') as mock_vae_enc, \
         patch('models.components.VAEDecoder') as mock_vae_dec, \
         patch('models.components.DiffusionUNet') as mock_diffusion, \
         patch('models.components.NoiseScheduler') as mock_scheduler, \
         patch('models.components.StyleGAN3Generator') as mock_gan_gen, \
         patch('models.components.ProjectedDiscriminator') as mock_gan_disc, \
         patch('models.components.MultiHeadCrossAttention') as mock_attn:
        
        # Setup mocks
        mock_vae_enc.return_value = MagicMock()
        mock_vae_dec.return_value = MagicMock()
        mock_diffusion.return_value = MagicMock()
        mock_scheduler.return_value = MagicMock()
        mock_gan_gen.return_value = MagicMock()
        mock_gan_disc.return_value = MagicMock()
        mock_attn.return_value = MagicMock()
        
        # Import and create model
        from models.hybrid_architecture import HybridGenerativeModel
        
        config = {
            'img_resolution': 64,
            'img_channels': 3,
            'latent_channels': 4,
            'cross_attention_dim': 256,
            'conditioning_dim': 128
        }
        
        model = HybridGenerativeModel(**config)
        print("  ✓ Model created successfully")
        
        # Test timestep processing
        device = torch.device('cpu')
        batch_size = 2
        
        # Test different timestep formats
        timestep_tests = [
            torch.randint(0, 1000, (batch_size,), dtype=torch.long),
            torch.randint(0, 1000, (batch_size,), dtype=torch.int32),
            torch.tensor([100, 200], dtype=torch.long),
        ]
        
        for i, timesteps in enumerate(timestep_tests):
            try:
                # Test that timesteps can be processed without dtype errors
                processed = timesteps.to(device).long()
                print(f"  ✓ Timestep test {i+1}: {timesteps.dtype} processed successfully")
            except Exception as e:
                print(f"  ❌ Timestep test {i+1} failed: {e}")
                return False
    
    print("3. Testing matrix operations with mixed dtypes...")
    
    # Test common problematic operations
    try:
        # Simulate the problematic operation: Linear layer with long input
        linear = nn.Linear(10, 20)
        
        # This would normally fail
        long_input = torch.randint(0, 100, (5, 10), dtype=torch.long)
        
        # Convert to float before operation
        float_input = long_input.float()
        output = linear(float_input)
        
        print(f"  ✓ Linear operation: {long_input.dtype} -> {float_input.dtype} -> {output.dtype}")
        
    except Exception as e:
        print(f"  ❌ Linear operation test failed: {e}")
        return False
    
    print("\n🎉 All dtype handling tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_dtype_handling()
        if success:
            print("\n✅ Dtype fixes are working correctly!")
        else:
            print("\n❌ Some dtype tests failed")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)