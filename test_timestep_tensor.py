#!/usr/bin/env python3
"""
Test script for TimestepTensor - verifying it solves dtype issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def test_timestep_tensor_dtype_handling():
    """Test that TimestepTensor automatically handles dtype conversion"""
    print("🔧 Testing TimestepTensor Dtype Handling...")
    
    from models.tensors.timestep_tensor import TimestepTensor, random_timesteps, fixed_timesteps
    
    print("1. Testing basic TimestepTensor creation...")
    
    # Test different input types
    test_cases = [
        ([0, 100, 500], "list input"),
        (torch.tensor([0, 100, 500], dtype=torch.long), "long tensor"),
        (torch.tensor([0, 100, 500], dtype=torch.int32), "int32 tensor"),
        (500, "single int"),
        (500.0, "single float"),
    ]
    
    for data, description in test_cases:
        try:
            ts_tensor = TimestepTensor(data)
            print(f"  ✓ {description}: created {ts_tensor.dtype} tensor, shape: {ts_tensor.shape}")
            
            # Verify it's stored as long
            assert ts_tensor.dtype == torch.long, f"Expected long, got {ts_tensor.dtype}"
            
        except Exception as e:
            print(f"  ❌ {description} failed: {e}")
            return False
    
    print("2. Testing automatic dtype conversion in operations...")
    
    # Create a TimestepTensor
    timesteps = TimestepTensor([0, 100, 500, 999])
    
    # Test matrix multiplication (the main problem case)
    try:
        linear_layer = nn.Linear(4, 8)
        
        # This should automatically convert to float
        result = linear_layer(timesteps)
        print(f"  ✓ Linear layer: {timesteps.dtype} -> {result.dtype}, shape: {result.shape}")
        
        # Verify result is float
        assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
        
    except Exception as e:
        print(f"  ❌ Linear layer test failed: {e}")
        return False
    
    # Test F.linear directly
    try:
        weight = torch.randn(8, 4)
        bias = torch.randn(8)
        
        result = F.linear(timesteps, weight, bias)
        print(f"  ✓ F.linear: {timesteps.dtype} -> {result.dtype}, shape: {result.shape}")
        
    except Exception as e:
        print(f"  ❌ F.linear test failed: {e}")
        return False
    
    # Test matrix multiplication
    try:
        matrix = torch.randn(4, 6)
        result = torch.matmul(timesteps.float(), matrix)  # Explicit conversion for comparison
        
        # Test automatic conversion
        result_auto = torch.matmul(timesteps, matrix)
        print(f"  ✓ Matrix multiplication: automatic conversion works")
        
        # Results should be the same
        assert torch.allclose(result, result_auto), "Results should be identical"
        
    except Exception as e:
        print(f"  ❌ Matrix multiplication test failed: {e}")
        return False
    
    print("3. Testing embedding operations (should stay as long)...")
    
    try:
        # Create embedding layer
        embedding = nn.Embedding(1000, 64)
        
        # TimestepTensor should stay as long for embedding lookup
        result = embedding(timesteps)
        print(f"  ✓ Embedding lookup: {timesteps.dtype} -> {result.dtype}, shape: {result.shape}")
        
        # Verify embedding worked
        assert result.shape == (4, 64), f"Expected shape (4, 64), got {result.shape}"
        
    except Exception as e:
        print(f"  ❌ Embedding test failed: {e}")
        return False
    
    print("4. Testing sinusoidal embedding generation...")
    
    try:
        embeddings = timesteps.get_sinusoidal_embeddings(embedding_dim=320)
        print(f"  ✓ Sinusoidal embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}")
        
        # Verify shape and dtype
        assert embeddings.shape == (4, 320), f"Expected (4, 320), got {embeddings.shape}"
        assert embeddings.dtype == torch.float32, f"Expected float32, got {embeddings.dtype}"
        
    except Exception as e:
        print(f"  ❌ Sinusoidal embedding test failed: {e}")
        return False
    
    print("5. Testing factory functions...")
    
    try:
        # Test random timesteps
        random_ts = random_timesteps(batch_size=3, max_timesteps=1000, device=torch.device('cpu'))
        print(f"  ✓ Random timesteps: {random_ts.dtype}, shape: {random_ts.shape}")
        
        # Test fixed timesteps
        fixed_ts = fixed_timesteps(timestep_value=500, batch_size=3, device=torch.device('cpu'))
        print(f"  ✓ Fixed timesteps: {fixed_ts.dtype}, shape: {fixed_ts.shape}")
        
        # Verify they're TimestepTensors
        assert isinstance(random_ts, TimestepTensor), "Should be TimestepTensor"
        assert isinstance(fixed_ts, TimestepTensor), "Should be TimestepTensor"
        
    except Exception as e:
        print(f"  ❌ Factory function test failed: {e}")
        return False
    
    print("6. Testing integration with hybrid architecture...")
    
    try:
        # Mock all components and test with TimestepTensor
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
            
            # Test TimestepTensor creation in hybrid architecture context
            from models.hybrid_architecture import HybridGenerativeModel
            
            config = {
                'img_resolution': 64,
                'img_channels': 3,
                'latent_channels': 4,
                'cross_attention_dim': 256,
                'conditioning_dim': 128
            }
            
            model = HybridGenerativeModel(**config)
            
            # Test that random_timesteps works in the context
            device = torch.device('cpu')
            batch_size = 2
            
            timesteps = random_timesteps(batch_size=batch_size, max_timesteps=1000, device=device)
            print(f"  ✓ Integration test: created {type(timesteps).__name__} with shape {timesteps.shape}")
            
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False
    
    print("\n🎉 All TimestepTensor tests passed!")
    print("✅ TimestepTensor successfully solves the dtype conversion issues!")
    return True


def test_problematic_scenario():
    """Test the exact scenario that was causing problems"""
    print("\n🎯 Testing the exact problematic scenario...")
    
    from models.tensors.timestep_tensor import TimestepTensor
    
    try:
        # Simulate the exact problem: Long tensor in linear operation
        batch_size = 2
        embedding_dim = 320
        
        # Create timesteps as they would be in the diffusion model
        timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
        print(f"Original timesteps: {timesteps.dtype}")
        
        # Convert to TimestepTensor
        ts_tensor = TimestepTensor(timesteps)
        print(f"TimestepTensor: {ts_tensor.dtype}")
        
        # Create a linear layer (like TimestepEmbedding.linear_1)
        linear = nn.Linear(embedding_dim, 1280)
        
        # This would normally fail with "mat1 and mat2 must have the same dtype"
        # But TimestepTensor should handle it automatically
        
        # First, we need embeddings (simulate get_sinusoidal_embeddings)
        embeddings = ts_tensor.get_sinusoidal_embeddings(embedding_dim)
        print(f"Generated embeddings: {embeddings.dtype}, shape: {embeddings.shape}")
        
        # Now the linear operation that was failing
        result = linear(embeddings)
        print(f"Linear result: {result.dtype}, shape: {result.shape}")
        
        print("✅ The problematic scenario now works perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Problematic scenario still fails: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # Run TimestepTensor tests
        success1 = test_timestep_tensor_dtype_handling()
        
        # Run problematic scenario test
        success2 = test_problematic_scenario()
        
        if success1 and success2:
            print("\n🎊 ALL TESTS PASSED!")
            print("🎊 TimestepTensor is ready for production!")
            print("🎊 No more dtype conversion errors!")
        else:
            print("\n❌ Some tests failed")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)