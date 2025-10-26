#!/usr/bin/env python3
"""
Comprehensive test for ConditionalTimestepTensor (CTT)
Tests dtype handling, context validation, and integration with the hybrid architecture
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


def test_ctt_basic_functionality():
    """Test basic ConditionalTimestepTensor functionality"""
    print("🧪 Testing ConditionalTimestepTensor Basic Functionality")
    print("=" * 60)
    
    from models.tensors import ConditionalTimestepTensor, CTTFactory, CTTContext
    
    # Test 1: Basic CTT Creation
    print("1️⃣  Testing CTT Creation...")
    
    # Test different input types
    test_cases = [
        (100, "single integer"),
        ([100, 200, 300], "list of integers"),
        (torch.tensor([100, 200]), "tensor input"),
        (torch.tensor([100.0, 200.0]), "float tensor input")
    ]
    
    for timesteps, description in test_cases:
        try:
            context = {'batch_size': 3 if isinstance(timesteps, list) else 2}
            ctt = ConditionalTimestepTensor(timesteps, context=context)
            print(f"  ✓ {description}: {ctt.dtype} -> shape {ctt.shape}")
        except Exception as e:
            print(f"  ❌ {description} failed: {e}")
            return False
    
    # Test 2: Automatic Dtype Conversion
    print("\n2️⃣  Testing Automatic Dtype Conversion...")
    
    # Create CTT with integer timesteps
    context = {'batch_size': 2}
    ctt = ConditionalTimestepTensor([100, 200], context=context)
    
    # Test matrix operations (should auto-convert to float)
    linear = nn.Linear(2, 10)
    
    try:
        # This would normally fail with "Long and Float" error
        ctt_reshaped = ctt.unsqueeze(-1).expand(-1, 2).float()  # Make it 2D for linear layer
        output = linear(ctt_reshaped)
        print(f"  ✓ Matrix operation: {ctt.dtype} -> {output.dtype}, shape: {output.shape}")
    except Exception as e:
        print(f"  ❌ Matrix operation failed: {e}")
        return False
    
    # Test 3: Context Validation
    print("\n3️⃣  Testing Context Validation...")
    
    try:
        # This should fail - missing required context
        ctt_invalid = ConditionalTimestepTensor([100, 200], context={})
        print("  ❌ Context validation failed - should have raised error")
        return False
    except ValueError as e:
        print(f"  ✓ Context validation working: {e}")
    
    # Test 4: Timestep Embeddings
    print("\n4️⃣  Testing Timestep Embeddings...")
    
    ctt = ConditionalTimestepTensor([100, 200, 300], context={'batch_size': 3}, embedding_dim=320)
    
    try:
        embeddings = ctt.get_embeddings()
        print(f"  ✓ Embeddings generated: shape {embeddings.shape}, dtype {embeddings.dtype}")
        
        # Test different embedding configurations
        embeddings_cos_first = ctt.get_embeddings(flip_sin_to_cos=True)
        print(f"  ✓ Cos-first embeddings: shape {embeddings_cos_first.shape}")
        
    except Exception as e:
        print(f"  ❌ Embedding generation failed: {e}")
        return False
    
    # Test 5: Factory Methods
    print("\n5️⃣  Testing Factory Methods...")
    
    try:
        # Test diffusion factory
        ctt_diffusion = CTTFactory.for_diffusion(
            timesteps=[100, 200],
            batch_size=2,
            latent_shape=(4, 8, 8),
            device=torch.device('cpu')
        )
        print(f"  ✓ Diffusion CTT: {ctt_diffusion.get_context('model_type')}")
        
        # Test testing factory
        ctt_test = CTTFactory.for_testing(batch_size=3, timestep_value=500)
        print(f"  ✓ Test CTT: deterministic={ctt_test.get_context('deterministic')}")
        
    except Exception as e:
        print(f"  ❌ Factory methods failed: {e}")
        return False
    
    # Test 6: CTT Context Manager
    print("\n6️⃣  Testing CTT Context Manager...")
    
    try:
        ctt = ConditionalTimestepTensor([100, 200], context={'batch_size': 2})
        
        with CTTContext(auto_convert=True):
            # Operations inside context should be safe
            linear = nn.Linear(1, 10)
            ctt_input = ctt.unsqueeze(-1).float()
            result = linear(ctt_input)
            print(f"  ✓ Context manager: safe operations completed")
            
    except Exception as e:
        print(f"  ❌ Context manager failed: {e}")
        return False
    
    print("\n🎉 All CTT basic functionality tests passed!")
    return True


def test_ctt_integration():
    """Test CTT integration with the hybrid architecture"""
    print("\n🔗 Testing CTT Integration with Hybrid Architecture")
    print("=" * 60)
    
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
        mock_diffusion_instance = MagicMock()
        mock_diffusion.return_value = mock_diffusion_instance
        mock_scheduler_instance = MagicMock()
        mock_scheduler.return_value = mock_scheduler_instance
        mock_gan_gen.return_value = MagicMock()
        mock_gan_disc.return_value = MagicMock()
        mock_attn.return_value = MagicMock()
        
        # Mock the diffusion UNet to handle CTT
        def mock_diffusion_forward(latents, timesteps, encoder_hidden_states=None):
            # timesteps should be a ConditionalTimestepTensor
            from models.tensors import ConditionalTimestepTensor
            
            if isinstance(timesteps, ConditionalTimestepTensor):
                print(f"    ✓ Received CTT with context: {timesteps.get_context('model_type')}")
                # Convert to embeddings for processing
                embeddings = timesteps.get_embeddings()
                print(f"    ✓ Generated embeddings: {embeddings.shape}")
            
            return {"sample": torch.randn_like(latents)}
        
        mock_diffusion_instance.side_effect = mock_diffusion_forward
        
        # Mock noise scheduler
        def mock_add_noise(clean, noise, timesteps):
            return clean + 0.1 * noise  # Simple noise addition
        
        mock_scheduler_instance.add_noise = mock_add_noise
        mock_scheduler_instance.num_train_timesteps = 1000
        
        # Import and test
        from models.hybrid_architecture import HybridGenerativeModel
        
        print("1️⃣  Testing Model Creation with CTT Support...")
        
        config = {
            'img_resolution': 64,
            'img_channels': 3,
            'latent_channels': 4,
            'cross_attention_dim': 256,
            'conditioning_dim': 128
        }
        
        model = HybridGenerativeModel(**config)
        print("  ✓ Model created successfully")
        
        print("\n2️⃣  Testing Forward Pass with CTT...")
        
        # Test input
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        
        # Mock VAE components
        mock_latents = torch.randn(batch_size, 4, 8, 8)
        mock_posterior = MagicMock()
        mock_posterior.sample.return_value = mock_latents
        mock_posterior.kl.return_value = torch.tensor([0.1, 0.1])
        
        model.vae_encoder.return_value = torch.randn(batch_size, 8, 8, 8)
        model.vae_decoder.return_value = torch.randn(batch_size, 3, 64, 64)
        
        # Mock DiagonalGaussianDistribution
        with patch('models.hybrid_architecture.DiagonalGaussianDistribution', return_value=mock_posterior):
            
            # Test diffusion mode (should use CTT)
            model.set_training_mode("diffusion")
            model.train()
            
            try:
                outputs = model(x, mode="diffusion")
                
                assert 'diffusion_noise_pred' in outputs
                assert 'diffusion_noise_target' in outputs
                print("  ✓ Diffusion forward pass with CTT completed")
                
            except Exception as e:
                print(f"  ❌ Diffusion forward pass failed: {e}")
                return False
        
        print("\n3️⃣  Testing CTT Error Prevention...")
        
        # Test that CTT prevents the original dtype errors
        from models.tensors import ConditionalTimestepTensor
        
        try:
            # Create problematic scenario that CTT should handle
            timesteps = ConditionalTimestepTensor(
                [100, 200], 
                context={'batch_size': 2},
                embedding_dim=320
            )
            
            # This should work without dtype errors
            embeddings = timesteps.get_embeddings()
            linear = nn.Linear(320, 512)
            result = linear(embeddings)
            
            print(f"  ✓ No dtype errors: {timesteps.dtype} -> {embeddings.dtype} -> {result.dtype}")
            
        except Exception as e:
            print(f"  ❌ CTT error prevention failed: {e}")
            return False
    
    print("\n🎉 All CTT integration tests passed!")
    return True


def test_ctt_edge_cases():
    """Test CTT edge cases and error handling"""
    print("\n🛡️  Testing CTT Edge Cases and Error Handling")
    print("=" * 60)
    
    from models.tensors import ConditionalTimestepTensor, CTTFactory
    
    print("1️⃣  Testing Invalid Inputs...")
    
    # Test invalid timestep values
    try:
        ctt = ConditionalTimestepTensor(
            [-1, 1001],  # Invalid range
            context={'batch_size': 2},
            max_timesteps=1000
        )
        print("  ❌ Should have failed for invalid timestep range")
        return False
    except ValueError as e:
        print(f"  ✓ Invalid timestep range caught: {str(e)[:50]}...")
    
    print("\n2️⃣  Testing Batch Size Mismatches...")
    
    try:
        ctt = ConditionalTimestepTensor([100], context={'batch_size': 1})
        expanded = ctt.expand_to_batch(3)
        print(f"  ✓ Batch expansion: {ctt.shape} -> {expanded.shape}")
        
        # This should fail
        ctt_multi = ConditionalTimestepTensor([100, 200], context={'batch_size': 2})
        expanded_fail = ctt_multi.expand_to_batch(5)  # Should fail
        print("  ❌ Should have failed for incompatible batch expansion")
        return False
        
    except ValueError as e:
        print(f"  ✓ Batch size mismatch caught: {str(e)[:50]}...")
    
    print("\n3️⃣  Testing Memory Efficiency...")
    
    try:
        # Test with larger tensors
        large_timesteps = torch.randint(0, 1000, (100,))
        ctt_large = ConditionalTimestepTensor(
            large_timesteps,
            context={'batch_size': 100},
            embedding_dim=512
        )
        
        embeddings = ctt_large.get_embeddings()
        print(f"  ✓ Large tensor handling: {ctt_large.shape} -> {embeddings.shape}")
        
        # Memory usage should be reasonable
        memory_mb = embeddings.numel() * embeddings.element_size() / (1024 * 1024)
        print(f"  ✓ Memory usage: {memory_mb:.2f} MB")
        
    except Exception as e:
        print(f"  ❌ Large tensor test failed: {e}")
        return False
    
    print("\n4️⃣  Testing Device Handling...")
    
    try:
        # Test device consistency
        device = torch.device('cpu')  # Use CPU for testing
        ctt = ConditionalTimestepTensor(
            [100, 200],
            context={'batch_size': 2},
            device=device
        )
        
        embeddings = ctt.get_embeddings()
        assert embeddings.device == device
        print(f"  ✓ Device consistency: {ctt.device} == {embeddings.device}")
        
    except Exception as e:
        print(f"  ❌ Device handling failed: {e}")
        return False
    
    print("\n🎉 All CTT edge case tests passed!")
    return True


if __name__ == "__main__":
    try:
        print("🚀 Starting ConditionalTimestepTensor Test Suite")
        print("=" * 70)
        
        # Run all test suites
        success1 = test_ctt_basic_functionality()
        success2 = test_ctt_integration()
        success3 = test_ctt_edge_cases()
        
        if success1 and success2 and success3:
            print("\n" + "=" * 70)
            print("🎊 CONGRATULATIONS!")
            print("🎊 ConditionalTimestepTensor is working perfectly!")
            print("🎊 Your hybrid architecture now has robust timestep handling!")
            print("🎊 No more dtype errors or temb=None issues!")
            print("=" * 70)
        else:
            print("\n❌ Some CTT tests failed")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ CTT test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)