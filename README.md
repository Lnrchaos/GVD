# GVD - Hybrid Generative Architecture

A modular hybrid generative model combining **VAE**, **GAN**, and **Diffusion** models with cross-modal attention and fusion capabilities.

## 🚀 Features

- **Hybrid Architecture**: Seamlessly combines three generative paradigms
- **Cross-Modal Fusion**: Attention mechanisms between different model outputs  
- **Flexible Training**: Train individual components or all together
- **ConditionalTimestepTensor**: Custom tensor class for robust timestep handling
- **Modular Design**: Easy to extend and customize

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridGenerativeModel                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│       VAE       │    Diffusion    │          GAN            │
│   ┌─────────┐   │   ┌─────────┐   │   ┌─────────────────┐   │
│   │ Encoder │   │   │  U-Net  │   │   │   StyleGAN3     │   │
│   │ Decoder │   │   │Scheduler│   │   │   Generator     │   │
│   └─────────┘   │   └─────────┘   │   │  Discriminator  │   │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
                  ┌────────────────┐
                  │ Cross-Modal    │
                  │ Attention &    │
                  │ Fusion Layers  │
                  └────────────────┘
```

## 📦 Installation

```bash
git clone <repository-url>
cd GVD
pip install torch torchvision
pip install einops  # Optional but recommended
```

## 🎯 Quick Start

### Basic Usage

```python
from models.hybrid_architecture import HybridGenerativeModel

# Create model
model = HybridGenerativeModel(
    img_resolution=512,
    img_channels=3,
    latent_channels=4,
    training_mode="hybrid"
)

# Generate with different modes
x = torch.randn(1, 3, 512, 512)

# VAE reconstruction
outputs = model(x, mode="vae")
reconstruction = outputs['vae_reconstruction']

# Diffusion generation
outputs = model(mode="diffusion", batch_size=1)
diffusion_images = outputs['diffusion_images']

# GAN generation  
outputs = model(mode="gan", batch_size=1)
gan_images = outputs['gan_images']

# Hybrid mode (all components)
outputs = model(x, mode="hybrid")
fused_images = outputs['fused_images']
```

### Training Setup

```python
# Set training mode
model.set_training_mode("hybrid")

# Get optimizers
optimizers = model.get_optimizers(lr=1e-4)

# Training loop
for batch in dataloader:
    # Forward pass
    outputs = model(batch['images'], mode="hybrid")
    
    # Compute losses
    losses = model.compute_losses(outputs, batch['images'])
    
    # Backward pass
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    
    losses['total'].backward()
    
    for optimizer in optimizers.values():
        optimizer.step()
```

## 🔧 Configuration

### Model Configuration

```python
config = {
    # Image parameters
    'img_resolution': 512,
    'img_channels': 3,
    'latent_channels': 4,
    
    # VAE configuration
    'vae_config': {
        'ch': 128,
        'ch_mult': (1, 2, 4, 8),
        'num_res_blocks': 2,
        'attn_resolutions': [16]
    },
    
    # Diffusion configuration
    'diffusion_config': {
        'block_out_channels': (320, 640, 1280, 1280),
        'layers_per_block': 2,
        'cross_attention_dim': 1024
    },
    
    # GAN configuration
    'gan_config': {
        'z_dim': 512,
        'w_dim': 512
    },
    
    # Hybrid parameters
    'cross_attention_dim': 1024,
    'conditioning_dim': 512,
    'enable_cross_modal': True,
    'training_mode': 'hybrid'
}

model = HybridGenerativeModel(**config)
```

### Training Modes

- **`"vae"`**: Train only VAE components
- **`"diffusion"`**: Train only diffusion U-Net
- **`"gan"`**: Train only GAN generator/discriminator
- **`"hybrid"`**: Train all components with cross-modal fusion

## 🧪 ConditionalTimestepTensor (CTT)

Advanced tensor class that solves common issues:

```python
from models.tensors import CTTFactory

# Create CTT for diffusion
timesteps = CTTFactory.for_diffusion(
    timesteps=[0, 100, 500],
    batch_size=3,
    latent_shape=(4, 64, 64),
    device=device
)

# Automatic dtype conversion
embeddings = timesteps.get_embeddings()  # Always float32
indices = timesteps.to_long()           # For embedding lookup

# Context validation
batch_size = timesteps.get_context('batch_size')
```

## 📁 Project Structure

```
GVD/
├── models/
│   ├── hybrid_architecture.py      # Main hybrid model
│   ├── components/
│   │   ├── vae.py                 # VAE encoder/decoder
│   │   ├── diffusion.py           # Diffusion U-Net
│   │   ├── gan.py                 # StyleGAN3 components
│   │   ├── attention.py           # Cross-attention layers
│   │   └── layers.py              # Common layers
│   ├── tensors/
│   │   └── conditional_timestep_tensor.py  # CTT implementation
│   └── utils/
│       └── tensor_utils.py        # Utility functions
├── tests/
│   ├── test_hybrid_architecture.py
│   ├── test_components.py
│   └── test_integration.py
└── README.md
```

## 🔬 Testing

Run the test suite:

```bash
# Quick architecture test
python test_architecture.py

# Dtype handling test
python test_dtype_fix.py

# Full test suite (if pytest available)
python -m pytest tests/
```

## 🎛️ Advanced Features

### Cross-Modal Fusion

The hybrid mode enables cross-modal attention between different generative approaches:

```python
# Enable cross-modal fusion
model = HybridGenerativeModel(enable_cross_modal=True)

# Forward pass generates fused outputs
outputs = model(x, mode="hybrid")
fused_latents = outputs['fused_latents']
fused_images = outputs['fused_images']
```

### Loss Weighting

Customize loss weights for different components:

```python
model.loss_weights = {
    'vae_recon': 1.0,
    'vae_kl': 0.1,
    'diffusion': 1.0,
    'gan_g': 1.0,
    'gan_d': 1.0,
    'cross_modal': 0.5
}
```

### Conditional Generation

Support for text and image conditioning:

```python
# Text conditioning
text_embeddings = torch.randn(1, 77, 768)  # CLIP embeddings
outputs = model(mode="diffusion", conditioning=text_embeddings)

# Image conditioning  
image_features = torch.randn(1, 256)
outputs = model(mode="gan", conditioning=image_features)
```

## 🚨 Common Issues & Solutions

### Dtype Mismatch Errors

**Problem**: `mat1 and mat2 must have the same dtype, but got Long and Float`

**Solution**: Use ConditionalTimestepTensor or ensure proper dtype conversion:

```python
# Bad
timesteps = torch.randint(0, 1000, (batch_size,))  # Long tensor
embedding = linear_layer(timesteps)  # Error!

# Good  
timesteps = CTTFactory.for_diffusion(...)  # Auto-converts
embedding = linear_layer(timesteps)  # Works!

# Or manual conversion
timesteps = timesteps.float()
```

### Missing Context Errors

**Problem**: `temb is None` in ResNet blocks

**Solution**: Use CTT with proper context validation:

```python
# CTT ensures context is always available
timesteps = CTTFactory.for_diffusion(
    timesteps=raw_timesteps,
    batch_size=batch_size,
    latent_shape=latent_shape,
    device=device
)
```

## 📊 Performance Tips

1. **Memory Optimization**: Use smaller resolutions during development
2. **Training Efficiency**: Start with individual components before hybrid training
3. **Gradient Accumulation**: Use for large batch sizes
4. **Mixed Precision**: Enable for faster training (if supported)

```python
# Memory-efficient config for development
dev_config = {
    'img_resolution': 64,  # Smaller resolution
    'vae_config': {'ch': 32},  # Fewer channels
    'diffusion_config': {'block_out_channels': (64, 128, 256)}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 MIT License


## 🙏 Acknowledgments

- StyleGAN3 architecture
- Stable Diffusion U-Net design
- VAE implementation patterns
- PyTorch tensor subclassing examples

## Tips can be sent to paypal account
- lylerichards17@gmail.com 
---

**Note**: This is a research/experimental architecture. Use appropriate hardware (GPU recommended) and expect high memory usage for large resolutions.
