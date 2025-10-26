"""
ConditionalTimestepTensor (CTT) - A custom PyTorch tensor subclass
for handling timesteps with automatic dtype conversion and context validation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Tuple
import warnings
import math


class ConditionalTimestepTensor(torch.Tensor):
    """
    A specialized tensor subclass that handles timesteps with automatic dtype conversion
    and mandatory context validation for hybrid generative models.
    
    Key Features:
    - Automatic dtype casting (Long -> Float32) for matrix operations
    - Mandatory context validation (prevents temb=None issues)
    - Built-in timestep embedding generation
    - Deterministic behavior for testing
    - Context preservation across operations
    """
    
    @staticmethod
    def __new__(cls, 
                timesteps: Union[torch.Tensor, int, float, list],
                context: Optional[Dict[str, Any]] = None,
                embedding_dim: int = 320,
                max_timesteps: int = 1000,
                device: Optional[torch.device] = None,
                requires_grad: bool = False,
                validate_context: bool = True):
        """
        Create a new ConditionalTimestepTensor
        
        Args:
            timesteps: The timestep values (int, float, tensor, or list)
            context: Required context dictionary with keys like 'batch_size', 'latent_shape', etc.
            embedding_dim: Dimension for timestep embeddings
            max_timesteps: Maximum number of timesteps (for validation)
            device: Target device
            requires_grad: Whether to track gradients
            validate_context: Whether to enforce context validation
        """
        
        # Convert timesteps to tensor
        if not torch.is_tensor(timesteps):
            if isinstance(timesteps, (int, float)):
                timesteps = torch.tensor([timesteps])
            elif isinstance(timesteps, (list, tuple)):
                timesteps = torch.tensor(timesteps)
            else:
                raise ValueError(f"Unsupported timesteps type: {type(timesteps)}")
        
        # Ensure proper device
        if device is not None:
            timesteps = timesteps.to(device)
        
        # Always store as long for timestep indexing, but enable auto-conversion
        if timesteps.dtype not in [torch.long, torch.int32, torch.int64]:
            timesteps = timesteps.long()
        
        # Create the tensor
        obj = torch.Tensor._make_subclass(cls, timesteps, requires_grad)
        
        # Store metadata
        obj._embedding_dim = embedding_dim
        obj._max_timesteps = max_timesteps
        obj._context = context or {}
        obj._validate_context = validate_context
        obj._original_shape = timesteps.shape
        
        # Validate context if required
        if validate_context:
            obj._validate_required_context()
        
        return obj
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Override torch functions to handle dtype conversion automatically"""
        kwargs = kwargs or {}
        
        # Handle matrix multiplication operations
        if func in [torch.matmul, torch.mm, torch.bmm, F.linear]:
            # Convert self to float for matrix operations
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, ConditionalTimestepTensor):
                    args[i] = arg.to_float()
            args = tuple(args)
        
        # Handle embedding operations
        elif func in [F.embedding, torch.embedding]:
            # Ensure we're long for embedding lookup
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, ConditionalTimestepTensor):
                    args[i] = arg.to_long()
            args = tuple(args)
        
        # Call the original function
        result = super().__torch_function__(func, types, args, kwargs)
        
        # Preserve CTT properties if result is a tensor
        if isinstance(result, torch.Tensor) and not isinstance(result, ConditionalTimestepTensor):
            # For operations that should preserve CTT nature
            if func in [torch.unsqueeze, torch.squeeze, torch.reshape, torch.view]:
                return ConditionalTimestepTensor._from_tensor(result, self)
        
        return result
    
    @classmethod
    def _from_tensor(cls, tensor: torch.Tensor, source: 'ConditionalTimestepTensor'):
        """Create CTT from existing tensor, preserving source metadata"""
        obj = torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)
        obj._embedding_dim = source._embedding_dim
        obj._max_timesteps = source._max_timesteps
        obj._context = source._context.copy()
        obj._validate_context = source._validate_context
        obj._original_shape = source._original_shape
        return obj
    
    def _validate_required_context(self):
        """Validate that required context is present"""
        required_keys = ['batch_size']
        
        for key in required_keys:
            if key not in self._context:
                raise ValueError(f"ConditionalTimestepTensor requires '{key}' in context. "
                               f"Got context keys: {list(self._context.keys())}")
        
        # Validate timestep values
        if self.max() >= self._max_timesteps or self.min() < 0:
            raise ValueError(f"Timesteps must be in range [0, {self._max_timesteps}). "
                           f"Got range [{self.min()}, {self.max()}]")
    
    def to_float(self) -> torch.Tensor:
        """Convert to float32 for matrix operations"""
        return self.float()
    
    def to_long(self) -> torch.Tensor:
        """Convert to long for indexing operations"""
        return self.long()
    
    def get_embeddings(self, 
                      flip_sin_to_cos: bool = False,
                      downscale_freq_shift: int = 1,
                      scale: float = 1.0,
                      max_period: int = 10000) -> torch.Tensor:
        """
        Generate sinusoidal timestep embeddings
        
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        timesteps = self.to_float()
        
        half_dim = self._embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=self.device
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
            
        # Zero pad if needed
        if self._embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
            
        return emb
    
    def expand_to_batch(self, batch_size: int) -> 'ConditionalTimestepTensor':
        """Expand timesteps to match batch size"""
        if self.shape[0] == batch_size:
            return self
        elif self.shape[0] == 1:
            expanded = self.expand(batch_size)
            new_context = self._context.copy()
            new_context['batch_size'] = batch_size
            return ConditionalTimestepTensor._from_tensor(expanded, self)
        else:
            raise ValueError(f"Cannot expand timesteps of size {self.shape[0]} to batch size {batch_size}")
    
    def add_context(self, **kwargs) -> 'ConditionalTimestepTensor':
        """Add additional context information"""
        new_context = self._context.copy()
        new_context.update(kwargs)
        
        result = ConditionalTimestepTensor._from_tensor(self, self)
        result._context = new_context
        return result
    
    def get_context(self, key: str, default=None):
        """Get context value"""
        return self._context.get(key, default)
    
    def __repr__(self):
        return (f"ConditionalTimestepTensor({super().__repr__()}, "
                f"embedding_dim={self._embedding_dim}, "
                f"context={self._context})")


class CTTFactory:
    """Factory class for creating ConditionalTimestepTensors with common configurations"""
    
    @staticmethod
    def for_diffusion(timesteps: Union[torch.Tensor, int, float, list],
                     batch_size: int,
                     latent_shape: Tuple[int, ...],
                     device: torch.device,
                     embedding_dim: int = 320) -> ConditionalTimestepTensor:
        """Create CTT for diffusion models"""
        context = {
            'batch_size': batch_size,
            'latent_shape': latent_shape,
            'model_type': 'diffusion',
            'device': device
        }
        
        return ConditionalTimestepTensor(
            timesteps=timesteps,
            context=context,
            embedding_dim=embedding_dim,
            device=device
        )
    
    @staticmethod
    def for_vae(timesteps: Union[torch.Tensor, int, float, list],
                batch_size: int,
                image_shape: Tuple[int, ...],
                device: torch.device,
                embedding_dim: int = 512) -> ConditionalTimestepTensor:
        """Create CTT for VAE models"""
        context = {
            'batch_size': batch_size,
            'image_shape': image_shape,
            'model_type': 'vae',
            'device': device
        }
        
        return ConditionalTimestepTensor(
            timesteps=timesteps,
            context=context,
            embedding_dim=embedding_dim,
            device=device
        )
    
    @staticmethod
    def for_testing(batch_size: int = 2,
                   timestep_value: int = 500,
                   device: torch.device = torch.device('cpu'),
                   embedding_dim: int = 320) -> ConditionalTimestepTensor:
        """Create deterministic CTT for testing"""
        timesteps = torch.full((batch_size,), timestep_value, dtype=torch.long)
        
        context = {
            'batch_size': batch_size,
            'model_type': 'test',
            'device': device,
            'deterministic': True
        }
        
        return ConditionalTimestepTensor(
            timesteps=timesteps,
            context=context,
            embedding_dim=embedding_dim,
            device=device,
            validate_context=False  # Relaxed validation for testing
        )


class CTTLinear(torch.nn.Linear):
    """Linear layer that automatically handles CTT inputs"""
    
    def forward(self, input):
        if isinstance(input, ConditionalTimestepTensor):
            # For timestep embeddings, generate embeddings first
            if input.shape[-1] != self.in_features:
                input = input.get_embeddings()
            else:
                input = input.to_float()
        
        return super().forward(input)


class CTTEmbedding(torch.nn.Embedding):
    """Embedding layer that automatically handles CTT inputs"""
    
    def forward(self, input):
        if isinstance(input, ConditionalTimestepTensor):
            input = input.to_long()
        
        return super().forward(input)


# Utility functions for working with CTT
def ensure_ctt(timesteps, **kwargs) -> ConditionalTimestepTensor:
    """Ensure input is a ConditionalTimestepTensor"""
    if isinstance(timesteps, ConditionalTimestepTensor):
        return timesteps
    else:
        return ConditionalTimestepTensor(timesteps, **kwargs)


def ctt_safe_linear(input_tensor, weight, bias=None):
    """Safe linear operation that handles CTT automatically"""
    if isinstance(input_tensor, ConditionalTimestepTensor):
        input_tensor = input_tensor.to_float()
    
    return F.linear(input_tensor, weight, bias)


def ctt_safe_embedding(input_tensor, weight, padding_idx=None, max_norm=None, 
                      norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Safe embedding operation that handles CTT automatically"""
    if isinstance(input_tensor, ConditionalTimestepTensor):
        input_tensor = input_tensor.to_long()
    
    return F.embedding(input_tensor, weight, padding_idx, max_norm, 
                      norm_type, scale_grad_by_freq, sparse)


# Context manager for CTT operations
class CTTContext:
    """Context manager for safe CTT operations"""
    
    def __init__(self, auto_convert: bool = True):
        self.auto_convert = auto_convert
        self.original_linear = None
        self.original_embedding = None
    
    def __enter__(self):
        if self.auto_convert:
            # Monkey patch common operations
            self.original_linear = F.linear
            self.original_embedding = F.embedding
            
            F.linear = ctt_safe_linear
            F.embedding = ctt_safe_embedding
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_convert and self.original_linear:
            F.linear = self.original_linear
            F.embedding = self.original_embedding