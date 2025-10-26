"""
TimestepTensor - A specialized torch.Tensor subclass for timesteps
that automatically handles dtype conversion for matrix operations
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Union, Optional, Any


class TimestepTensor(torch.Tensor):
    """
    A tensor subclass for timesteps that enforces proper dtype conversion for operations.
    
    Key Features:
    - Stores timesteps as Long (for indexing operations like embeddings)
    - Automatically converts to Float for matrix operations (Linear layers, matmul, etc.)
    - Prevents "mat1 and mat2 must have the same dtype" errors
    - Maintains timestep semantics and validation
    - Seamless integration with existing PyTorch operations
    """
    
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        """
        Create a new TimestepTensor
        
        Args:
            data: Timestep values (int, float, list, tuple, np.ndarray, or torch.Tensor)
        """
        # Convert input data to tensor if needed
        if isinstance(data, (int, float)):
            data = torch.tensor([data], dtype=torch.long)
        elif isinstance(data, (list, tuple, np.ndarray)) and not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.long)
        elif isinstance(data, torch.Tensor):
            # Ensure it's long type for timestep indexing
            if data.dtype != torch.long:
                data = data.long()
        else:
            raise ValueError(f"Unsupported data type for TimestepTensor: {type(data)}")
        
        # Create the tensor using the parent __new__
        return super().__new__(cls, data, *args, **kwargs)
    
    def __matmul__(self, other):
        """Override matrix multiplication to ensure float casting."""
        # Convert self to float for matrix operations
        self_float = self.float()
        
        if isinstance(other, torch.Tensor):
            if other.is_floating_point():
                return torch.matmul(self_float, other)
            else:
                return torch.matmul(self_float, other.float())
        else:
            return torch.matmul(self_float, other)
    
    def __rmatmul__(self, other):
        """Override reverse matrix multiplication."""
        self_float = self.float()
        
        if isinstance(other, torch.Tensor):
            if other.is_floating_point():
                return torch.matmul(other, self_float)
            else:
                return torch.matmul(other.float(), self_float)
        else:
            return torch.matmul(other, self_float)
    
    def __mul__(self, other):
        """Override multiplication for float conversion when needed."""
        if isinstance(other, torch.Tensor) and other.is_floating_point():
            return torch.mul(self.float(), other)
        elif isinstance(other, (float, np.floating)):
            return torch.mul(self.float(), other)
        else:
            # Keep as long for integer operations
            return super().__mul__(other)
    
    def __rmul__(self, other):
        """Override reverse multiplication."""
        if isinstance(other, torch.Tensor) and other.is_floating_point():
            return torch.mul(other, self.float())
        elif isinstance(other, (float, np.floating)):
            return torch.mul(other, self.float())
        else:
            return super().__rmul__(other)
    
    def __add__(self, other):
        """Override addition with smart dtype handling."""
        if isinstance(other, torch.Tensor) and other.is_floating_point():
            return torch.add(self.float(), other)
        elif isinstance(other, (float, np.floating)):
            return torch.add(self.float(), other)
        else:
            return super().__add__(other)
    
    def __radd__(self, other):
        """Override reverse addition."""
        if isinstance(other, torch.Tensor) and other.is_floating_point():
            return torch.add(other, self.float())
        elif isinstance(other, (float, np.floating)):
            return torch.add(other, self.float())
        else:
            return super().__radd__(other)
    
    def __sub__(self, other):
        """Override subtraction with smart dtype handling."""
        if isinstance(other, torch.Tensor) and other.is_floating_point():
            return torch.sub(self.float(), other)
        elif isinstance(other, (float, np.floating)):
            return torch.sub(self.float(), other)
        else:
            return super().__sub__(other)
    
    def __rsub__(self, other):
        """Override reverse subtraction."""
        if isinstance(other, torch.Tensor) and other.is_floating_point():
            return torch.sub(other, self.float())
        elif isinstance(other, (float, np.floating)):
            return torch.sub(other, self.float())
        else:
            return super().__rsub__(other)
    
    def __truediv__(self, other):
        """Override division - always results in float."""
        return torch.div(self.float(), other)
    
    def __rtruediv__(self, other):
        """Override reverse division."""
        return torch.div(other, self.float())
    
    def __pow__(self, other):
        """Override power operation."""
        if isinstance(other, (float, np.floating)) or (isinstance(other, torch.Tensor) and other.is_floating_point()):
            return torch.pow(self.float(), other)
        else:
            return super().__pow__(other)
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """
        Override torch functions to handle dtype conversion automatically.
        This catches operations like F.linear, torch.mm, etc.
        """
        kwargs = kwargs or {}
        
        # Functions that need float conversion
        float_ops = {
            torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.baddbmm,
            F.linear, F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, 
            F.conv_transpose2d, F.conv_transpose3d, torch.dot, torch.mv,
            torch.ger, torch.addr, torch.addmv
        }
        
        # Functions that need long conversion (indexing operations)
        long_ops = {
            F.embedding, torch.embedding, torch.index_select, torch.gather,
            torch.scatter, torch.scatter_add
        }
        
        # Convert args based on operation type
        if func in float_ops:
            # Convert TimestepTensor instances to float
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, TimestepTensor):
                    args[i] = arg.float()
            args = tuple(args)
        elif func in long_ops:
            # Keep TimestepTensor instances as long
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, TimestepTensor):
                    args[i] = arg.long()
            args = tuple(args)
        
        # Call the original function
        return super().__torch_function__(func, types, args, kwargs)
    
    def float(self):
        """Convert to float32 tensor."""
        return self.to(torch.float32)
    
    def long(self):
        """Convert to long tensor (maintains original behavior)."""
        return self.to(torch.long)
    
    def double(self):
        """Convert to double tensor."""
        return self.to(torch.float64)
    
    def int(self):
        """Convert to int tensor."""
        return self.to(torch.int32)
    
    def expand_to_batch(self, batch_size: int) -> 'TimestepTensor':
        """Expand timesteps to match batch size."""
        if self.shape[0] == batch_size:
            return self
        elif self.shape[0] == 1:
            expanded = self.expand(batch_size)
            return TimestepTensor(expanded)
        else:
            raise ValueError(f"Cannot expand timesteps of size {self.shape[0]} to batch size {batch_size}")
    
    def get_sinusoidal_embeddings(self, 
                                 embedding_dim: int,
                                 flip_sin_to_cos: bool = False,
                                 downscale_freq_shift: int = 1,
                                 scale: float = 1.0,
                                 max_period: int = 10000) -> torch.Tensor:
        """
        Generate sinusoidal timestep embeddings.
        
        Args:
            embedding_dim: Dimension of the embeddings
            flip_sin_to_cos: Whether to flip sin and cos
            downscale_freq_shift: Frequency shift parameter
            scale: Scale factor for embeddings
            max_period: Maximum period for sinusoidal encoding
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        timesteps = self.float()  # Convert to float for embedding computation
        
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        
        half_dim = embedding_dim // 2
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
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
            
        return emb
    
    def validate_range(self, max_timesteps: int = 1000):
        """Validate that timesteps are in valid range."""
        if self.max() >= max_timesteps or self.min() < 0:
            raise ValueError(f"Timesteps must be in range [0, {max_timesteps}). "
                           f"Got range [{self.min()}, {self.max()}]")
        return True
    
    def __repr__(self):
        return f"TimestepTensor({super().__repr__()})"


# Factory functions for common use cases
def create_timestep_tensor(timesteps: Union[int, float, list, tuple, np.ndarray, torch.Tensor],
                          device: Optional[torch.device] = None) -> TimestepTensor:
    """
    Create a TimestepTensor from various input types.
    
    Args:
        timesteps: Timestep values
        device: Target device
        
    Returns:
        TimestepTensor instance
    """
    tensor = TimestepTensor(timesteps)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def random_timesteps(batch_size: int, 
                    max_timesteps: int = 1000,
                    device: Optional[torch.device] = None) -> TimestepTensor:
    """
    Generate random timesteps for training.
    
    Args:
        batch_size: Number of timesteps to generate
        max_timesteps: Maximum timestep value
        device: Target device
        
    Returns:
        TimestepTensor with random timesteps
    """
    timesteps = torch.randint(0, max_timesteps, (batch_size,), dtype=torch.long)
    if device is not None:
        timesteps = timesteps.to(device)
    return TimestepTensor(timesteps)


def fixed_timesteps(timestep_value: int,
                   batch_size: int,
                   device: Optional[torch.device] = None) -> TimestepTensor:
    """
    Generate fixed timesteps (useful for testing).
    
    Args:
        timestep_value: The timestep value to use
        batch_size: Number of timesteps
        device: Target device
        
    Returns:
        TimestepTensor with fixed timesteps
    """
    timesteps = torch.full((batch_size,), timestep_value, dtype=torch.long)
    if device is not None:
        timesteps = timesteps.to(device)
    return TimestepTensor(timesteps)


# Utility function to ensure TimestepTensor
def ensure_timestep_tensor(timesteps) -> TimestepTensor:
    """Ensure input is a TimestepTensor."""
    if isinstance(timesteps, TimestepTensor):
        return timesteps
    else:
        return TimestepTensor(timesteps)