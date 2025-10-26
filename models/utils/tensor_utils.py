import torch
import torch.nn as nn
from typing import Union, Optional


class DTypeHandler:
    """Utility class to handle dtype mismatches in tensor operations"""
    
    @staticmethod
    def ensure_float(tensor: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Ensure tensor is float type for matrix operations"""
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            return tensor.float().to(target_dtype)
        elif tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor
    
    @staticmethod
    def match_dtypes(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match dtypes of two tensors, preferring float types"""
        if tensor1.dtype == tensor2.dtype:
            return tensor1, tensor2
            
        # Prefer float types over integer types
        if tensor1.dtype.is_floating_point and not tensor2.dtype.is_floating_point:
            return tensor1, tensor2.to(tensor1.dtype)
        elif tensor2.dtype.is_floating_point and not tensor1.dtype.is_floating_point:
            return tensor1.to(tensor2.dtype), tensor2
        else:
            # Both are same category, convert to float32
            target_dtype = torch.float32
            return tensor1.to(target_dtype), tensor2.to(target_dtype)


class SafeLinear(nn.Linear):
    """Linear layer that handles dtype mismatches automatically"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Ensure input is float type
        input = DTypeHandler.ensure_float(input, self.weight.dtype)
        return super().forward(input)


class SafeEmbedding(nn.Embedding):
    """Embedding layer that handles dtype mismatches"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Ensure input is long type for embedding lookup
        if input.dtype != torch.long:
            input = input.long()
        return super().forward(input)


class TimestepProcessor:
    """Handles timestep tensor processing with proper dtype management"""
    
    @staticmethod
    def process_timesteps(timesteps: Union[torch.Tensor, int, float], 
                         batch_size: int, 
                         device: torch.device,
                         embedding_dim: int = 320) -> torch.Tensor:
        """Process timesteps ensuring proper dtype and shape"""
        
        # Convert to tensor if needed
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif timesteps.dtype not in [torch.long, torch.int32, torch.int64]:
            timesteps = timesteps.long()
            
        # Ensure proper device
        timesteps = timesteps.to(device)
        
        # Handle scalar timesteps
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
            
        # Broadcast to batch size
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape[0] != batch_size:
            # If mismatch, create new tensor with repeated values
            timesteps = timesteps[0].unsqueeze(0).expand(batch_size)
            
        return timesteps
    
    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, 
                              embedding_dim: int,
                              flip_sin_to_cos: bool = False,
                              downscale_freq_shift: int = 1,
                              scale: float = 1.0,
                              max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings with proper dtype handling
        """
        # Ensure timesteps are float for embedding computation
        timesteps = DTypeHandler.ensure_float(timesteps)
        
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        
        half_dim = embedding_dim // 2
        exponent = -torch.log(torch.tensor(max_period, dtype=torch.float32)) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
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
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            
        return emb


class SafeMatMul:
    """Safe matrix multiplication with dtype handling"""
    
    @staticmethod
    def matmul(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with automatic dtype matching"""
        tensor1, tensor2 = DTypeHandler.match_dtypes(tensor1, tensor2)
        return torch.matmul(tensor1, tensor2)
    
    @staticmethod
    def bmm(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Perform batch matrix multiplication with automatic dtype matching"""
        tensor1, tensor2 = DTypeHandler.match_dtypes(tensor1, tensor2)
        return torch.bmm(tensor1, tensor2)


class TensorCompatibilityWrapper:
    """Wrapper to make tensors compatible for operations"""
    
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.original_dtype = tensor.dtype
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def to_float(self) -> torch.Tensor:
        """Convert to float for operations"""
        return DTypeHandler.ensure_float(self.tensor)
        
    def restore_dtype(self, result: torch.Tensor) -> torch.Tensor:
        """Restore original dtype if appropriate"""
        if self.original_dtype in [torch.int32, torch.int64, torch.long]:
            # Don't convert back to int if result should be float
            return result
        return result.to(self.original_dtype)


# Monkey patch common operations to handle dtype issues
def safe_linear_forward(self, input):
    """Safe linear forward that handles dtype mismatches"""
    input = DTypeHandler.ensure_float(input, self.weight.dtype)
    return torch.nn.functional.linear(input, self.weight, self.bias)


def patch_linear_layers():
    """Patch all Linear layers to use safe forward"""
    original_forward = nn.Linear.forward
    nn.Linear.forward = safe_linear_forward
    return original_forward


def unpatch_linear_layers(original_forward):
    """Restore original Linear forward"""
    nn.Linear.forward = original_forward


# Context manager for safe operations
class SafeTensorOps:
    """Context manager for safe tensor operations"""
    
    def __init__(self):
        self.original_linear_forward = None
        
    def __enter__(self):
        self.original_linear_forward = patch_linear_layers()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_linear_forward:
            unpatch_linear_layers(self.original_linear_forward)


# Example usage functions
def safe_timestep_embedding(timesteps, embedding_dim, model_device):
    """Create timestep embeddings safely"""
    processor = TimestepProcessor()
    
    # Process timesteps
    if isinstance(timesteps, (int, float)):
        batch_size = 1
    else:
        batch_size = timesteps.shape[0] if torch.is_tensor(timesteps) else len(timesteps)
        
    processed_timesteps = processor.process_timesteps(
        timesteps, batch_size, model_device, embedding_dim
    )
    
    # Create embeddings
    embeddings = processor.get_timestep_embedding(processed_timesteps, embedding_dim)
    
    return embeddings


def safe_attention_computation(q, k, v):
    """Compute attention safely with dtype handling"""
    # Ensure all tensors have same dtype
    q = DTypeHandler.ensure_float(q)
    k = DTypeHandler.ensure_float(k)
    v = DTypeHandler.ensure_float(v)
    
    # Compute attention scores
    scores = SafeMatMul.matmul(q, k.transpose(-2, -1))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply to values
    output = SafeMatMul.matmul(attn_weights, v)
    
    return output