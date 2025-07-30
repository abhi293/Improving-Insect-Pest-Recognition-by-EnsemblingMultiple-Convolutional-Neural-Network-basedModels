"""
Performance optimization configuration for the CNN models.
This file contains optimized settings for different hardware configurations.
"""

import torch
import psutil
import os

class PerformanceConfig:
    """Configuration class for performance optimization"""
    
    def __init__(self, device):
        self.device = device
        self.device_type = device.type
        
    def get_optimal_settings(self):
        """Get optimal settings based on device type and available resources"""
        
        if self.device_type == "cuda":
            return self._get_cuda_settings()
        elif self.device_type == "dml":
            return self._get_dml_settings()
        elif self.device_type == "cpu":
            return self._get_cpu_settings()
        elif self.device_type == "mps":
            return self._get_mps_settings()
        else:
            return self._get_default_settings()
    
    def _get_cuda_settings(self):
        """Optimized settings for CUDA devices"""
        try:
            gpu_props = torch.cuda.get_device_properties(self.device)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            
            if gpu_memory_gb >= 8:  # High-end GPU
                return {
                    'batch_size_multiplier': 1.0,
                    'num_workers': 8,
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 4,
                    'mixed_precision': True,
                    'compile_model': True,
                    'memory_fraction': 0.9
                }
            elif gpu_memory_gb >= 4:  # Mid-range GPU
                return {
                    'batch_size_multiplier': 0.7,
                    'num_workers': 4,
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 2,
                    'mixed_precision': True,
                    'compile_model': False,
                    'memory_fraction': 0.8
                }
            else:  # Low-end GPU
                return {
                    'batch_size_multiplier': 0.5,
                    'num_workers': 2,
                    'pin_memory': True,
                    'persistent_workers': False,
                    'prefetch_factor': 2,
                    'mixed_precision': True,
                    'compile_model': False,
                    'memory_fraction': 0.7
                }
        except:
            return self._get_default_settings()
    
    def _get_dml_settings(self):
        """Conservative settings for DirectML"""
        return {
            'batch_size_multiplier': 0.3,
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'mixed_precision': True,
            'compile_model': False,
            'memory_fraction': 0.6
        }
    
    def _get_cpu_settings(self):
        """Optimized settings for CPU"""
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return {
            'batch_size_multiplier': 0.2 if memory_gb < 8 else 0.4,
            'num_workers': min(cpu_count, 4),
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'mixed_precision': False,
            'compile_model': False,
            'memory_fraction': 0.5
        }
    
    def _get_mps_settings(self):
        """Settings for Apple Metal Performance Shaders"""
        return {
            'batch_size_multiplier': 0.6,
            'num_workers': 4,
            'pin_memory': False,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'mixed_precision': True,
            'compile_model': False,
            'memory_fraction': 0.7
        }
    
    def _get_default_settings(self):
        """Conservative default settings"""
        return {
            'batch_size_multiplier': 0.5,
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'mixed_precision': False,
            'compile_model': False,
            'memory_fraction': 0.6
        }

def optimize_torch_settings(device):
    """Apply PyTorch optimizations based on device"""
    
    if device.type == "cuda":
        # Enable optimizations for CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory management
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
    elif device.type == "cpu":
        # CPU optimizations
        torch.set_num_threads(min(psutil.cpu_count(logical=False), 8))
        
    # Set float32 matmul precision for better performance
    torch.set_float32_matmul_precision('medium')

def get_optimal_image_size(device, model_complexity='medium'):
    """Get optimal image size based on device and model complexity"""
    
    size_configs = {
        'simple': {'cuda': 224, 'dml': 128, 'cpu': 128, 'mps': 224},
        'medium': {'cuda': 224, 'dml': 196, 'cpu': 128, 'mps': 224},
        'complex': {'cuda': 256, 'dml': 224, 'cpu': 196, 'mps': 256}
    }
    
    device_type = device.type
    if device_type not in size_configs[model_complexity]:
        device_type = 'cpu'  # fallback
        
    return size_configs[model_complexity][device_type]
