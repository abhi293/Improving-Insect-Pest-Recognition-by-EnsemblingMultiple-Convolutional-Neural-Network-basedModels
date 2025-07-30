"""Training Parameter Analyzer and Optimizer"""

import os
import argparse
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import numpy as np

def analyze_dataset(dataset_path, sample_ratio=0.05):
    """Analyze dataset characteristics and provide training recommendations"""
    
    # Quick transforms for analysis
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=basic_transform)
    
    # Get class distribution
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    num_classes = len(class_counts)
    total_samples = len(dataset)
    min_class_size = min(class_counts.values())
    max_class_size = max(class_counts.values())
    avg_class_size = total_samples / num_classes
    
    # Calculate sample characteristics
    sample_size = int(total_samples * sample_ratio)
    samples_per_class = sample_size / num_classes
    
    print(f"📊 Dataset Analysis: {os.path.basename(dataset_path)}")
    print(f"   Total samples: {total_samples}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Min class size: {min_class_size}")
    print(f"   Max class size: {max_class_size}")
    print(f"   Avg class size: {avg_class_size:.1f}")
    print(f"   Class balance ratio: {min_class_size/max_class_size:.3f}")
    
    print(f"\n📈 Sample Analysis (ratio: {sample_ratio*100:.1f}%)")
    print(f"   Sample size: {sample_size}")
    print(f"   Avg samples per class: {samples_per_class:.1f}")
    
    return {
        'total_samples': total_samples,
        'num_classes': num_classes,
        'min_class_size': min_class_size,
        'max_class_size': max_class_size,
        'avg_class_size': avg_class_size,
        'sample_size': sample_size,
        'samples_per_class': samples_per_class,
        'balance_ratio': min_class_size/max_class_size
    }

def recommend_training_params(dataset_stats, device_type="auto"):
    """Provide training parameter recommendations based on dataset characteristics"""
    
    samples_per_class = dataset_stats['samples_per_class']
    num_classes = dataset_stats['num_classes']
    total_sample_size = dataset_stats['sample_size']
    balance_ratio = dataset_stats['balance_ratio']
    
    print(f"\n🎯 Training Recommendations:")
    
    # Learning rate recommendations
    if samples_per_class < 5:
        lr_range = "0.0001 - 0.0003"
        lr_rec = 0.0001
        complexity = "Very Low"
    elif samples_per_class < 15:
        lr_range = "0.0003 - 0.0007"
        lr_rec = 0.0005
        complexity = "Low"
    elif samples_per_class < 30:
        lr_range = "0.0005 - 0.001"
        lr_rec = 0.0007
        complexity = "Medium"
    else:
        lr_range = "0.0007 - 0.002"
        lr_rec = 0.001
        complexity = "High"
    
    print(f"   Sample complexity: {complexity}")
    print(f"   Recommended learning rate: {lr_rec}")
    print(f"   Learning rate range: {lr_range}")
    
    # Epoch recommendations
    if samples_per_class < 5:
        epoch_range = "15-25"
        epoch_rec = 20
    elif samples_per_class < 15:
        epoch_range = "10-15"
        epoch_rec = 12
    elif samples_per_class < 30:
        epoch_range = "8-12"
        epoch_rec = 10
    else:
        epoch_range = "5-10"
        epoch_rec = 7
    
    print(f"   Recommended epochs: {epoch_rec}")
    print(f"   Epoch range: {epoch_range}")
    
    # Batch size recommendations
    if total_sample_size < 500:
        batch_rec = 4
        batch_range = "2-8"
    elif total_sample_size < 2000:
        batch_rec = 8
        batch_range = "4-16"
    elif total_sample_size < 5000:
        batch_rec = 16
        batch_range = "8-32"
    else:
        batch_rec = 32
        batch_range = "16-64"
    
    print(f"   Recommended batch size: {batch_rec}")
    print(f"   Batch size range: {batch_range}")
    
    # Sample ratio recommendations
    if samples_per_class < 3:
        sample_ratios = "0.1, 0.15, 0.2"
        print(f"   ⚠️  Very few samples per class! Consider larger ratios: {sample_ratios}")
    elif samples_per_class < 10:
        sample_ratios = "0.08, 0.12, 0.15"
        print(f"   Consider testing larger ratios: {sample_ratios}")
    
    # Balance recommendations
    if balance_ratio < 0.5:
        print(f"   ⚠️  Dataset imbalanced (ratio: {balance_ratio:.3f})")
        print(f"   Consider weighted sampling or balanced sample creation")
    
    # Device-specific adjustments
    device_adjustments = {
        'cpu': {'lr_mult': 0.3, 'batch_mult': 0.5, 'note': 'Conservative settings for CPU'},
        'privateuseone': {'lr_mult': 0.5, 'batch_mult': 0.7, 'note': 'DirectML optimizations'},
        'cuda': {'lr_mult': 1.0, 'batch_mult': 1.0, 'note': 'Full performance settings'},
        'mps': {'lr_mult': 0.8, 'batch_mult': 0.9, 'note': 'Metal Performance Shaders'}
    }
    
    if device_type != "auto" and device_type in device_adjustments:
        adj = device_adjustments[device_type]
        adj_lr = lr_rec * adj['lr_mult']
        adj_batch = max(1, int(batch_rec * adj['batch_mult']))
        print(f"\n   Device adjustments for {device_type}:")
        print(f"   Adjusted learning rate: {adj_lr}")
        print(f"   Adjusted batch size: {adj_batch}")
        print(f"   Note: {adj['note']}")
        
        return adj_lr, epoch_rec, adj_batch
    
    return lr_rec, epoch_rec, batch_rec

def generate_command_line(dataset, lr, epochs, batch_size, sample_ratio=0.05, optimizer='Adam', scheduler='steplr'):
    """Generate optimized command line"""
    
    cmd = f"python OptimizedSampleTrain.py "
    cmd += f"-data {dataset} "
    cmd += f"-optim {optimizer} "
    cmd += f"-sch {scheduler} "
    cmd += f"-l2 0.01 "
    cmd += f"-do 0.2 "
    cmd += f"-predt True "
    cmd += f"-mn resnet "
    cmd += f"-lr {lr} "
    cmd += f"-bz {batch_size} "
    cmd += f"-ep {epochs} "
    cmd += f"-dv auto "
    cmd += f"-sample {sample_ratio} "
    cmd += f"-min_per_class 10"
    
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset and recommend training parameters")
    parser.add_argument("-data", "--dataset", required=True, help="Dataset name (IP102 or D0)")
    parser.add_argument("-sample", "--sample_ratio", type=float, default=0.05, help="Sample ratio to analyze")
    parser.add_argument("-device", "--device_type", default="auto", help="Device type for adjustments")
    
    args = parser.parse_args()
    
    # Map dataset names to paths
    dataset_paths = {
        'IP102': 'IP102/train',
        'D0': 'unzip_D0/train'
    }
    
    if args.dataset not in dataset_paths:
        print(f"❌ Unsupported dataset: {args.dataset}")
        print(f"Available datasets: {list(dataset_paths.keys())}")
        exit(1)
    
    dataset_path = dataset_paths[args.dataset]
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Make sure you're running from the code directory")
        exit(1)
    
    # Analyze dataset
    stats = analyze_dataset(dataset_path, args.sample_ratio)
    
    # Get recommendations
    lr, epochs, batch_size = recommend_training_params(stats, args.device_type)
    
    # Generate command lines for different scenarios
    print(f"\n🚀 Recommended Command Lines:")
    print(f"\n1. Conservative (recommended):")
    cmd1 = generate_command_line(args.dataset, lr, epochs, batch_size, args.sample_ratio)
    print(f"   {cmd1}")
    
    print(f"\n2. More aggressive (if conservative is too slow):")
    cmd2 = generate_command_line(args.dataset, lr*1.5, max(3, epochs-2), min(64, batch_size*2), args.sample_ratio)
    print(f"   {cmd2}")
    
    print(f"\n3. Larger sample (for better results):")
    larger_ratio = min(0.2, args.sample_ratio * 2)
    cmd3 = generate_command_line(args.dataset, lr, epochs, batch_size, larger_ratio)
    print(f"   {cmd3}")
    
    print(f"\n4. Quick test (minimal settings):")
    cmd4 = generate_command_line(args.dataset, lr*0.5, 3, max(2, batch_size//2), args.sample_ratio*0.5)
    print(f"   {cmd4}")
    
    print(f"\n💡 Tips:")
    print(f"   - Start with the conservative command")
    print(f"   - If accuracy < 0.1 after training, try larger sample ratio or more epochs")
    print(f"   - Monitor training progress - loss should decrease consistently")
    print(f"   - For final training, use full dataset with these optimized parameters")

# Usage examples:
# python TrainingAnalyzer.py -data IP102 -sample 0.05
# python TrainingAnalyzer.py -data IP102 -sample 0.1 -device privateuseone
