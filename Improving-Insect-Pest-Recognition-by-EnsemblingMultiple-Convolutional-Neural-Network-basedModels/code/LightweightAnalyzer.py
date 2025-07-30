"""Lightweight Machine Training Analyzer - Optimized for Limited Resources"""

import psutil
import os
import argparse
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

def analyze_machine_capabilities():
    """Analyze current machine capabilities and recommend limits"""
    
    # Get system info
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"🖥️  Machine Analysis:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Total RAM: {memory_gb:.1f} GB")
    print(f"   Available RAM: {available_memory_gb:.1f} GB")
    
    # Determine machine category
    if cpu_count <= 4 and memory_gb <= 8:
        category = "Low-end"
        max_workers = 0
        max_batch = 4
        max_sample = 0.02
    elif cpu_count <= 8 and memory_gb <= 16:
        category = "Mid-range Laptop"  # Your machine
        max_workers = 2
        max_batch = 8
        max_sample = 0.03
    elif cpu_count <= 12 and memory_gb <= 32:
        category = "High-end Laptop"
        max_workers = 4
        max_batch = 16
        max_sample = 0.05
    else:
        category = "Workstation"
        max_workers = 6
        max_batch = 32
        max_sample = 0.1
    
    print(f"   Machine category: {category}")
    print(f"   Recommended max workers: {max_workers}")
    print(f"   Recommended max batch size: {max_batch}")
    print(f"   Recommended max sample ratio: {max_sample*100:.1f}%")
    
    return {
        'category': category,
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'available_memory_gb': available_memory_gb,
        'max_workers': max_workers,
        'max_batch': max_batch,
        'max_sample': max_sample
    }

def get_lightweight_recommendations(dataset_stats, machine_specs):
    """Get training recommendations optimized for lightweight machines"""
    
    samples_per_class = dataset_stats['samples_per_class']
    machine_category = machine_specs['category']
    
    print(f"\n🚀 Lightweight Training Recommendations:")
    print(f"   Target: Fast iteration for {machine_category}")
    
    # Ultra-conservative settings for your machine
    if machine_category == "Mid-range Laptop":
        # Your machine settings
        recommended_settings = {
            'sample_ratio': 0.02,  # 2% only
            'batch_size': 4,       # Very small batches
            'epochs': 5,           # Quick test
            'learning_rate': 0.0003,  # Conservative LR
            'num_workers': 0,      # No multiprocessing to save memory
            'image_size': 224,     # Standard size
            'min_per_class': 5     # Minimum viable
        }
    elif machine_category == "Low-end":
        recommended_settings = {
            'sample_ratio': 0.01,
            'batch_size': 2,
            'epochs': 3,
            'learning_rate': 0.0001,
            'num_workers': 0,
            'image_size': 224,
            'min_per_class': 3
        }
    else:
        recommended_settings = machine_specs['max_sample']
        recommended_settings = {
            'sample_ratio': machine_specs['max_sample'],
            'batch_size': machine_specs['max_batch'],
            'epochs': 8,
            'learning_rate': 0.0005,
            'num_workers': machine_specs['max_workers'],
            'image_size': 224,
            'min_per_class': 8
        }
    
    # Calculate resulting metrics
    sample_size = int(dataset_stats['total_samples'] * recommended_settings['sample_ratio'])
    samples_per_class_result = sample_size / dataset_stats['num_classes']
    
    print(f"   Sample ratio: {recommended_settings['sample_ratio']*100:.1f}%")
    print(f"   Sample size: {sample_size}")
    print(f"   Samples per class: {samples_per_class_result:.1f}")
    print(f"   Batch size: {recommended_settings['batch_size']}")
    print(f"   Epochs: {recommended_settings['epochs']}")
    print(f"   Learning rate: {recommended_settings['learning_rate']}")
    print(f"   Workers: {recommended_settings['num_workers']}")
    
    # Estimate training time
    batches_per_epoch = sample_size // recommended_settings['batch_size']
    estimated_time_per_epoch = batches_per_epoch * 2  # 2 seconds per batch estimate
    total_estimated_time = estimated_time_per_epoch * recommended_settings['epochs']
    
    print(f"\n⏱️  Time Estimates:")
    print(f"   Batches per epoch: {batches_per_epoch}")
    print(f"   Est. time per epoch: {estimated_time_per_epoch//60:.0f}m {estimated_time_per_epoch%60:.0f}s")
    print(f"   Est. total time: {total_estimated_time//60:.0f}m {total_estimated_time%60:.0f}s")
    
    if total_estimated_time > 1800:  # 30 minutes
        print(f"   ⚠️  Training may take over 30 minutes - consider smaller sample")
    
    return recommended_settings

def generate_lightweight_commands(dataset, settings):
    """Generate lightweight command variations"""
    
    base_cmd = f"python OptimizedSampleTrain.py -data {dataset} -optim Adam -sch steplr -l2 0.01 -do 0.2 -predt True -mn resnet -dv auto"
    
    commands = {
        'ultra_light': f"{base_cmd} -lr {settings['learning_rate']*0.5} -bz {max(1, settings['batch_size']//2)} -ep 3 -sample {settings['sample_ratio']*0.5} -min_per_class 3",
        
        'recommended': f"{base_cmd} -lr {settings['learning_rate']} -bz {settings['batch_size']} -ep {settings['epochs']} -sample {settings['sample_ratio']} -min_per_class {settings['min_per_class']}",
        
        'slightly_more': f"{base_cmd} -lr {settings['learning_rate']*1.2} -bz {settings['batch_size']} -ep {settings['epochs']+2} -sample {min(0.05, settings['sample_ratio']*1.5)} -min_per_class {settings['min_per_class']}",
        
        'cpu_only': f"{base_cmd} -lr {settings['learning_rate']*0.3} -bz {max(1, settings['batch_size']//2)} -ep {settings['epochs']} -sample {settings['sample_ratio']} -min_per_class {settings['min_per_class']} -dv cpu"
    }
    
    return commands

def analyze_dataset_lightweight(dataset_path, sample_ratio=0.02):
    """Quick dataset analysis for lightweight machines"""
    
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    try:
        dataset = ImageFolder(root=dataset_path, transform=basic_transform)
        
        # Get basic stats without loading all data
        class_counts = {}
        sample_count = 0
        for _, label in dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
            sample_count += 1
            if sample_count > 1000:  # Quick sample for speed
                break
        
        num_classes = len(class_counts)
        total_samples = len(dataset)
        
        # Estimate stats
        sample_size = int(total_samples * sample_ratio)
        samples_per_class = sample_size / num_classes
        
        return {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'sample_size': sample_size,
            'samples_per_class': samples_per_class,
            'estimated': sample_count <= 1000
        }
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight machine training analyzer")
    parser.add_argument("-data", "--dataset", required=True, help="Dataset name (IP102 or D0)")
    parser.add_argument("-sample", "--sample_ratio", type=float, default=0.02, help="Sample ratio to analyze")
    
    args = parser.parse_args()
    
    print("🔧 Lightweight Machine Training Analyzer")
    print("="*50)
    
    # Analyze machine
    machine_specs = analyze_machine_capabilities()
    
    # Dataset paths
    dataset_paths = {
        'IP102': 'IP102/train',
        'D0': 'unzip_D0/train'
    }
    
    if args.dataset not in dataset_paths:
        print(f"❌ Unsupported dataset: {args.dataset}")
        exit(1)
    
    dataset_path = dataset_paths[args.dataset]
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        exit(1)
    
    # Quick dataset analysis
    print(f"\n📊 Quick Dataset Analysis:")
    dataset_stats = analyze_dataset_lightweight(dataset_path, args.sample_ratio)
    
    if dataset_stats:
        print(f"   Total samples: {dataset_stats['total_samples']}")
        print(f"   Classes: {dataset_stats['num_classes']}")
        print(f"   Sample size ({args.sample_ratio*100:.1f}%): {dataset_stats['sample_size']}")
        print(f"   Samples per class: {dataset_stats['samples_per_class']:.1f}")
        
        # Get recommendations
        settings = get_lightweight_recommendations(dataset_stats, machine_specs)
        
        # Generate commands
        commands = generate_lightweight_commands(args.dataset, settings)
        
        print(f"\n🚀 Recommended Commands for {machine_specs['category']}:")
        
        print(f"\n1. ⚡ Ultra Light (fastest, 2-3 min):")
        print(f"   {commands['ultra_light']}")
        
        print(f"\n2. 🎯 Recommended (balanced, 5-10 min):")
        print(f"   {commands['recommended']}")
        
        print(f"\n3. 📈 Slightly More (better results, 10-15 min):")
        print(f"   {commands['slightly_more']}")
        
        print(f"\n4. 🐌 CPU Only (if GPU issues, slower):")
        print(f"   {commands['cpu_only']}")
        
        print(f"\n💡 Tips for Your Machine:")
        print(f"   - Start with Ultra Light to test setup")
        print(f"   - Close other applications to free memory")
        print(f"   - Monitor task manager during training")
        print(f"   - If memory errors occur, reduce batch size to 2")
        print(f"   - Use recommended command for actual testing")

# Usage: python LightweightAnalyzer.py -data IP102 -sample 0.02
