"""Simple Lightweight Training Recommender - No External Dependencies"""

import os
import argparse

def get_machine_recommendations():
    """Get recommendations for mid-range laptop (your specs)"""
    
    # Based on your specs: i7-1165G7, 16GB RAM, Intel Iris Xe
    print(f"🖥️  Machine Profile: Mid-range Laptop")
    print(f"   CPU: Intel i7-1165G7 (4 cores, 8 threads)")
    print(f"   RAM: ~16GB")
    print(f"   GPU: Intel Iris Xe (integrated)")
    print(f"   Category: DirectML-capable lightweight machine")
    
    return {
        'category': 'Mid-range Laptop',
        'max_batch': 4,        # Very conservative for integrated GPU
        'max_sample': 0.02,    # 2% max to avoid memory issues
        'max_workers': 0,      # Single-threaded to avoid overhead
        'recommended_epochs': 5
    }

def generate_ultra_lightweight_commands(dataset='IP102'):
    """Generate ultra-lightweight commands for your machine"""
    
    base_cmd = f"python OptimizedSampleTrain.py -data {dataset} -optim Adam -sch steplr -l2 0.01 -do 0.2 -predt True -mn resnet -dv auto"
    
    commands = {
        'quick_test': {
            'cmd': f"{base_cmd} -lr 0.0001 -bz 2 -ep 3 -sample 0.01 -min_per_class 3",
            'description': "Ultra-fast test (1-2 minutes)",
            'sample_size': "~450 images",
            'time_estimate': "1-2 minutes"
        },
        
        'minimal_viable': {
            'cmd': f"{base_cmd} -lr 0.0002 -bz 4 -ep 5 -sample 0.015 -min_per_class 5",
            'description': "Minimal viable training",
            'sample_size': "~675 images", 
            'time_estimate': "3-5 minutes"
        },
        
        'recommended': {
            'cmd': f"{base_cmd} -lr 0.0003 -bz 4 -ep 7 -sample 0.02 -min_per_class 8",
            'description': "Recommended for your machine",
            'sample_size': "~900 images",
            'time_estimate': "5-8 minutes"
        },
        
        'if_feeling_brave': {
            'cmd': f"{base_cmd} -lr 0.0004 -bz 6 -ep 10 -sample 0.025 -min_per_class 10",
            'description': "Maximum safe settings",
            'sample_size': "~1125 images",
            'time_estimate': "8-12 minutes"
        },
        
        'cpu_fallback': {
            'cmd': f"{base_cmd} -lr 0.0001 -bz 2 -ep 5 -sample 0.015 -min_per_class 5 -dv cpu",
            'description': "CPU-only if GPU fails",
            'sample_size': "~675 images",
            'time_estimate': "10-15 minutes"
        }
    }
    
    return commands

def calculate_sample_stats(total_samples=45095, num_classes=102, sample_ratio=0.02):
    """Calculate sample statistics"""
    
    sample_size = int(total_samples * sample_ratio)
    samples_per_class = sample_size / num_classes
    
    return {
        'sample_size': sample_size,
        'samples_per_class': samples_per_class,
        'sample_ratio_percent': sample_ratio * 100
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple lightweight training recommender")
    parser.add_argument("-data", "--dataset", default="IP102", help="Dataset name")
    
    args = parser.parse_args()
    
    print("🚀 Simple Lightweight Training Recommender")
    print("="*60)
    
    # Machine recommendations
    machine_specs = get_machine_recommendations()
    
    # Dataset stats (hardcoded for IP102)
    if args.dataset == "IP102":
        total_samples = 45095
        num_classes = 102
        print(f"\n📊 IP102 Dataset Stats:")
        print(f"   Total training samples: {total_samples:,}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Highly imbalanced (42 to 3,444 samples per class)")
    else:
        print(f"⚠️  Dataset {args.dataset} not optimized for this analyzer")
        total_samples = 10000  # fallback
        num_classes = 40
    
    # Generate commands
    commands = generate_ultra_lightweight_commands(args.dataset)
    
    print(f"\n🎯 Ultra-Lightweight Commands for Your Machine:")
    print(f"   Target: Intel i7-1165G7 + 16GB RAM + Iris Xe")
    
    for i, (key, cmd_info) in enumerate(commands.items(), 1):
        # Calculate stats for this command
        sample_ratio = float(cmd_info['cmd'].split('-sample ')[1].split(' ')[0])
        stats = calculate_sample_stats(total_samples, num_classes, sample_ratio)
        
        print(f"\n{i}. {cmd_info['description'].upper()}")
        print(f"   Sample: {stats['sample_ratio_percent']:.1f}% ({stats['sample_size']} images)")
        print(f"   Per class: {stats['samples_per_class']:.1f} samples")
        print(f"   Time: {cmd_info['time_estimate']}")
        print(f"   Command:")
        print(f"   {cmd_info['cmd']}")
    
    print(f"\n💡 RECOMMENDATIONS FOR YOUR MACHINE:")
    print(f"   🟢 START HERE: Try 'quick_test' first (1-2 min)")
    print(f"   🟡 THEN: Run 'recommended' for actual testing")
    print(f"   🔴 AVOID: Batch sizes > 6, sample ratios > 0.025")
    print(f"   💾 MEMORY: Close browsers/apps before training")
    print(f"   📊 MONITORING: Watch Task Manager for memory usage")
    
    print(f"\n⚠️  TROUBLESHOOTING:")
    print(f"   - If 'CUDA out of memory': Use CPU fallback")
    print(f"   - If training freezes: Reduce batch size to 2")
    print(f"   - If very slow: Check antivirus isn't scanning")
    print(f"   - If accuracy < 0.05: Try 'if_feeling_brave' settings")
    
    print(f"\n🎓 WHAT TO EXPECT:")
    print(f"   - Quick test: ~0.05-0.15 accuracy (5-15%)")
    print(f"   - Recommended: ~0.10-0.25 accuracy (10-25%)")
    print(f"   - Good baseline for full training parameter tuning")

# Usage: python SimpleLightweight.py -data IP102
