"""Optimized Sample Training with Better Parameters for Learning"""

from utils.myFunctions import train_model, evaluate_model, initialize_model, myScheduler
import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils.myFunctions import initialize_model
from utils.myFunctions import resizePadding
import torch.optim as optim
import argparse
import torch_directml
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from performance_config import PerformanceConfig, optimize_torch_settings, get_optimal_image_size
import warnings
import gc
import time
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Custom strtobool function
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0)."""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")

ap = argparse.ArgumentParser()

# Standard arguments
ap.add_argument("-data", "--dataset", required=True, help="Choose dataset: IP102 or D0")
ap.add_argument("-optim", "--optimizer", required=True, help="Optimizer function : Only SGD or Adam")
ap.add_argument("-sch", "--scheduler", required=True, help="Scheduler:\nnone\nsteplr\nexpdecay\nmyscheduler")
ap.add_argument("-l2", "--weight_decay", required=True, help="L2 regularzation")
ap.add_argument("-do", "--dropout", required=True, help="Dropout rate")
ap.add_argument("-predt", "--use_pretrained", required=True, help="Use pretrained model's weight")
ap.add_argument("-mn", "--model_name", required=True, help="Model name: resnet, residual-attention, fpn")
ap.add_argument("-lr", "--learning_rate", required=True, help="Inital learning rate")
ap.add_argument("-bz", "--batch_size", required=True, help="Batch size")
ap.add_argument("-ep", "--epochs", required=True, help="Number of Epochs")
ap.add_argument("-dv", "--device", required=True, help="Device type")

# Sample-specific parameters with better defaults
ap.add_argument("-sample", "--sample_ratio", required=False, help="Percentage of dataset to use (0.05 = 5%)", default='0.05')  # Increased default
ap.add_argument("-seed", "--random_seed", required=False, help="Random seed for reproducible sampling", default='42')
ap.add_argument("-min_per_class", "--min_samples_per_class", required=False, help="Minimum samples per class", default='10')

ap.add_argument("-istra", "--istrain", required=False, help="Train mode", default='True')
ap.add_argument("-iseva", "--iseval", required=False, help="Eval mode", default='True')
ap.add_argument("-issavck", "--issavechkp", required=False, help="Save checkpoint", default='False')
ap.add_argument("-issavmd", "--issavemodel", required=False, help="Save model", default='False')

args = vars(ap.parse_args())

dataset_info = {
    'IP102': 102,
    'D0': 40
}

def create_balanced_sample_dataset(dataset, sample_ratio, min_samples_per_class=10, random_seed=42):
    """Create a better balanced sample ensuring minimum samples per class"""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    total_samples = len(dataset)
    min_samples_per_class = int(min_samples_per_class)
    
    # Get all class indices
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    num_classes = len(class_indices)
    target_total_samples = int(total_samples * sample_ratio)
    
    # Calculate samples per class
    base_samples_per_class = max(min_samples_per_class, target_total_samples // num_classes)
    
    sampled_indices = []
    actual_samples_per_class = {}
    
    for class_label, indices in class_indices.items():
        available_samples = len(indices)
        
        if available_samples <= base_samples_per_class:
            # Use all available samples
            sampled_indices.extend(indices)
            actual_samples_per_class[class_label] = available_samples
        else:
            # Sample the required number
            selected = random.sample(indices, base_samples_per_class)
            sampled_indices.extend(selected)
            actual_samples_per_class[class_label] = base_samples_per_class
    
    total_sampled = len(sampled_indices)
    actual_ratio = total_sampled / total_samples
    
    print(f"📊 Balanced Sampling Results:")
    print(f"   Original dataset size: {total_samples}")
    print(f"   Target ratio: {sample_ratio*100:.1f}%")
    print(f"   Actual ratio: {actual_ratio*100:.1f}%")
    print(f"   Sample dataset size: {total_sampled}")
    print(f"   Classes represented: {num_classes}")
    print(f"   Min samples per class: {min(actual_samples_per_class.values())}")
    print(f"   Max samples per class: {max(actual_samples_per_class.values())}")
    print(f"   Avg samples per class: {total_sampled/num_classes:.1f}")
    
    return Subset(dataset, sampled_indices), actual_ratio

def get_optimized_training_params(device, sample_size, num_classes, base_lr):
    """Get optimized training parameters based on sample characteristics"""
    
    # Calculate complexity metrics
    samples_per_class = sample_size / num_classes
    complexity_score = min(1.0, samples_per_class / 50)  # 50 samples per class = full complexity
    
    # Adjust learning rate based on sample size and device
    if samples_per_class < 5:
        lr_multiplier = 0.1  # Very conservative for tiny samples
    elif samples_per_class < 20:
        lr_multiplier = 0.3  # Conservative
    elif samples_per_class < 50:
        lr_multiplier = 0.7  # Moderate
    else:
        lr_multiplier = 1.0  # Full learning rate
    
    # Device-specific adjustments
    if device.type in ["privateuseone", "dml"]:
        lr_multiplier *= 0.5  # More conservative for DirectML
    elif device.type == "cpu":
        lr_multiplier *= 0.3  # Much more conservative for CPU
    
    optimized_lr = base_lr * lr_multiplier
    
    # Suggest minimum epochs based on sample characteristics
    if samples_per_class < 10:
        min_epochs = 10
    elif samples_per_class < 30:
        min_epochs = 5
    else:
        min_epochs = 3
    
    print(f"📈 Training Optimization Recommendations:")
    print(f"   Samples per class: {samples_per_class:.1f}")
    print(f"   Complexity score: {complexity_score:.2f}")
    print(f"   Recommended learning rate: {optimized_lr:.6f} (multiplier: {lr_multiplier:.2f})")
    print(f"   Minimum recommended epochs: {min_epochs}")
    
    return {
        'optimized_lr': optimized_lr,
        'lr_multiplier': lr_multiplier,
        'min_epochs': min_epochs,
        'complexity_score': complexity_score,
        'samples_per_class': samples_per_class
    }

if __name__ == "__main__":
    # Device detection
    def get_device():
        try:
            if torch.cuda.is_available():
                test_tensor = torch.rand(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("Using CUDA (Dedicated GPU)")
                return torch.device("cuda")
        except Exception as e:
            print(f"CUDA test failed: {e}")
        
        try:
            if torch.backends.mps.is_available():
                print("Using Metal Performance Shaders (MPS)")
                return torch.device("mps")
        except Exception as e:
            print(f"MPS test failed: {e}")
        
        try:
            import torch_directml
            dml_device = torch_directml.device()
            test_tensor = torch.rand(1).to(dml_device)
            del test_tensor
            print("Using DirectML (Integrated GPU)")
            return dml_device
        except (ImportError, Exception) as e:
            print(f"DirectML not available or failed: {e}")
        
        print("Using CPU")
        return torch.device("cpu")

    device = get_device()
    optimize_torch_settings(device)
    
    # Get performance configuration
    perf_config = PerformanceConfig(device)
    settings = perf_config.get_optimal_settings()
    
    print(f"Device: {device}")
    print(f"Performance settings: {settings}")
    
    # Parse arguments
    batch_size = int(args['batch_size'])
    num_epochs = int(args['epochs'])
    sample_ratio = float(args['sample_ratio'])
    random_seed = int(args['random_seed'])
    min_samples_per_class = int(args['min_samples_per_class'])

    # Hyperparameters
    init_lr = float(args['learning_rate'])
    weight_decay = float(args['weight_decay'])
    dropout = float(args['dropout'])
    optimizer_name = args['optimizer']
    scheduler = args['scheduler']
    use_pretrained = strtobool(args['use_pretrained'])
    model_name = args['model_name']
    dataset_name = args['dataset']
    n_classes = dataset_info[dataset_name]

    # Initialize the model
    model, input_size = initialize_model(model_name=model_name, num_classes=n_classes, use_pretrained=use_pretrained, dropout=dropout)
    
    if settings['compile_model'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compilation enabled")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    model = model.to(device)

    # Setup paths and configs
    exp_name = 'insect_recognition_optimized_sample'
    is_train = strtobool(args['istrain'])
    is_eval = strtobool(args['iseval'])
    save_model_dict = strtobool(args['issavemodel'])
    is_save_checkpoint = strtobool(args['issavechkp'])
    checkpoint = f'checkpoint_optimized_sample_{dataset_name}_{model_name}.pt'

    if dataset_name == 'D0':
        dataset_name = 'unzip_D0'

    # Paths
    train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
    valid_root_path = os.path.join(os.getcwd(), dataset_name, 'valid')
    test_root_path = os.path.join(os.getcwd(), dataset_name, 'test')

    model_ft, input_size = initialize_model(model_name, n_classes, dropout=dropout, use_pretrained=use_pretrained)

    # Get optimal image size
    optimal_image_size = get_optimal_image_size(device, 'medium')
    print(f"Using image size: {optimal_image_size}x{optimal_image_size}")
    print(f"Sample ratio: {sample_ratio*100:.1f}%")
    
    # Enhanced data transforms with stronger augmentation for small samples
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((optimal_image_size, optimal_image_size)),
            transforms.RandomHorizontalFlip(p=0.7),  # Increased augmentation
            transforms.RandomVerticalFlip(p=0.3),    # Additional augmentation
            transforms.RandomRotation(degrees=30),   # More rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Stronger color augmentation
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Geometric augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((optimal_image_size, optimal_image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load and create balanced sample datasets
    print("Loading original datasets...")
    original_train_set = ImageFolder(root=train_root_path, transform=data_transforms['train'])
    original_valid_set = ImageFolder(root=valid_root_path, transform=data_transforms['val'])
    
    print("Creating balanced sample datasets...")
    train_set, actual_train_ratio = create_balanced_sample_dataset(original_train_set, sample_ratio, min_samples_per_class, random_seed)
    valid_set, actual_valid_ratio = create_balanced_sample_dataset(original_valid_set, sample_ratio, min_samples_per_class, random_seed)

    # Get optimized training parameters
    optimization_params = get_optimized_training_params(device, len(train_set), n_classes, init_lr)
    
    # Suggest better epoch count if too low
    if num_epochs < optimization_params['min_epochs']:
        print(f"⚠️  WARNING: {num_epochs} epochs may be too few!")
        print(f"   Recommended minimum: {optimization_params['min_epochs']} epochs")
        print(f"   For good results with this sample size, try: {optimization_params['min_epochs']*2} epochs")

    # Calculate optimal batch sizes
    optimal_batch_size = max(1, int(batch_size * settings['batch_size_multiplier']))
    current_batch_size = optimal_batch_size
    
    print(f"Using batch size: {current_batch_size}")
    
    # DataLoader settings
    dataloader_kwargs = {
        'num_workers': settings['num_workers'],
        'pin_memory': settings['pin_memory'],
        'persistent_workers': settings['persistent_workers'] and settings['num_workers'] > 0,
        'prefetch_factor': settings['prefetch_factor'] if settings['num_workers'] > 0 else 2
    }

    if is_train:
        # Create DataLoaders
        train_loader = DataLoader(train_set, batch_size=current_batch_size, shuffle=True, **dataloader_kwargs)
        valid_loader = DataLoader(valid_set, batch_size=current_batch_size, shuffle=False, **dataloader_kwargs)

        datasets_dict = {'train': train_loader, 'val': valid_loader}

        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()

        # Use optimized learning rate
        dynamic_learning_rate = optimization_params['optimized_lr']
        print(f"Using optimized learning rate: {dynamic_learning_rate}")

        # Create optimizer
        if optimizer_name.lower() == 'sgd':
            optimizer_ft = optim.SGD(params_to_update, lr=dynamic_learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer_ft = optim.Adam(params_to_update, lr=dynamic_learning_rate, betas=(0.9, 0.999),
                                eps=1e-08, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Create scheduler with adjusted parameters
        if scheduler.lower() == 'none':
            scheduler_ft = None
        elif scheduler.lower() == 'expdecay':
            scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.9)  # Less aggressive decay
        elif scheduler.lower() == 'steplr':
            step_size = max(3, num_epochs // 3)  # Adjust step size based on epochs
            scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.5)  # Less aggressive
        elif scheduler.lower() == 'myscheduler':
            scheduler_ft = myScheduler(optimizer_ft, gamma=0.5)  # Less aggressive
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        # Initialize training components
        scaler = GradScaler(enabled=settings['mixed_precision'] and device.type in ["cuda", "dml", "privateuseone"])
        criterion = nn.CrossEntropyLoss()
        
        # Memory management
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Training
        print("🎯 Starting optimized sample training...")
        start_time = time.time()
        
        try:
            model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                        scheduler=scheduler_ft, num_epochs=num_epochs, checkpointFn=checkpoint,
                                        device=device, is_save_checkpoint=is_save_checkpoint,
                                        is_load_checkpoint=False)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            print(f"\n✅ Optimized sample training completed!")
            print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
            
            # Analyze results
            if 'hist' in locals() and hist:
                final_train_acc = hist['train_acc'][-1].item() if hist['train_acc'] else 0
                final_val_acc = hist['val_acc'][-1].item() if hist['val_acc'] else 0
                
                print(f"📊 Training Results:")
                print(f"   Final training accuracy: {final_train_acc:.4f}")
                print(f"   Final validation accuracy: {final_val_acc:.4f}")
                
                if final_val_acc > 0.1:  # 10% accuracy threshold
                    print("✅ Good! Model is learning effectively")
                elif final_val_acc > 0.01:  # 1% accuracy threshold
                    print("⚠️  Model showing some learning, consider more epochs or larger sample")
                else:
                    print("❌ Poor learning detected. Recommendations:")
                    print("   - Increase sample ratio (try -sample 0.1 or 0.2)")
                    print("   - Increase epochs (try -ep 10 or higher)")
                    print("   - Check if pretrained weights are being used")
            
            # Time estimation
            estimated_full_time = training_time / actual_train_ratio
            print(f"\n📊 Time Estimation:")
            print(f"   Estimated time per epoch (full dataset): {estimated_full_time/num_epochs:.2f} seconds ({estimated_full_time/num_epochs/60:.1f} minutes)")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

    # Evaluation
    if is_eval and 'model_ft' in locals():
        print("Starting optimized sample evaluation...")
        original_test_set = ImageFolder(root=test_root_path, transform=data_transforms['val'])
        test_set, _ = create_balanced_sample_dataset(original_test_set, sample_ratio, min_samples_per_class, random_seed)
        test_loader = DataLoader(test_set, batch_size=current_batch_size, shuffle=False, **dataloader_kwargs)

        evaluate_model(model_ft, testloader=test_loader, 
                        path_weight_dict=None, device=device, model_hyper={
                            'batch_size': current_batch_size,
                            'num_epochs': num_epochs,
                            'init_lr': dynamic_learning_rate,
                            'weight_decay': weight_decay,
                            'dropout': dropout,
                            'model_name': model_name,
                            'exp_name': exp_name,
                            'dataset': dataset_name,
                            'sample_ratio': sample_ratio,
                            'optimization_params': optimization_params
                        })

    print("\n" + "="*70)
    print("🎯 OPTIMIZED SAMPLE TRAINING SUMMARY")
    print("="*70)
    print(f"Dataset: {args['dataset']}")
    print(f"Model: {model_name}")
    print(f"Sample ratio: {sample_ratio*100:.1f}% (actual: {actual_train_ratio*100:.1f}%)")
    print(f"Sample size: {len(train_set) if 'train_set' in locals() else 'N/A'}")
    print(f"Samples per class (avg): {optimization_params['samples_per_class']:.1f}")
    print(f"Device: {device}")
    print(f"Batch size: {current_batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimized learning rate: {dynamic_learning_rate:.6f}")
    print(f"LR multiplier applied: {optimization_params['lr_multiplier']:.2f}")
    print("="*70)

# Usage examples:
# python OptimizedSampleTrain.py -data IP102 -optim Adam -sch steplr -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 5 -dv auto -sample 0.05
# python OptimizedSampleTrain.py -data IP102 -optim Adam -sch steplr -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 10 -dv auto -sample 0.1 -min_per_class 15
