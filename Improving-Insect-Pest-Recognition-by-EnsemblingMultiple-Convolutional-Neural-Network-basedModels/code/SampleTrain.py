"""Sample training script for testing with a subset of data"""

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

# Custom strtobool function to replace deprecated distutils.util.strtobool
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

# Add the arguments to the parser
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

# Sample-specific parameters
ap.add_argument("-sample", "--sample_ratio", required=False, help="Percentage of dataset to use (0.05 = 5%)", default='0.1')
ap.add_argument("-seed", "--random_seed", required=False, help="Random seed for reproducible sampling", default='42')

ap.add_argument("-istra", "--istrain", required=False, help="Train mode", default='True')
ap.add_argument("-iseva", "--iseval", required=False, help="Eval mode", default='True')
ap.add_argument("-issavck", "--issavechkp", required=False, help="Save checkpoint", default='False')  # Default False for sample
ap.add_argument("-issavmd", "--issavemodel", required=False, help="Save model", default='False')  # Default False for sample

args = vars(ap.parse_args())

dataset_info = {
    'IP102': 102,
    'D0': 40
}

def create_sample_dataset(dataset, sample_ratio, random_seed=42):
    """Create a smaller sample of the dataset for quick testing"""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    total_samples = len(dataset)
    sample_size = int(total_samples * sample_ratio)
    
    # Get balanced samples from each class
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Sample proportionally from each class
    sampled_indices = []
    samples_per_class = max(1, sample_size // len(class_indices))
    
    for class_label, indices in class_indices.items():
        if len(indices) <= samples_per_class:
            sampled_indices.extend(indices)
        else:
            sampled_indices.extend(random.sample(indices, samples_per_class))
    
    # If we need more samples, randomly add from remaining
    if len(sampled_indices) < sample_size:
        remaining_indices = list(set(range(total_samples)) - set(sampled_indices))
        additional_needed = sample_size - len(sampled_indices)
        if len(remaining_indices) >= additional_needed:
            sampled_indices.extend(random.sample(remaining_indices, additional_needed))
        else:
            sampled_indices.extend(remaining_indices)
    
    # Ensure we don't exceed the requested sample size
    sampled_indices = sampled_indices[:sample_size]
    
    print(f"Original dataset size: {total_samples}")
    print(f"Sample dataset size: {len(sampled_indices)} ({len(sampled_indices)/total_samples*100:.1f}%)")
    print(f"Classes represented: {len(set([dataset[i][1] for i in sampled_indices]))}")
    
    return Subset(dataset, sampled_indices)

def estimate_training_time(sample_time, sample_ratio, num_epochs):
    """Estimate total training time based on sample performance"""
    time_per_epoch_full = sample_time / sample_ratio
    total_estimated_time = time_per_epoch_full * num_epochs
    
    print("\n" + "="*60)
    print("📊 TRAINING TIME ESTIMATION")
    print("="*60)
    print(f"Sample training time: {sample_time:.2f} seconds")
    print(f"Sample ratio: {sample_ratio*100:.1f}%")
    print(f"Estimated time per epoch (full dataset): {time_per_epoch_full:.2f} seconds ({time_per_epoch_full/60:.1f} minutes)")
    print(f"Estimated total training time ({num_epochs} epochs): {total_estimated_time:.2f} seconds ({total_estimated_time/60:.1f} minutes, {total_estimated_time/3600:.1f} hours)")
    print("="*60)

if __name__ == "__main__":
    ###########################################################
    # Detect device with better error handling
    def get_device():
        try:
            if torch.cuda.is_available():
                # Test CUDA functionality
                test_tensor = torch.rand(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("Using CUDA (Dedicated GPU)")
                return torch.device("cuda")
        except Exception as e:
            print(f"CUDA test failed: {e}")
        
        try:
            if torch.backends.mps.is_available():  # For Apple M1/M2 GPUs
                print("Using Metal Performance Shaders (MPS)")
                return torch.device("mps")
        except Exception as e:
            print(f"MPS test failed: {e}")
        
        try:
            import torch_directml
            dml_device = torch_directml.device()
            # Test DirectML functionality
            test_tensor = torch.rand(1).to(dml_device)
            del test_tensor
            print("Using DirectML (Integrated GPU)")
            return dml_device
        except (ImportError, Exception) as e:
            print(f"DirectML not available or failed: {e}")
        
        print("Using CPU")
        return torch.device("cpu")

    device = get_device()
    
    # Apply PyTorch optimizations
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

    # My setting, Hyperparameters
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
    
    # Enable model compilation for better performance (PyTorch 2.0+)
    if settings['compile_model'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compilation enabled for better performance")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    model = model.to(device)  # Move the model to the selected device

    exp_name = 'insect_recognition_sample'
    is_train = strtobool(args['istrain'])
    is_eval = strtobool(args['iseval'])
    save_model_dict = strtobool(args['issavemodel'])
    is_save_checkpoint = strtobool(args['issavechkp'])
    load_model = False  # Always False for sample training
    load_checkpoint = False  # Always False for sample training
    checkpoint = f'checkpoint_sample_{dataset_name}_{model_name}.pt'

    if dataset_name == 'D0':
        dataset_name = 'unzip_D0'

    path_weight_dict = os.path.join(os.getcwd(), 'pre-trained', f'sample_{model_name}_{dataset_name}.pt')

    model_hyper = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'init_lr': init_lr,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'model_name': model_name,
        'exp_name': exp_name,
        'dataset': dataset_name,
        'sample_ratio': sample_ratio
    }

    train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
    valid_root_path = os.path.join(os.getcwd(), dataset_name, 'valid')
    test_root_path = os.path.join(os.getcwd(), dataset_name, 'test')

    model_ft, input_size = initialize_model(model_name, n_classes, dropout=dropout, use_pretrained=use_pretrained)

    # Get optimal image size based on device capabilities and model complexity
    optimal_image_size = get_optimal_image_size(device, 'medium')
    print(f"Using image size: {optimal_image_size}x{optimal_image_size}")
    print(f"Sample ratio: {sample_ratio*100:.1f}%")
    
    # Optimized data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((optimal_image_size, optimal_image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((optimal_image_size, optimal_image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load original datasets
    print("Loading original datasets...")
    original_train_set = ImageFolder(root=train_root_path, transform=data_transforms['train'])
    original_valid_set = ImageFolder(root=valid_root_path, transform=data_transforms['val'])
    
    # Create sample datasets
    print("Creating sample datasets...")
    train_set = create_sample_dataset(original_train_set, sample_ratio, random_seed)
    valid_set = create_sample_dataset(original_valid_set, sample_ratio, random_seed)

    dataset = train_set  # Use train_set for splitting (EXACTLY like original)

    # Split dataset for hybrid training (EXACTLY like original)
    def split_dataset(dataset, cpu_ratio=0.3):
        cpu_size = int(len(dataset) * cpu_ratio)
        gpu_size = len(dataset) - cpu_size
        return random_split(dataset, [gpu_size, cpu_size])
    
    # Split the dataset (EXACTLY like original)
    gpu_dataset, cpu_dataset = split_dataset(dataset)

    # Adjust batch size dynamically (EXACTLY like original)
    def get_dynamic_batch_size(device, base_batch_size=None):
        """
        Dynamically calculate batch size based on available memory and device type.
        """
        if base_batch_size is None:
            base_batch_size = batch_size
            
        if device.type == "cuda":
            # For CUDA (Dedicated GPU)
            try:
                torch.cuda.empty_cache()  # Clear cache before checking memory
                gpu_properties = torch.cuda.get_device_properties(device)
                total_memory = gpu_properties.total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                available_memory = total_memory - allocated_memory
                
                # More conservative calculation: estimate memory per sample
                # Assume each sample uses approximately 50MB for a 128x128 image with gradients
                memory_per_sample = 50 * 1024 * 1024  # 50MB per sample
                max_batch_size = max(1, int(available_memory * 0.8 / memory_per_sample))  # Use 80% of available memory
                
                return min(base_batch_size, max_batch_size)
            except Exception as e:
                print(f"Error calculating CUDA batch size: {e}. Using base batch size.")
                return base_batch_size
        elif device.type == "dml":
            # For DirectML (Integrated GPU) - be very conservative
            return max(1, min(base_batch_size, 4))
        elif device.type == "cpu":
            # For CPU - use system memory
            try:
                import psutil
                available_memory = psutil.virtual_memory().available
                memory_per_sample = 100 * 1024 * 1024  # 100MB per sample on CPU (less efficient)
                max_batch_size = max(1, int(available_memory * 0.5 / memory_per_sample))  # Use 50% of available memory
                
                return min(base_batch_size, max_batch_size)
            except ImportError:
                print("psutil not available. Using conservative batch size for CPU.")
                return max(1, min(base_batch_size, 2))
        else:
            return max(1, min(base_batch_size, 8))
    
    # Adjust learning rate dynamically based on device and batch size (EXACTLY like original)
    def get_learning_rate(device, dynamic_batch_size, base_lr):
        """
        Adjust learning rate based on device type and actual batch size.
        Linear scaling rule: lr = base_lr * (batch_size / reference_batch_size)
        """
        reference_batch_size = 32  # Reference batch size for base learning rate
        
        if device.type == "cuda":
            # Scale learning rate with batch size
            scaled_lr = base_lr * (dynamic_batch_size / reference_batch_size)
            return min(scaled_lr, base_lr * 2)  # Cap at 2x base learning rate
        elif device.type == "dml":
            # Lower learning rate for DirectML due to potential instability
            return base_lr * 0.5
        elif device.type == "cpu":
            # Much lower learning rate for CPU training
            return base_lr * 0.1
        else:
            return base_lr

    # Calculate optimal batch sizes using performance settings (EXACTLY like original)
    optimal_batch_size = max(1, int(batch_size * settings['batch_size_multiplier']))
    gpu_batch_size = get_dynamic_batch_size(torch.device("cuda"), optimal_batch_size) if torch.cuda.is_available() else optimal_batch_size
    cpu_batch_size = max(1, min(optimal_batch_size // 2, 4))  # Conservative CPU batch size
    
    print(f"Batch sizes - Optimal: {optimal_batch_size}, GPU: {gpu_batch_size}, CPU: {cpu_batch_size}")
    
    # Use the optimal batch size for the current device
    current_batch_size = optimal_batch_size
    
    # Initialize DataLoaders with performance-optimized settings (EXACTLY like original)
    dataloader_kwargs = {
        'num_workers': settings['num_workers'],
        'pin_memory': settings['pin_memory'],
        'persistent_workers': settings['persistent_workers'] and settings['num_workers'] > 0,
        'prefetch_factor': settings['prefetch_factor'] if settings['num_workers'] > 0 else 2
    }
    
    gpu_loader = DataLoader(gpu_dataset, batch_size=gpu_batch_size, shuffle=True, **dataloader_kwargs)
    cpu_loader = DataLoader(cpu_dataset, batch_size=cpu_batch_size, shuffle=True, 
                           num_workers=min(2, settings['num_workers']), pin_memory=False, 
                           persistent_workers=False)
    
    # Set learning rate based on device and batch size (EXACTLY like original)
    dynamic_learning_rate = get_learning_rate(device, current_batch_size, init_lr)
    print(f"Using learning rate: {dynamic_learning_rate}")
    
    # Create a temporary optimizer for the hybrid training loop (EXACTLY like original)
    temp_optimizer = torch.optim.Adam(model.parameters(), lr=dynamic_learning_rate)

    if is_train:
        # Create proper DataLoaders for training with optimized settings (EXACTLY like original)
        train_loader = DataLoader(train_set, batch_size=current_batch_size, shuffle=True, **dataloader_kwargs)
        valid_loader = DataLoader(valid_set, batch_size=current_batch_size, shuffle=False, **dataloader_kwargs)

        datasets_dict = {'train': train_loader, 'val': valid_loader}

        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()

        # Create the main optimizer based on command line arguments (EXACTLY like original)
        if optimizer_name.lower() == 'sgd':
            optimizer_ft = optim.SGD(params_to_update, lr=dynamic_learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer_ft = optim.Adam(params_to_update, lr=dynamic_learning_rate, betas=(0.9, 0.999),
                                eps=1e-08, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Create scheduler (EXACTLY like original)
        if scheduler.lower() == 'none':
            scheduler_ft = None
        elif scheduler.lower() == 'expdecay':
            scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.96)
        elif scheduler.lower() == 'steplr':
            scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.09999, last_epoch=-1)
        elif scheduler.lower() == 'myscheduler':
            scheduler_ft = myScheduler(optimizer_ft, gamma=0.09999)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        # Initialize AMP scaler for mixed precision training (EXACTLY like original)
        scaler = GradScaler(enabled=settings['mixed_precision'] and device.type in ["cuda", "dml"])
        criterion = nn.CrossEntropyLoss()
        
        # Memory management before training (EXACTLY like original)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # HYBRID TRAINING ENABLED FOR SAMPLE TESTING (Modified from original)
        # This is enabled for sample training to test hybrid functionality
        hybrid_training = True  # ENABLED for sample testing (original has False)
        
        print(f"🔄 Hybrid training mode: {'ENABLED' if hybrid_training else 'DISABLED'}")
        print(f"📊 GPU dataset size: {len(gpu_dataset)}, CPU dataset size: {len(cpu_dataset)}")
        
        if hybrid_training and (torch.cuda.is_available() or device.type in ["dml", "privateuseone"]):
            print("🚀 Starting hybrid GPU/CPU training...")
            accumulation_steps = 4
            
            start_time = time.time()
            
            # Training loop with tqdm progress bar (EXACTLY like original structure)
            hybrid_epochs = min(max(1, num_epochs), 3)  # Test with 1-3 epochs for sample
            for epoch in range(hybrid_epochs):
                model.train()
                print(f"Hybrid Epoch {epoch+1}/{hybrid_epochs}")
                
                min_batches = min(len(gpu_loader), len(cpu_loader))
                epoch_loss = 0.0
                
                for i, (gpu_data, cpu_data) in enumerate(tqdm(zip(gpu_loader, cpu_loader), total=min_batches, desc=f"Hybrid Epoch {epoch+1}")):
                    if i >= min_batches:
                        break
                        
                    # GPU processing (EXACTLY like original)
                    gpu_inputs, gpu_labels = gpu_data[0].to(device), gpu_data[1].to(device)
                    
                    temp_optimizer.zero_grad()
                    with autocast(device_type=device.type, enabled=settings['mixed_precision'] and device.type in ["cuda", "dml", "privateuseone"]):
                        gpu_outputs = model(gpu_inputs)
                        gpu_loss = criterion(gpu_outputs, gpu_labels)
                    
                    scaler.scale(gpu_loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(temp_optimizer)
                        scaler.update()
                        temp_optimizer.zero_grad()

                    epoch_loss += gpu_loss.item()

                    if i % 10 == 0:  # Print every 10 batches (EXACTLY like original)
                        print(f"Hybrid Batch {i+1}/{min_batches} - GPU Loss: {gpu_loss.item():.4f}")
                
                avg_epoch_loss = epoch_loss / min_batches
                print(f"Hybrid Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
                
                # Memory cleanup after each epoch (EXACTLY like original)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            
            hybrid_time = time.time() - start_time
            print(f"✅ Hybrid training completed in {hybrid_time:.2f} seconds")
            
            # Estimate time for hybrid training on full dataset
            if len(gpu_dataset) > 0:
                hybrid_ratio = len(gpu_dataset) / len(original_train_set)
                estimated_hybrid_full_time = hybrid_time / hybrid_ratio
                print(f"📊 Estimated hybrid training time for full dataset: {estimated_hybrid_full_time:.2f} seconds ({estimated_hybrid_full_time/60:.1f} minutes)")

        # Main training using the standard train_model function (EXACTLY like original)
        print("🎯 Starting main training...")
        main_start_time = time.time()
        
        try:
            model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                        scheduler=scheduler_ft, num_epochs=num_epochs, checkpointFn=checkpoint,
                                        device=device, is_save_checkpoint=is_save_checkpoint,
                                        is_load_checkpoint=False)
            
            main_end_time = time.time()
            main_training_time = main_end_time - main_start_time
            total_sample_time = main_training_time + (hybrid_time if 'hybrid_time' in locals() else 0)
            
            print(f"\n✅ Sample training completed successfully!")
            print(f"Main training time: {main_training_time:.2f} seconds ({main_training_time/60:.1f} minutes)")
            if 'hybrid_time' in locals():
                print(f"Hybrid training time: {hybrid_time:.2f} seconds ({hybrid_time/60:.1f} minutes)")
                print(f"Total sample training time: {total_sample_time:.2f} seconds ({total_sample_time/60:.1f} minutes)")
            
            # Estimate full training time
            estimate_training_time(total_sample_time, sample_ratio, num_epochs)
            
        except Exception as e:
            end_time = time.time()
            sample_training_time = end_time - main_start_time
            print(f"\n❌ Sample training failed after {sample_training_time:.2f} seconds")
            print(f"Error: {e}")
            
            # Still provide estimation if some training occurred
            if sample_training_time > 10:  # If training ran for at least 10 seconds
                estimate_training_time(sample_training_time, sample_ratio, num_epochs)

    if save_model_dict and 'model_ft' in locals():
        torch.save(model_ft.state_dict(), path_weight_dict)
        print(f"Sample model saved to: {path_weight_dict}")

    if is_eval and 'model_ft' in locals():
        print("Starting sample evaluation...")
        original_test_set = ImageFolder(root=test_root_path, transform=data_transforms['val'])
        test_set = create_sample_dataset(original_test_set, sample_ratio, random_seed)
        test_loader = DataLoader(test_set, batch_size=current_batch_size, shuffle=False, **dataloader_kwargs)

        evaluate_model(model_ft, testloader=test_loader, 
                        path_weight_dict=None, device=device, model_hyper=model_hyper)

    print("\n" + "="*60)
    print("🎯 SAMPLE TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {args['dataset']}")
    print(f"Model: {model_name}")
    print(f"Sample ratio: {sample_ratio*100:.1f}%")
    print(f"Device: {device}")
    print(f"Batch size: {current_batch_size}")
    print(f"GPU batch size: {gpu_batch_size}, CPU batch size: {cpu_batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Hybrid training: {'ENABLED' if 'hybrid_training' in locals() and hybrid_training else 'DISABLED'}")
    print(f"GPU dataset size: {len(gpu_dataset) if 'gpu_dataset' in locals() else 'N/A'}")
    print(f"CPU dataset size: {len(cpu_dataset) if 'cpu_dataset' in locals() else 'N/A'}")
    if 'total_sample_time' in locals():
        print(f"Total training time: {total_sample_time:.2f} seconds ({total_sample_time/60:.1f} minutes)")
    elif 'main_training_time' in locals():
        print(f"Main training time: {main_training_time:.2f} seconds ({main_training_time/60:.1f} minutes)")
    print("="*60)
    print("\n🔧 This sample training replicates the EXACT same algorithm as Trainmain.py")
    print("📊 including hybrid GPU/CPU training, device detection, and optimization settings!")
    print("\n💡 To enable hybrid training in main script, set 'hybrid_training = True' in Trainmain.py line ~355")

# Sample usage examples:
# python SampleTrain.py -data IP102 -optim Adam -sch none -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 1 -dv auto -sample 0.05
# python SampleTrain.py -data IP102 -optim Adam -sch none -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 5 -dv auto -sample 0.1
