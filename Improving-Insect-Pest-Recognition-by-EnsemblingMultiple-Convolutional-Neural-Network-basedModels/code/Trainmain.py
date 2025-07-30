"""This file is the main training and testing file, except for Fine-Grained model"""

from utils.myFunctions import train_model, evaluate_model, initialize_model, myScheduler
import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
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
ap.add_argument("-data", "--dataset", required=True, help= "Choose dataset: IP102 or D0")
ap.add_argument("-optim", "--optimizer", required=True, help= "Optimizer function : Only SGD or Adam")
ap.add_argument("-sch", "--scheduler", required=True, help= "Scheduler:\nnone\nsteplr\nexpdecay\nmyscheduler")
ap.add_argument("-l2", "--weight_decay", required= True, help= "L2 regularzation")
ap.add_argument("-do", "--dropout", required= True, help= "Dropout rate")
ap.add_argument("-predt", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-mn", "--model_name", required=True, help="Model name: resnet, residual-attention, fpn")  # Keep this
ap.add_argument("-lr", "--learning_rate", required= True, help= "Inital learning rate")
ap.add_argument("-bz", "--batch_size", required= True, help= "Batch size")
ap.add_argument("-ep", "--epochs", required= True, help= "Number of Epochs")
ap.add_argument("-dv", "--device", required= True, help= "Device type")

ap.add_argument("-istra", "--istrain", required= False, help= "Train mode", default='True')
ap.add_argument("-iseva", "--iseval", required= False, help= "Eval mode", default='True')
ap.add_argument("-issavck", "--issavechkp", required= False, help= "Save checkpoint", default='True')
ap.add_argument("-issavmd", "--issavemodel", required= False, help= "Save model", default='True')
ap.add_argument("-isloadck", "--isloadchkp", required= False, help= "Load checkpoint", default='False')
ap.add_argument("-isloadmd", "--isloadmodel", required= False, help= "Load model", default='False')

args = vars(ap.parse_args())

dataset_info = {
    'IP102' : 102,
    'D0' : 40
}

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
    
    batch_size = int(args['batch_size'])
    num_epochs = int(args['epochs'])

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


    exp_name = 'insect_recognition'
    is_train = strtobool(args['istrain']) # ENABLE TRAIN
    is_eval = strtobool(args['iseval']) # ENABLE EVAL
    save_model_dict = strtobool(args['issavemodel']) # SAVE MODEL
    is_save_checkpoint = strtobool(args['issavechkp']) # SAVE CHECKPOINT
    load_model = strtobool(args['isloadmodel']) # MUST HAVE MODEL FIRST
    load_checkpoint = strtobool(args['isloadchkp']) # MUST HAVE CHECKPOINT MODEL FIRST
    checkpoint = 'checkpoint_' + dataset_name + '_' + model_name + '.pt'
    ############################################################

    if is_train == False and load_checkpoint == True:
        raise Exception('Error, checkpoint can be load during training')

    if load_model and load_checkpoint:
        raise Exception('Error, conflict between checkpoint and model weight')

    if dataset_name == 'D0':
        dataset_name = 'unzip_D0'

    path_weight_dict = os.path.join(os.getcwd(), 'pre-trained', model_name + '_' + dataset_name + '.pt')

    model_hyper = {
        'batch_size' : batch_size,
        'num_epochs' : num_epochs,
        'init_lr' : init_lr,
        'weight_decay' : weight_decay,
        'dropout' : dropout,
        'model_name' : model_name,
        'exp_name' : exp_name,
        'dataset' : dataset_name
    }

    train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
    valid_root_path = os.path.join(os.getcwd(), dataset_name, 'valid')
    test_root_path = os.path.join(os.getcwd(), dataset_name, 'test')

    model_ft, input_size = initialize_model(model_name, n_classes, dropout= dropout, use_pretrained= use_pretrained)

    if load_model:
        model_ft.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))
    
    # Get optimal image size based on device capabilities and model complexity
    optimal_image_size = get_optimal_image_size(device, 'medium')
    print(f"Using image size: {optimal_image_size}x{optimal_image_size}")
    
    # Optimized data transforms with better augmentation for training
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
    
    train_set = ImageFolder(root= train_root_path, transform= data_transforms['train']) 
    valid_set = ImageFolder(root= valid_root_path, transform= data_transforms['val'])

    dataset = train_set  # Use train_set for splitting

        # Split dataset for hybrid training
    def split_dataset(dataset, cpu_ratio=0.3):
        cpu_size = int(len(dataset) * cpu_ratio)
        gpu_size = len(dataset) - cpu_size
        return random_split(dataset, [gpu_size, cpu_size])
    
    # Split the dataset
    gpu_dataset, cpu_dataset = split_dataset(dataset)

    # Adjust batch size dynamically
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
    
    # Adjust learning rate dynamically based on device and batch size
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
    
    # Calculate optimal batch sizes using performance settings
    optimal_batch_size = max(1, int(batch_size * settings['batch_size_multiplier']))
    gpu_batch_size = get_dynamic_batch_size(torch.device("cuda"), optimal_batch_size) if torch.cuda.is_available() else optimal_batch_size
    cpu_batch_size = max(1, min(optimal_batch_size // 2, 4))  # Conservative CPU batch size
    
    print(f"Batch sizes - Optimal: {optimal_batch_size}, GPU: {gpu_batch_size}, CPU: {cpu_batch_size}")
    
    # Use the optimal batch size for the current device
    current_batch_size = optimal_batch_size
    
    # Initialize DataLoaders with performance-optimized settings
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
    
    # Set learning rate based on device and batch size
    dynamic_learning_rate = get_learning_rate(device, current_batch_size, init_lr)
    print(f"Using learning rate: {dynamic_learning_rate}")
    
    # Create a temporary optimizer for the hybrid training loop (this will be replaced later)
    temp_optimizer = torch.optim.Adam(model.parameters(), lr=dynamic_learning_rate)

    if is_train:
        # Create proper DataLoaders for training with optimized settings
        train_loader = DataLoader(train_set, batch_size=current_batch_size, shuffle=True, **dataloader_kwargs)
        valid_loader = DataLoader(valid_set, batch_size=current_batch_size, shuffle=False, **dataloader_kwargs)

        datasets_dict = {'train': train_loader, 'val': valid_loader}

        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()

        # Create the main optimizer based on command line arguments
        if optimizer_name.lower() == 'sgd':
            optimizer_ft = optim.SGD(params_to_update, lr=dynamic_learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer_ft = optim.Adam(params_to_update, lr=dynamic_learning_rate, betas=(0.9, 0.999),
                                eps=1e-08, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Create scheduler
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

        # Initialize AMP scaler for mixed precision training
        scaler = GradScaler(enabled=settings['mixed_precision'] and device.type in ["cuda", "dml"])
        criterion = nn.CrossEntropyLoss()
        
        # Memory management before training
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Optional: Hybrid training loop (experimental feature)
        # This can be enabled for multi-device training scenarios
        hybrid_training = True  # Set to False to disable hybrid GPU/CPU training
        
        if hybrid_training and torch.cuda.is_available():
            print("Starting hybrid GPU/CPU training...")
            accumulation_steps = 4
            
            # Training loop with tqdm progress bar
            for epoch in range(min(2, num_epochs)):  # Limit hybrid training to 2 epochs
                model.train()
                print(f"Hybrid Epoch {epoch+1}/{min(2, num_epochs)}")
                
                min_batches = min(len(gpu_loader), len(cpu_loader))
                for i, (gpu_data, cpu_data) in enumerate(tqdm(zip(gpu_loader, cpu_loader), total=min_batches)):
                    if i >= min_batches:
                        break
                        
                    gpu_inputs, gpu_labels = gpu_data[0].to(device), gpu_data[1].to(device)
                    
                    temp_optimizer.zero_grad()
                    with autocast(device_type=device.type, enabled=settings['mixed_precision'] and device.type in ["cuda", "dml"]):
                        gpu_outputs = model(gpu_inputs)
                        gpu_loss = criterion(gpu_outputs, gpu_labels)
                    
                    scaler.scale(gpu_loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(temp_optimizer)
                        scaler.update()
                        temp_optimizer.zero_grad()

                    if i % 10 == 0:  # Print every 10 batches
                        print(f"Batch {i+1}/{min_batches} - Loss: {gpu_loss.item():.4f}")
                
                # Memory cleanup after each epoch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        # Main training using the standard train_model function
        print("Starting main training...")
        model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                    scheduler=scheduler_ft, num_epochs=num_epochs, checkpointFn=checkpoint,
                                    device=device, is_save_checkpoint=is_save_checkpoint,
                                    is_load_checkpoint=load_checkpoint)

    if save_model_dict:
        torch.save(model_ft.state_dict(), path_weight_dict)

    if is_eval:
        test_set = ImageFolder(root=test_root_path, transform=data_transforms['val'])
        test_loader = DataLoader(test_set, batch_size=current_batch_size, shuffle=False, **dataloader_kwargs)

        evaluate_model(model_ft, testloader=test_loader, 
                        path_weight_dict=None, device=device, model_hyper=model_hyper)

    if is_train:
        train_acc = [h.cpu().numpy() for h in hist['train_acc']]
        val_acc = [h.cpu().numpy() for h in hist['val_acc']]

        fig = plt.figure()
        path_folder = os.path.join(os.getcwd(), exp_name, model_name + '_' + dataset_name + '_torch')
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        plt.subplot(2, 1, 1)
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,len(train_acc) + 1), train_acc, label= "Train")
        plt.plot(range(1,len(val_acc) + 1), val_acc, label= "Val")
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, len(train_acc) + 1, 1.0))
        plt.legend()

        train_loss = hist['train_loss']
        val_loss = hist['val_loss']

        plt.subplot(2, 1, 2)
        plt.title("Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,len(train_acc) + 1), train_loss, label= "Train")
        plt.plot(range(1,len(val_loss) + 1), val_loss, label= "Val")
        plt.xticks(np.arange(1, len(train_acc) + 1, 1.0))
        plt.legend()
        plt.savefig(os.path.join(path_folder, model_name + '.png'))
        plt.show()

        ran_eps = len(val_acc)

        with open(os.path.join(path_folder, (model_name + '.txt')), 'w') as f:
            f.write('Number epochs : %d\n' %(ran_eps))
            f.write('Learning rate : %f\n' %(init_lr))
            f.write('L2 regularization lambda: %f\n' %(weight_decay))
            for i in range(ran_eps):
                f.write('Epoch %d :' %(i + 1))
                f.write('Train acc : %f, Train loss : %f, Val acc : %f, Val loss : %f\n'
                        %(train_acc[i], train_loss[i], val_acc[i], val_loss[i]))

#How to run the code?
#python Trainmain.py -data IP102 -optim SGD -sch none -l2 0.01 -do 0.5 -predt True -mn resnet -lr 0.001 -bz 4 -ep 10 -dv auto
#python Trainmain.py -data IP102 -optim Adam -sch steplr -l2 0.01 -do 0.5 -predt True -mn resnet -lr 0.001 -bz 16 -ep 50 -dv auto