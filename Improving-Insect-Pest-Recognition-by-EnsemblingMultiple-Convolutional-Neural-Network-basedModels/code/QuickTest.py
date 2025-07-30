"""Quick test script for immediate error detection"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import os
import time
import random

def quick_test(data_path, device_type='auto', batch_size=4, num_samples=50):
    """
    Quick test with minimal samples to detect errors immediately
    """
    print("🚀 Starting Quick Test...")
    print(f"Testing with {num_samples} samples, batch size {batch_size}")
    
    # Device detection
    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                print("Using DirectML")
            except:
                device = torch.device('cpu')
                print("Using CPU")
    else:
        device = torch.device(device_type)
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    try:
        dataset = ImageFolder(root=data_path, transform=transform)
        print(f"✅ Dataset loaded: {len(dataset)} total samples")
        
        # Create small subset
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        subset = Subset(dataset, indices)
        
        # Create dataloader
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"✅ DataLoader created: {len(subset)} samples, {len(dataloader)} batches")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Simple model
    try:
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 102)  # IP102 has 102 classes
        model = model.to(device)
        print(f"✅ Model loaded and moved to {device}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test forward pass
    try:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                print(f"Batch {batch_idx + 1}: Input shape: {inputs.shape}, Output shape: {outputs.shape}, Loss: {loss.item():.4f}")
                
                if batch_idx >= 2:  # Test only first 3 batches
                    break
        
        print("✅ Forward pass successful!")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test training step
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print(f"Training Batch {batch_idx + 1}: Loss: {loss.item():.4f}")
            
            if batch_idx >= 2:  # Test only first 3 batches
                break
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✅ Training step successful!")
        print(f"Training time for 3 batches: {training_time:.2f} seconds")
        
        # Estimate time for full dataset
        time_per_batch = training_time / 3
        total_batches = len(DataLoader(dataset, batch_size=batch_size))
        estimated_epoch_time = time_per_batch * total_batches
        
        print(f"📊 Time estimation:")
        print(f"   Time per batch: {time_per_batch:.2f} seconds")
        print(f"   Total batches (full dataset): {total_batches}")
        print(f"   Estimated time per epoch: {estimated_epoch_time:.2f} seconds ({estimated_epoch_time/60:.1f} minutes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

if __name__ == "__main__":
    # Test with IP102 dataset
    data_path = os.path.join(os.getcwd(), "IP102", "train")
    
    if os.path.exists(data_path):
        print(f"Testing with dataset: {data_path}")
        success = quick_test(data_path)
        
        if success:
            print("\n🎉 All tests passed! Your setup should work fine.")
            print("\nNow you can run:")
            print("python SampleTrain.py -data IP102 -optim Adam -sch none -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 1 -dv auto -sample 0.05")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print(f"❌ Dataset path not found: {data_path}")
        print("Please check if your dataset is in the correct location.")
