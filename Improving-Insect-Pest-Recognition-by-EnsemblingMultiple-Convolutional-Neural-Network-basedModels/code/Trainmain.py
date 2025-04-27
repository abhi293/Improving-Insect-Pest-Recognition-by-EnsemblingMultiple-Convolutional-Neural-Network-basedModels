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
from distutils.util import strtobool
import torch_directml
import torch.distributed as dist
from torch.amp import GradScaler, autocast
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
    # Detect device
    def get_device():
        if torch.cuda.is_available():
            print("Using CUDA (Dedicated GPU)")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # For Apple M1/M2 GPUs
            print("Using Metal Performance Shaders (MPS)")
            return torch.device("mps")
        try:
            import torch_directml
            dml_device = torch_directml.device()
            print("Using DirectML (Integrated GPU)")
            return dml_device
        except ImportError:
            print("DirectML is not available. Falling back to CPU.")
        return torch.device("cpu")
    
    device = get_device()
    batch_size = int(args['batch_size'])
    num_epochs = int(args['epochs'])

    # My setting, Hyperparameters
    init_lr = float(args['learning_rate'])
    weight_decay = float(args['weight_decay'])
    dropout = float(args['dropout'])
    optimizer = args['optimizer']
    scheduler = args['scheduler']
    use_pretrained = strtobool(args['use_pretrained'])
    model_name = args['model_name']
    dataset_name = args['dataset']
    n_classes = dataset_info[dataset_name]

    # Initialize the model
    num_classes = 102  # Replace with the actual number of classes in your dataset
    dropout = float(args['dropout'])  # Dropout rate from the command-line argument
    model_name = args['model_name']  # Correct key
    
    model, input_size = initialize_model(model_name=model_name, num_classes=n_classes, use_pretrained=use_pretrained, dropout=dropout)
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
    
    # Reduce input image size
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),  # Smaller image size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
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
    def get_dynamic_batch_size(device):
      """
      Dynamically calculate batch size based on available memory.
      """
      if device.type == "cuda":
          # For CUDA (Dedicated GPU)
          try:
              gpu_properties = torch.cuda.get_device_properties(device)
              total_memory = gpu_properties.total_memory  # Total GPU memory in bytes
              reserved_memory = torch.cuda.memory_reserved(device)  # Reserved memory
              available_memory = total_memory - reserved_memory
              # Set batch size based on available memory (e.g., 1 batch per 100MB)
              return max(1, available_memory // (100 * 1024 * 1024))  # Batch size per 100MB
          except AssertionError:
              print("CUDA is not available. Falling back to default batch size.")
              return 16  # Default batch size for CUDA fallback
      elif device.type == "dml":
          # For DirectML (Integrated GPU)
          print("Using DirectML. Setting a conservative batch size.")
          return 2  # Conservative batch size for DirectML
      elif device.type == "cpu":
          # For CPU
          import psutil
          available_memory = psutil.virtual_memory().available  # Available system memory in bytes
          # Set batch size based on available memory (e.g., 1 batch per 500MB)
          return max(1, available_memory // (500 * 1024 * 1024))  # Batch size per 500MB
      else:
          # Default batch size for unknown devices
          print("Unknown device type. Using default batch size.")
          return 8
    
    # Adjust learning rate dynamically
    def get_learning_rate(device):
        if device.type == "cuda":
            return 0.001  # Higher learning rate for GPU
        elif device.type == "cpu":
            return 0.0001  # Lower learning rate for CPU
        else:
            return 0.0005  # Default learning rate for other devices
    
        # Reduce batch size dynamically
    gpu_batch_size = 1  # Start with the smallest batch size
    cpu_batch_size = 1

    # Dynamically calculate batch sizes
    gpu_batch_size = get_dynamic_batch_size(torch.device("cuda"))
    cpu_batch_size = get_dynamic_batch_size(torch.device("cpu"))
    
    # Use CPU if DirectML fails
    device = get_device()

    # Initialize DataLoaders
    gpu_loader = DataLoader(gpu_dataset, batch_size=gpu_batch_size, shuffle=True, num_workers=0, pin_memory=False)
    cpu_loader = DataLoader(cpu_dataset, batch_size=cpu_batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    # Set learning rate
    learning_rate = get_learning_rate(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if is_train:
        train_set = DataLoader(train_set, batch_size= batch_size, shuffle= True, 
                            num_workers= 8, pin_memory=True)

        valid_set = DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
                                num_workers= 8, pin_memory= True)

        datasets_dict = {'train': train_set, 'val': valid_set}

        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()

        # Initialize AMP scaler
        scaler = GradScaler()
        accumulation_steps = 4  # Simulate a larger batch size by accumulating gradients
        
        criterion = nn.CrossEntropyLoss()


        # Training loop with tqdm progress bar
        for epoch in range(num_epochs):
            model.train()
            print(f"Epoch {epoch+1}/{num_epochs}")
            for i, (gpu_data, cpu_data) in enumerate(tqdm(zip(gpu_loader, cpu_loader), total=len(gpu_loader))):
                gpu_inputs, gpu_labels = gpu_data[0].to(device), gpu_data[1].to(device)
                cpu_inputs, cpu_labels = cpu_data[0].to(torch.device("cpu")), cpu_data[1].to(torch.device("cpu"))
        
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=device.type in ["cuda", "dml"]):
                    gpu_outputs = model(gpu_inputs)
                    gpu_loss = criterion(gpu_outputs, gpu_labels)
                    optimizer.step()
                scaler.scale(gpu_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Optionally, print loss for each batch
                print(f"Batch {i+1}/{len(gpu_loader)} - Loss: {gpu_loss.item():.4f}")


        if optimizer.lower() == 'sgd':
            optimizer_ft = optim.SGD(params_to_update, lr= init_lr, momentum= 0.9, weight_decay= weight_decay)
        elif optimizer.lower() == 'adam':
            optimizer_ft = optim.Adam(params_to_update, lr= init_lr, betas= (0.9, 0.999),
                                eps= 1e-08, weight_decay= weight_decay)

        if scheduler.lower() == 'none':
            scheduler_ft = None
        elif scheduler.lower() == 'expdecay':
            scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma= 0.96)
        elif scheduler.lower() == 'steplr':
            scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma= 0.09999, last_epoch=-1)
        elif scheduler.lower() == 'myscheduler':
            scheduler_ft = myScheduler(optimizer_ft, gamma= 0.09999)

        model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                        scheduler= scheduler_ft, num_epochs= num_epochs, checkpointFn= checkpoint
                                        , device= device, is_save_checkpoint= is_save_checkpoint
                                        ,is_load_checkpoint= load_checkpoint)

    if save_model_dict:
        torch.save(model_ft.state_dict(), path_weight_dict)

    if is_eval:
        test_set = ImageFolder(root= test_root_path, transform= data_transforms['val'])
        test_set = DataLoader(test_set, batch_size= batch_size, shuffle= False,
                            num_workers= 8, pin_memory= True)

        evaluate_model(model_ft, testloader= test_set, 
                        path_weight_dict= None, device= device, model_hyper= model_hyper)

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