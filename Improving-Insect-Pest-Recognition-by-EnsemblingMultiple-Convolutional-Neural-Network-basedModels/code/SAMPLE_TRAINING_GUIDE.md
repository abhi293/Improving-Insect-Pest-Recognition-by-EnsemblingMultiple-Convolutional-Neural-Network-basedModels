# Sample Training Guide

## Problem Solved
✅ **Fixed DirectML Float64 Error**: Changed `.double()` to `.float()` in training function  
✅ **Created Sample Training Scripts**: Test with small dataset portions to catch errors early
✅ **Replicated EXACT Algorithm**: SampleTrain.py now mirrors Trainmain.py completely
✅ **Enabled Hybrid Training**: Tests both GPU and CPU training simultaneously

## Testing Strategy

### 1. 🚀 Quick Test (1-2 minutes)
**Purpose**: Immediate error detection  
**File**: `QuickTest.py`  
**Usage**: 
```bash
python QuickTest.py
```

### 2. 📊 Sample Training with Hybrid Mode (5-30 minutes)
**Purpose**: Comprehensive testing with EXACT same algorithm as original  
**File**: `SampleTrain.py` ⭐ **NOW WITH HYBRID TRAINING**  
**Usage**:
```bash
# Test with 5% of dataset - HYBRID ENABLED
python SampleTrain.py -data IP102 -optim Adam -sch none -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 1 -dv auto -sample 0.05

# Test with 10% of dataset for 5 epochs - HYBRID ENABLED
python SampleTrain.py -data IP102 -optim Adam -sch none -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 5 -dv auto -sample 0.1
```

### 3. 🎯 Full Training
**Purpose**: Production training  
**File**: `Trainmain.py` (fixed version)  
**Usage**:
```bash
python Trainmain.py -data IP102 -optim Adam -sch steplr -l2 0.01 -do 0.2 -predt True -mn resnet -lr 0.001 -bz 8 -ep 50 -dv auto
```

## 🔄 Hybrid Training Features

### What is Hybrid Training?
- **Simultaneous GPU + CPU**: Uses both GPU and CPU for training
- **Dataset Splitting**: 70% GPU, 30% CPU by default
- **Optimized Scheduling**: Processes batches from both devices
- **Memory Efficient**: Reduces GPU memory pressure

### SampleTrain.py Hybrid Features:
✅ **Always Enabled**: Hybrid training enabled by default for testing  
✅ **Exact Replication**: Same algorithm as Trainmain.py  
✅ **Device Detection**: Auto-detects CUDA, DirectML, or CPU  
✅ **Dynamic Batch Sizing**: Different batch sizes for GPU/CPU  
✅ **Performance Monitoring**: Tracks both GPU and CPU performance  
✅ **Time Estimation**: Estimates hybrid training time for full dataset  

### Trainmain.py Hybrid Control:
- **Default**: Hybrid training DISABLED (hybrid_training = False)
- **To Enable**: Use `HybridConfig.py` or manually set `hybrid_training = True`

## 🛠️ Hybrid Training Control

### Enable/Disable Hybrid in Main Script:
```bash
python HybridConfig.py
```

### Manual Control:
Edit `Trainmain.py` line ~355:
```python
# Enable hybrid training
hybrid_training = True  # Set to False to disable

# Disable hybrid training (default)
hybrid_training = False  # Set to True to enable
```

## Sample Training Parameters

### New Parameters in SampleTrain.py:
- **`-sample`**: Dataset percentage (0.05 = 5%, 0.1 = 10%)
- **`-seed`**: Random seed for reproducible sampling (default: 42)

### Recommended Sample Ratios:
- **0.01 (1%)**: Ultra-quick test (~1-5 minutes)
- **0.05 (5%)**: Quick validation (~5-15 minutes)  
- **0.1 (10%)**: Thorough test (~10-30 minutes)
- **0.2 (20%)**: Near-production test (~20-60 minutes)

## Features of Sample Training

### 🎯 Balanced Sampling
- Samples proportionally from each class
- Maintains class distribution
- Reproducible with random seed

### 📊 Time Estimation
- Calculates actual training time
- Estimates full dataset training time
- Projects total time for multiple epochs
- **NEW**: Separate estimates for hybrid and main training

### 🛡️ Error Detection
- Tests complete training pipeline
- Catches memory issues early
- Validates model compatibility
- **NEW**: Tests hybrid device compatibility

### ⚡ Performance Optimized
- Uses same optimizations as main script
- Device-specific settings
- Memory management
- **NEW**: Hybrid GPU/CPU utilization

## Example Output (NEW - With Hybrid Training)

```
Original dataset size: 45095
Sample dataset size: 2255 (5.0%)
Classes represented: 102
Using DirectML (Integrated GPU)
Device: privateuseone:0
Using image size: 128x128
Batch sizes - Optimal: 4, GPU: 4, CPU: 2

🔄 Hybrid training mode: ENABLED
📊 GPU dataset size: 1578, CPU dataset size: 677

🚀 Starting hybrid GPU/CPU training...
Hybrid Epoch 1/1: 100%|████████| 157/157 [00:45<00:00, 3.47it/s]
✅ Hybrid training completed in 45.23 seconds
📊 Estimated hybrid training time for full dataset: 904.6 seconds (15.1 minutes)

🎯 Starting main training...
Epoch 0/0: 100%|████████| 282/282 [01:30<00:00, 3.12it/s]
✅ Sample training completed successfully!

Main training time: 90.45 seconds (1.5 minutes)
Hybrid training time: 45.23 seconds (0.8 minutes)
Total sample training time: 135.68 seconds (2.3 minutes)

============================================================
📊 TRAINING TIME ESTIMATION
============================================================
Sample training time: 135.68 seconds
Sample ratio: 5.0%
Estimated time per epoch (full dataset): 2713.6 seconds (45.2 minutes)
Estimated total training time (50 epochs): 135680.0 seconds (2261.3 minutes, 37.7 hours)
============================================================

🎯 SAMPLE TRAINING SUMMARY
============================================================
Dataset: IP102
Model: resnet
Sample ratio: 5.0%
Device: privateuseone:0
Batch size: 4
GPU batch size: 4, CPU batch size: 2
Epochs: 1
Hybrid training: ENABLED
GPU dataset size: 1578
CPU dataset size: 677
Total training time: 135.68 seconds (2.3 minutes)
============================================================

🔧 This sample training replicates the EXACT same algorithm as Trainmain.py
📊 including hybrid GPU/CPU training, device detection, and optimization settings!

💡 To enable hybrid training in main script, set 'hybrid_training = True' in Trainmain.py line ~355
```

## Troubleshooting

### If QuickTest.py fails:
1. Check dataset path
2. Verify PyTorch installation
3. Test device compatibility

### If SampleTrain.py fails:
1. Try smaller sample ratio (0.01)
2. Reduce batch size
3. Use CPU device (`-dv cpu`)
4. **NEW**: Check hybrid training logs for device conflicts

### If hybrid training fails:
1. Ensure both GPU and CPU are available
2. Check memory constraints
3. Try smaller batch sizes
4. Disable hybrid training for testing

### If estimation seems too long:
1. Increase batch size if memory allows
2. Use CUDA if available
3. Consider reducing epochs
4. Try different model (`-mn resnet` is fastest)
5. **NEW**: Compare hybrid vs standard training times

## Recommended Testing Flow

1. **Start with QuickTest**: `python QuickTest.py`
2. **Run 1% sample**: `python SampleTrain.py ... -sample 0.01 -ep 1`
3. **Run 5% sample with hybrid**: `python SampleTrain.py ... -sample 0.05 -ep 3`  
4. **Analyze time estimates** (both hybrid and standard)
5. **Enable hybrid in main script** if beneficial: `python HybridConfig.py`
6. **Run full training** if estimates are acceptable

## 🚀 NEW: Algorithm Replication Features

### Exact Replication of Trainmain.py:
✅ **Device Detection**: Same device detection logic  
✅ **Performance Settings**: Identical optimization settings  
✅ **Batch Size Calculation**: Same dynamic batch sizing  
✅ **Learning Rate Scaling**: Same learning rate adjustments  
✅ **Dataset Splitting**: Same 70/30 GPU/CPU split  
✅ **Data Loading**: Same DataLoader configurations  
✅ **Optimizer Setup**: Same optimizer and scheduler logic  
✅ **Mixed Precision**: Same AMP settings  
✅ **Memory Management**: Same cleanup procedures  
✅ **Hybrid Training**: Same algorithm, enabled for testing  

### Differences from Original:
- ✅ **Hybrid Training**: ENABLED by default (vs DISABLED in original)
- ✅ **Sample Datasets**: Uses subset of original data
- ✅ **Time Estimation**: Adds comprehensive time analysis
- ✅ **Enhanced Logging**: More detailed progress information

This approach will help you test the EXACT same algorithm with a smaller dataset, catch errors early, and make informed decisions about training time and hybrid mode effectiveness!
