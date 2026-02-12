import torch
import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
from config import config
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': {
            'patch_size': config.PATCH_SIZE,
            'encoder_dim': config.ENCODER_DIM,
            'encoder_layers': config.ENCODER_LAYERS,
            'attention_heads': config.ATTENTION_HEADS,
            'decoder_dim': config.DECODER_DIM,
            'decoder_layers': config.DECODER_LAYERS,
            'total_channels': config.TOTAL_CHANNELS,
            'num_patches': config.NUM_PATCHES,
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = None
    scheduler = None
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss:.4f}")
    
    return model, optimizer, scheduler, start_epoch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_directory_structure():
    directories = [
        config.DATA_DIR,
        config.CHECKPOINT_DIR,
        config.RESULTS_DIR,
        config.LOG_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def print_model_summary(model):
    print("Model Summary")
    print(f"Total parameters: {count_parameters(model):,}")
    
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:20s} {str(module.__class__.__name__):30s} {num_params:12,}")
    
    print("=" * 80)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def save_training_history(history, path):

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    serializable_history = {}
    for key, value in history.items():
        serializable_history[key] = convert_to_serializable(value)
    
    with open(path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"Training history saved to {path}")