"""Training configuration"""

CONFIG = {
    "model": "efficientnet_b0",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 10,  # Increased from 1
    "learning_rate": 0.0001,
    "weight_decay": 0.00001,
    "num_workers": 4,
    "train_split": 0.8,
    "seed": 42,
    
    # Paths
    "data_dir": "./data",
    "models_dir": "../models",
    "checkpoint_dir": "./checkpoints",
    
    # Augmentation
    "augmentation": True,
    "mixup_alpha": 0.2,
}
