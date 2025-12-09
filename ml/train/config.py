"""Training configuration"""

CONFIG = {
    # Data
    "data_dir": "./data",
    "checkpoint_dir": "./checkpoints",
    "models_dir": "../models",  
    # Model
    "model": "efficientnet_b0",
    "image_size": 224,
    # Training
    "batch_size": 32,
<<<<<<< HEAD
    "epochs": 1,
=======
    "epochs": 20,
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    # Labels convention:
    # 0 = REAL, 1 = FAKE
    # Model output after sigmoid:
    #   > 0.5 = FAKE
    #   < 0.5 = REAL
    "label_convention": "fake_is_1"
}
