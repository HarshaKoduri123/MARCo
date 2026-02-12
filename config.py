import torch
from pathlib import Path

class Config:
    # ------------------
    # Paths
    # ------------------
    DATA_FILE = Path("./dataset/DFC_preprocessed.pt")
    CHECKPOINT_DIR = Path("./checkpoints")
    RESULTS_DIR = Path("./results")
    LOG_DIR = Path("./logs")

 
    for dir_path in [CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ------------------
    # Model hyperparameters
    # ------------------
    PATCH_SIZE = 8
    ENCODER_DIM = 768
    ENCODER_LAYERS = 12
    ATTENTION_HEADS = 16
    DECODER_DIM = 512
    DECODER_LAYERS = 1
    TOTAL_CHANNELS = 14
    NUM_PATCHES = (96 // PATCH_SIZE) ** 2

    # ------------------
    # Training hyperparameters
    # ------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    EPOCHS = 100
    WARMUP_EPOCHS = 10
    MASK_RATIO = 0.75
    USE_8BIT = True

    CONTRAST_WEIGHT = 1.0
    MAE_WEIGHT = 1.0
    SSL_WEIGHT = 0.1 



    # ------------------
    # Device
    # ------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------
    # Distributed (DISABLED)
    # ------------------
    WORLD_SIZE = 1
    RANK = 0
    DISTRIBUTED = False

    # ------------------
    # Checkpointing
    # ------------------
    LOAD_CHECKPOINT = None
    SAVE_FREQUENCY = 10

    # ------------------
    # Logging
    # ------------------
    LOG_FREQUENCY = 100
    VISUALIZE_FREQUENCY = 500


config = Config()
