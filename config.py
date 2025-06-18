import torch

# Data paths
PROCESSED_CSV_PATH = "data/processed_reports_with_images.csv"
IMAGE_DIR = "data/dummy_images/"
# IMAGE_DIR = "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/"

# Model general config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
AMP_ENABLED = True # Enable Automatic Mixed Precision if DEVICE is 'cuda'

# Tokenizer specific
TOKENIZER_NAME = "dmis-lab/biobert-base-cased-v1.1"
VOCAB_SIZE_PLACEHOLDER = 28996
SOS_TOKEN_ID_PLACEHOLDER = 101
EOS_TOKEN_ID_PLACEHOLDER = 102
PAD_TOKEN_ID_PLACEHOLDER = 0

# Training hyperparameters
BATCH_SIZE = 2 # Adjusted for potentially small dummy dataset size
NUM_EPOCHS = 3 # Keep small for initial testing of the loop
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
TEACHER_FORCING_RATIO = 0.5
NUM_WORKERS = 0 # Set to 0 for easier debugging with dummy data, can increase later
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping
LOG_INTERVAL = 1 # Log every batch for dummy data testing

# Model dimensions
# Image Encoder
IMAGE_ENCODER_MODEL_NAME = 'microsoft/swin-tiny-patch4-window7-224'
IMAGE_ENCODER_PRETRAINED = True

# Text Embedder
TEXT_EMBEDDER_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
TEXT_EMBEDDER_FREEZE = True

# Decoder
ATTENTION_DIM = 256
DECODER_HIDDEN_DIM = 512
LSTM_DROPOUT_RATE = 0.5

# Logging & Checkpointing
MODEL_SAVE_DIR = "trained_models/"
LOG_DIR = "logs/"

# Scheduler params (example for StepLR)
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.1
