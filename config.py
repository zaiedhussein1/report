# File Paths
DATA_DIR = 'data/'
MODEL_SAVE_DIR = 'models/saved_models/'
REPORTS_CSV_PATH = DATA_DIR + 'dummy_reports.csv'
PROJECTIONS_CSV_PATH = DATA_DIR + 'dummy_projections.csv'
PROCESSED_DATA_PATH = DATA_DIR + 'processed_reports_with_images.csv'

# Basic Training Settings
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Model Settings (placeholders)
MODEL_NAME = 'VLMModel'
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
VOCAB_SIZE = 10000  # Placeholder, will be determined by tokenizer
NUM_CLASSES = 1    # Placeholder, depends on the specific task
