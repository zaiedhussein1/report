# NC-VLM: Neural Vision-Language Model

## Overview

This project aims to be a Vision-Language Model (VLM). The specific application and architecture are currently under development. It utilizes PyTorch for model building and training.

## Current Status

The project is in its initial development phase. Core components like the main execution script (`main.py`), configuration (`config.py`), a placeholder model (`models/model.py`), and a placeholder training loop (`training/train.py`) have been set up. Data preprocessing scripts are also present.

## Directory Structure

-   `data/`: Contains datasets (e.g., CSV files, images). Includes dummy data for initial testing.
-   `evaluation/`: Scripts for evaluating model performance. (Currently contains placeholder `evaluate.py` and `qualitative.py`)
-   `models/`: Contains model definitions (`model.py`) and saved model checkpoints (planned for `models/saved_models/`).
-   `preprocessing/`: Scripts for data loading, preprocessing (`prepare_data.py`), and text processing (`preprocess_text.py`).
-   `training/`: Contains the training script (`train.py`).
-   `utils/`: Utility functions (currently contains `utils.py`).

## Key Files

-   `main.py`: The main script to run the data preprocessing, model training, and evaluation pipeline.
-   `config.py`: Stores all configurations, including file paths, hyperparameters, and model settings.
-   `requirements.txt`: Lists the Python dependencies for this project.
-   `models/model.py`: Defines the neural network architecture for the VLM (currently a placeholder).
-   `training/train.py`: Implements the training loop for the model (currently a placeholder with dummy data).
-   `preprocessing/prepare_data.py`: Script to load, merge, filter, and preprocess raw data.
-   `preprocessing/preprocess_text.py`: Script for text tokenization and related text preprocessing tasks.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Note: `requirements.txt` specifies CPU-specific PyTorch versions. Adjust if you have a GPU.

## How to Run

Currently, the main script can be executed to see the placeholder pipeline in action:

```bash
python main.py
```

This will run the placeholder data loading, model initialization, and training loop (with dummy data). Configuration can be adjusted in `config.py`.
