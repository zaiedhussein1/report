import torch
import torch.nn as nn
import torch.nn.functional as F

class VLMModel(nn.Module):
    def __init__(self, config):
        """
        Initializes the VLMModel.

        Args:
            config: A configuration object with model parameters.
                    Expected attributes:
                        VOCAB_SIZE (int): The size of the vocabulary.
                        EMBEDDING_DIM (int): The dimension of the text embeddings.
                        NUM_CLASSES (int): The number of output classes.
        """
        super(VLMModel, self).__init__()
        self.config = config

        # Text processing layers
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        self.fc = nn.Linear(config.EMBEDDING_DIM, config.NUM_CLASSES)

        # Placeholder for image features (optional, can be expanded later)
        # if hasattr(config, 'IMAGE_FEATURE_DIM') and config.IMAGE_FEATURE_DIM is not None:
        #     self.image_fc = nn.Linear(config.IMAGE_FEATURE_DIM, config.HIDDEN_DIM)
        # else:
        #     # Using a default placeholder if IMAGE_FEATURE_DIM is not in config
        #     # This part can be made more robust or configurable
        #     print("Warning: IMAGE_FEATURE_DIM not found in config. Skipping image_fc layer or using a default.")
        #     # self.image_fc = nn.Linear(2048, config.HIDDEN_DIM) # Example default

    def forward(self, text_input_ids):
        """
        Forward pass of the model.

        Args:
            text_input_ids (torch.Tensor): A batch of token IDs.
                                           Shape: [batch_size, seq_len]

        Returns:
            torch.Tensor: The model output.
                          Shape: [batch_size, num_classes]
        """
        # Embed text input
        # Shape: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded_text = self.embedding(text_input_ids)

        # Pool text features (simple averaging over sequence length)
        # Shape: [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim]
        pooled_text = embedded_text.mean(dim=1)

        # Final classification layer
        # Shape: [batch_size, embedding_dim] -> [batch_size, num_classes]
        output = self.fc(pooled_text)

        return output

if __name__ == '__main__':
    # Example Usage (requires a dummy config)
    class DummyConfig:
        VOCAB_SIZE = 1000
        EMBEDDING_DIM = 128
        NUM_CLASSES = 10
        # IMAGE_FEATURE_DIM = 2048 # Optional: for testing image part
        # HIDDEN_DIM = 256         # Optional: for testing image part

    dummy_config = DummyConfig()
    model = VLMModel(dummy_config)
    print("VLMModel initialized successfully with dummy config.")

    # Create dummy input
    batch_size = 4
    seq_len = 20
    dummy_text_input = torch.randint(0, dummy_config.VOCAB_SIZE, (batch_size, seq_len))
    print(f"Dummy text input shape: {dummy_text_input.shape}")

    # Perform a forward pass
    try:
        output = model(dummy_text_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, dummy_config.NUM_CLASSES)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
