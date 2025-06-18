import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import sys

# Adjust path to import from parent directory's modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import RadiologyReportGenerator # Assuming all model components are accessible via this

def generate_report_greedy(model, frontal_image_batch, lateral_image_batch, tokenizer, max_len, device):
    """
    Generates a radiology report for a batch of images using greedy search.
    Args:
        model (RadiologyReportGenerator): The trained model.
        frontal_image_batch (torch.Tensor): Batch of frontal images (B, C, H, W).
        lateral_image_batch (torch.Tensor): Batch of lateral images (B, C, H, W).
        tokenizer (AutoTokenizer): Tokenizer for decoding.
        max_len (int): Maximum length of the generated report.
        device (torch.device): Device to perform computations on.
    Returns:
        decoded_texts (list of str): List of generated report texts for the batch.
        all_attention_weights (list of torch.Tensor): Placeholder for attention weights (not fully implemented in this pass).
    """
    model.eval()

    with torch.no_grad():
        # ImageEncoderSwin now returns (fused_patch_features, label_logits)
        # We only need fused_patch_features for generation
        fused_patch_features, _ = model.image_encoder(frontal_image_batch, lateral_image_batch) # (B, num_patches, encoder_dim)
        encoder_output_for_decoder = fused_patch_features # Use this variable for clarity

        batch_size = frontal_image_batch.size(0)

        # Initialize LSTM states
        h, c = model.decoder.init_hidden_state(batch_size, device)

        # Start with SOS token (e.g., CLS token for BERT-like models)
        # model.sos_token_id should be set during model initialization
        current_token_ids = torch.full((batch_size, 1), model.sos_token_id, dtype=torch.long).to(device)

        generated_sequences_ids = [current_token_ids[i].tolist() for i in range(batch_size)] # Store lists of token IDs for each sample

        # Placeholder for storing attention weights if needed later; for now, focus on text generation
        # all_attention_weights_batch = [] # To store attention for each step for the whole batch

        for _ in range(max_len -1): # -1 because we already have SOS
            # Get embedding for the current token(s)
            # Text embedder expects (B, SeqLen), output (B, SeqLen, EmbedDim)
            # For single token input (B, 1), output is (B, 1, EmbedDim)
            # An attention mask of all ones is appropriate for single, non-padded tokens.
            single_token_attention_mask = torch.ones_like(current_token_ids, device=device)
            current_embeddings = model.text_embedder(current_token_ids, attention_mask=single_token_attention_mask)

            # Decoder expects current_word_embedding as (B, EmbedDim)
            logits, h, c, attention_w = model.decoder(current_embeddings.squeeze(1), encoder_output_for_decoder, h, c)
            # logits shape: (B, vocab_size)
            # attention_w shape: (B, num_patches)

            # Store attention weights (optional for now)
            # all_attention_weights_batch.append(attention_w.cpu())

            # Get the predicted next token ids (greedy search)
            predicted_token_ids = logits.argmax(dim=-1).unsqueeze(1) # (B, 1)

            # Append predicted tokens to sequences for each sample in batch
            all_eos_generated = True
            for i in range(batch_size):
                # Only append if EOS not already generated for this sequence
                # Assuming tokenizer.eos_token_id is available
                if tokenizer.eos_token_id not in generated_sequences_ids[i]:
                    generated_sequences_ids[i].append(predicted_token_ids[i].item())
                    if predicted_token_ids[i].item() != tokenizer.eos_token_id:
                        all_eos_generated = False # At least one sequence is still ongoing
                # If EOS was already there, this sample is done, keep it as is for EOS check
                elif predicted_token_ids[i].item() != tokenizer.eos_token_id : # check if current prediction is not EOS after EOS already found
                    all_eos_generated = False


            current_token_ids = predicted_token_ids

            if all_eos_generated: # If all sequences in batch have generated EOS
                 break

        # Decode generated sequences
        decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_sequences_ids]

        all_attention_weights_placeholder = []

    return decoded_texts, all_attention_weights_placeholder


if __name__ == '__main__':
    print("Testing generate_report_greedy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_name = 'dmis-lab/biobert-base-cased-v1.1'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size_test = tokenizer.vocab_size
    sos_token_id_test = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
    pad_token_id_test = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # Ensure EOS token ID is available for early stopping logic in generation
    eos_token_id_test = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102
    if eos_token_id_test is None: raise ValueError("EOS token for tokenizer not found, needed for generation logic.")


    img_enc_kwargs_test = {'model_name': 'microsoft/swin-tiny-patch4-window7-224', 'pretrained': False}
    text_emb_kwargs_test = {'model_name': tokenizer_name, 'freeze_bert': True}
    dec_kwargs_test = {'attention_dim': 256, 'decoder_h_dim': 512, 'dropout_rate': 0.1}

    test_model = RadiologyReportGenerator(
        image_encoder_kwargs=img_enc_kwargs_test,
        text_embedder_kwargs=text_emb_kwargs_test,
        decoder_kwargs=dec_kwargs_test,
        tokenizer_vocab_size=vocab_size_test,
        sos_token_id=sos_token_id_test,
        pad_token_id=pad_token_id_test,
        device=device
    ).to(device)
    # Manually set eos_token_id on the model instance if your model needs it for generation logic,
    # or pass it to generation function if that's cleaner.
    # For this test, tokenizer.eos_token_id is used directly in generate_report_greedy.

    test_model.eval()

    batch_size_test = 2
    dummy_frontal_images_test = torch.randn(batch_size_test, 3, 224, 224).to(device)
    dummy_lateral_images_test = torch.randn(batch_size_test, 3, 224, 224).to(device)
    print(f"Dummy frontal images shape: {dummy_frontal_images_test.shape}")

    max_generation_len = 20
    print(f"Generating reports with max_len={max_generation_len}...")

    try:
        generated_texts, _ = generate_report_greedy(
            test_model,
            dummy_frontal_images_test,
            dummy_lateral_images_test,
            tokenizer,
            max_generation_len,
            device
        )

        print("\nGenerated Reports (Greedy Search):")
        for i, text in enumerate(generated_texts):
            print(f"Sample {i+1}: {text}")

        print("\nTest for generate_report_greedy completed.")
        assert len(generated_texts) == batch_size_test, "Number of generated texts should match batch size."

    except Exception as e:
        print(f"Error during generate_report_greedy test: {e}")
        import traceback
        traceback.print_exc()
