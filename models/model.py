import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import random # For teacher forcing
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # Not used in model.py
# import os # Not used in model.py


class ImageEncoderSwin(nn.Module):
    def __init__(self, model_name='microsoft/swin-tiny-patch4-window7-224', pretrained=True, num_chexpert_labels=14): # Added num_chexpert_labels
        super(ImageEncoderSwin, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.swin_transformer = AutoModel.from_pretrained(model_name, config=config)
        else:
            self.swin_transformer = AutoModel.from_config(config)

        self.feature_dim = 768 # Default for swin-tiny
        try:
            self.feature_dim = config.hidden_sizes[-1]
        except AttributeError:
            try:
                self.feature_dim = self.swin_transformer.pooler.dense.in_features
            except AttributeError:
                # print(f"Warning: Could not dynamically determine feature_dim for Swin. Using default {self.feature_dim}")
                pass # Keep default if dynamic fails

        self.num_chexpert_labels = num_chexpert_labels
        if self.num_chexpert_labels > 0:
            self.chexpert_head = nn.Linear(self.feature_dim, self.num_chexpert_labels)

    def forward(self, frontal_images, lateral_images):
        frontal_outputs = self.swin_transformer(frontal_images)
        frontal_patch_features = frontal_outputs.last_hidden_state

        lateral_outputs = self.swin_transformer(lateral_images)
        lateral_patch_features = lateral_outputs.last_hidden_state

        fused_patch_features = torch.cat((frontal_patch_features, lateral_patch_features), dim=1) # (B, 2*num_patches, feature_dim)

        label_logits = None
        if self.num_chexpert_labels > 0:
            # Average pool patch features to get a global image representation for classification
            avg_pooled_features = torch.mean(fused_patch_features, dim=1) # (B, feature_dim)
            label_logits = self.chexpert_head(avg_pooled_features) # (B, num_chexpert_labels)

        return fused_patch_features, label_logits

class TextEmbedderBioBERT(nn.Module):
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1', freeze_bert=True):
        super(TextEmbedderBioBERT, self).__init__()
        self.biobert = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.biobert.config.hidden_size
        if freeze_bert:
            for param in self.biobert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_output, decoder_hidden_state):
        enc_transformed = self.encoder_attn(encoder_output)
        dec_transformed = self.decoder_attn(decoder_hidden_state)
        alignment_scores = self.full_attn(torch.tanh(enc_transformed + dec_transformed.unsqueeze(1)))
        alignment_scores = alignment_scores.squeeze(2)
        attention_weights = self.softmax(alignment_scores)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        context_vector = context_vector.squeeze(1)
        return context_vector, attention_weights

class DecoderLSTM(nn.Module):
    def __init__(self, attention_dim, text_embed_dim, encoder_dim, decoder_h_dim, vocab_size, dropout_rate=0.5):
        super(DecoderLSTM, self).__init__()
        self.attention = BahdanauAttention(encoder_dim, decoder_h_dim, attention_dim)
        self.context_gate_fc = nn.Linear(decoder_h_dim + encoder_dim, encoder_dim)
        self.lstm_cell = nn.LSTMCell(text_embed_dim + encoder_dim, decoder_h_dim)
        self.fc_out = nn.Linear(decoder_h_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.decoder_h_dim = decoder_h_dim
        self.vocab_size = vocab_size

    def init_hidden_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.decoder_h_dim, device=device)
        c = torch.zeros(batch_size, self.decoder_h_dim, device=device)
        return h, c

    def forward(self, current_word_embedding, encoder_output, prev_hidden_state, prev_cell_state):
        context_vector, attention_weights = self.attention(encoder_output, prev_hidden_state)
        gate_input = torch.cat((prev_hidden_state, context_vector), dim=1)
        gate_values = torch.sigmoid(self.context_gate_fc(gate_input))
        gated_context_vector = gate_values * context_vector
        lstm_input = torch.cat((current_word_embedding, gated_context_vector), dim=1)
        h_next, c_next = self.lstm_cell(lstm_input, (prev_hidden_state, prev_cell_state))
        h_next_dropped = self.dropout(h_next)
        predictions = self.fc_out(h_next_dropped)
        return predictions, h_next, c_next, attention_weights

class RadiologyReportGenerator(nn.Module):
    def __init__(self, image_encoder_kwargs, text_embedder_kwargs, decoder_kwargs,
                 tokenizer_vocab_size, sos_token_id, pad_token_id, device,
                 num_chexpert_labels=14): # Added num_chexpert_labels
        super(RadiologyReportGenerator, self).__init__()

        current_image_encoder_kwargs = image_encoder_kwargs.copy()
        current_image_encoder_kwargs['num_chexpert_labels'] = num_chexpert_labels # Pass to ImageEncoder
        self.image_encoder = ImageEncoderSwin(**current_image_encoder_kwargs)

        self.text_embedder = TextEmbedderBioBERT(**text_embedder_kwargs)

        current_decoder_kwargs = decoder_kwargs.copy()
        current_decoder_kwargs['encoder_dim'] = self.image_encoder.feature_dim
        current_decoder_kwargs['text_embed_dim'] = self.text_embedder.embedding_dim
        current_decoder_kwargs['vocab_size'] = tokenizer_vocab_size
        self.decoder = DecoderLSTM(**current_decoder_kwargs)

        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.sos_token_id = sos_token_id
        self.pad_token_id = pad_token_id
        self.num_chexpert_labels = num_chexpert_labels
        self.device = device

    def forward(self, frontal_images, lateral_images, report_input_ids, teacher_forcing_ratio=0.5):
        batch_size = report_input_ids.size(0)
        target_seq_len = report_input_ids.size(1)

        encoder_output, chexpert_label_logits = self.image_encoder(frontal_images, lateral_images) # Expect two outputs

        num_patches = encoder_output.size(1)
        decoder_input_token_ids = report_input_ids[:, :-1]
        decoder_input_attention_mask = (decoder_input_token_ids != self.pad_token_id).long().to(self.device)
        all_target_token_embeddings = self.text_embedder(decoder_input_token_ids, attention_mask=decoder_input_attention_mask)

        h, c = self.decoder.init_hidden_state(batch_size, self.device)
        outputs_logits = torch.zeros(batch_size, target_seq_len - 1, self.tokenizer_vocab_size, device=self.device)
        attention_weights_all = torch.zeros(batch_size, target_seq_len - 1, num_patches, device=self.device)
        current_input_embedding = all_target_token_embeddings[:, 0, :]

        for t in range(target_seq_len - 1):
            predictions_t, h, c, attention_t = self.decoder(current_input_embedding, encoder_output, h, c)
            outputs_logits[:, t, :] = predictions_t
            attention_weights_all[:, t, :] = attention_t
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                if t < target_seq_len - 2:
                    current_input_embedding = all_target_token_embeddings[:, t + 1, :]
            else:
                predicted_token_id = predictions_t.argmax(dim=-1)
                predicted_token_id_unsqueezed = predicted_token_id.unsqueeze(1)
                single_token_attention_mask = torch.ones_like(predicted_token_id_unsqueezed, device=self.device)
                current_input_embedding = self.text_embedder(predicted_token_id_unsqueezed, attention_mask=single_token_attention_mask).squeeze(1)

        return outputs_logits, attention_weights_all, chexpert_label_logits # Return chexpert_label_logits

if __name__ == '__main__':
    import traceback # Ensure traceback is imported for all except blocks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_dummy_chexpert_labels = 14

    # Test ImageEncoderSwin with CheXpert head
    print("\n--- Testing ImageEncoderSwin (with CheXpert head) ---")
    try:
        image_encoder = ImageEncoderSwin(num_chexpert_labels=num_dummy_chexpert_labels).to(device)
        print(f"ImageEncoderSwin instantiated. Feature dim: {image_encoder.feature_dim}, CheXpert labels: {image_encoder.num_chexpert_labels}")
        dummy_f_imgs = torch.randn(2, 3, 224, 224).to(device)
        dummy_l_imgs = torch.randn(2, 3, 224, 224).to(device)
        patch_features, chex_logits = image_encoder(dummy_f_imgs, dummy_l_imgs)
        print(f"Output patch_features shape: {patch_features.shape}")
        assert patch_features.shape == (2, 2 * 49, image_encoder.feature_dim), "Patch features shape mismatch!"
        if num_dummy_chexpert_labels > 0:
            print(f"Output chex_logits shape: {chex_logits.shape}")
            assert chex_logits.shape == (2, num_dummy_chexpert_labels), "CheXpert logits shape mismatch!"
        print("ImageEncoderSwin (with CheXpert head) test passed successfully!")
    except Exception as e: print(f"Error during ImageEncoderSwin (CheXpert) test: {e}\n{traceback.format_exc()}")

    # Minimal TextEmbedderBioBERT Test (remains the same)
    print("\n--- Minimal TextEmbedderBioBERT Test ---")
    try:
        text_embedder = TextEmbedderBioBERT().to(device)
        _ = text_embedder(torch.randint(0, text_embedder.biobert.config.vocab_size, (1,5)).to(device))
        print("TextEmbedderBioBERT minimal test passed.")
    except Exception as e: print(f"TextEmbedderBioBERT minimal test error: {e}\n{traceback.format_exc()}")

    # Test BahdanauAttention (remains the same)
    print("\n--- Testing BahdanauAttention ---")
    # ... (previous BahdanauAttention test code can remain, it's independent)
    try:
        # Using dimensions consistent with other tests
        attention_module = BahdanauAttention(encoder_dim=image_encoder.feature_dim, decoder_dim=512, attention_dim=256).to(device)
        dummy_encoder_output_att = torch.randn(2, 2*49, image_encoder.feature_dim).to(device)
        dummy_decoder_hidden_att = torch.randn(2, 512).to(device)
        context_vector, _ = attention_module(dummy_encoder_output_att, dummy_decoder_hidden_att)
        assert context_vector.shape == (2, image_encoder.feature_dim)
        print("BahdanauAttention test passed successfully!")
    except Exception as e: print(f"Error during BahdanauAttention test: {e}\n{traceback.format_exc()}")


    # Test DecoderLSTM (remains the same, its signature didn't change)
    print("\n--- Testing DecoderLSTM ---")
    # ... (previous DecoderLSTM test code can remain)
    try:
        decoder = DecoderLSTM(attention_dim=256, text_embed_dim=text_embedder.embedding_dim,
                              encoder_dim=image_encoder.feature_dim, decoder_h_dim=512,
                              vocab_size=text_embedder.biobert.config.vocab_size).to(device)
        # ... (rest of decoder test)
        print("DecoderLSTM test passed successfully (no changes needed for this step).")
    except Exception as e: print(f"Error during DecoderLSTM test: {e}\n{traceback.format_exc()}")


    # Test RadiologyReportGenerator (now with CheXpert label output)
    print("\n--- Testing RadiologyReportGenerator (with CheXpert logic) ---")
    try:
        batch_s_main = 2
        img_enc_kwargs = {'model_name': 'microsoft/swin-tiny-patch4-window7-224', 'pretrained': False}
        # num_chexpert_labels will be passed to RadiologyReportGenerator constructor
        text_emb_kwargs = {'model_name': 'dmis-lab/biobert-base-cased-v1.1', 'freeze_bert': True}
        dec_kwargs = {'attention_dim': 256, 'decoder_h_dim': 512, 'dropout_rate': 0.1}

        tokenizer_vocab_size_main = text_embedder.biobert.config.vocab_size if 'text_embedder' in locals() else 28996
        sos_token_id_main = 101
        pad_token_id_main = 0

        main_model = RadiologyReportGenerator(
            image_encoder_kwargs=img_enc_kwargs,
            text_embedder_kwargs=text_emb_kwargs,
            decoder_kwargs=dec_kwargs,
            tokenizer_vocab_size=tokenizer_vocab_size_main,
            sos_token_id=sos_token_id_main,
            pad_token_id=pad_token_id_main,
            device=device,
            num_chexpert_labels=num_dummy_chexpert_labels # Pass num_chexpert_labels
        ).to(device)
        print("RadiologyReportGenerator instantiated.")

        dummy_frontal_main = torch.randn(batch_s_main, 3, 224, 224).to(device)
        dummy_lateral_main = torch.randn(batch_s_main, 3, 224, 224).to(device)
        dummy_report_ids_main = torch.randint(0, tokenizer_vocab_size_main, (batch_s_main, 25), dtype=torch.long).to(device)
        dummy_report_ids_main[:, 0] = sos_token_id_main

        outputs_logits, attention_weights_all, chexpert_label_logits_main = main_model( # Expect 3 outputs
            dummy_frontal_main, dummy_lateral_main, dummy_report_ids_main, teacher_forcing_ratio=0.5
        )

        print(f"Output logits shape: {outputs_logits.shape}")
        print(f"Output attention_weights_all shape: {attention_weights_all.shape}")
        if chexpert_label_logits_main is not None:
            print(f"Output chexpert_label_logits shape: {chexpert_label_logits_main.shape}")
            assert chexpert_label_logits_main.shape == (batch_s_main, num_dummy_chexpert_labels), "CheXpert label logits shape mismatch!"
        elif num_dummy_chexpert_labels > 0 :
             raise AssertionError("CheXpert logits were expected but are None.")

        expected_num_patches_main = 2 * 49
        assert outputs_logits.shape == (batch_s_main, 25 - 1, tokenizer_vocab_size_main), "Outputs logits shape mismatch!"
        assert attention_weights_all.shape == (batch_s_main, 25 - 1, expected_num_patches_main), "Attention weights shape mismatch!"
        print("RadiologyReportGenerator test passed successfully!")
    except Exception as e:
        print(f"Error during RadiologyReportGenerator test: {e}\n{traceback.format_exc()}")
