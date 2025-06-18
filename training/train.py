import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import os
import random
import numpy as np
# NLTK import for BLEU is not directly needed here if calculate_nlp_metrics handles it.
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# pycocoevalcap imports will be in metrics.py

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.dataset import RadiologyDataset
from models.model import RadiologyReportGenerator
from evaluation.metrics import calculate_nlp_metrics # Import the new function

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    frontal_images = torch.stack([item['frontal_image'] for item in batch], dim=0)
    lateral_images = torch.stack([item['lateral_image'] for item in batch], dim=0)
    report_input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    chexpert_labels = None
    if 'chexpert_labels' in batch[0] and batch[0]['chexpert_labels'] is not None:
        if all('chexpert_labels' in item and item['chexpert_labels'] is not None for item in batch):
            chexpert_labels = torch.stack([item['chexpert_labels'] for item in batch], dim=0)
    return {
        'frontal_images': frontal_images,
        'lateral_images': lateral_images,
        'report_input_ids': report_input_ids,
        'chexpert_labels': chexpert_labels,
        'uids': [item['uid'] for item in batch] # Pass UIDs for metric calculation
    }

def validate_epoch(model, val_dataloader, tokenizer, device, args):
    model.eval()
    print(f"\n--- Starting Validation ---")

    references_for_coco = {}
    hypotheses_for_coco = {}

    # Import generate_report_greedy here as it's specific to validation/evaluation
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation')))
    try:
        from evaluate import generate_report_greedy
    except ImportError:
        print("Could not import generate_report_greedy. Validation cannot proceed with generation.")
        model.train()
        return {"CIDEr": 0.0, "Bleu_4": 0.0} # Return default scores

    if val_dataloader is None:
        print("Validation dataloader is not set up. Skipping validation.")
        model.train()
        return {"CIDEr": 0.0, "Bleu_4": 0.0}

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            frontal_images = batch_data['frontal_images'].to(device)
            lateral_images = batch_data['lateral_images'].to(device)
            report_ids_for_gt = batch_data['report_input_ids']
            uids_batch = batch_data['uids']

            generated_texts_batch, _ = generate_report_greedy(
                model, frontal_images, lateral_images,
                tokenizer, args.max_gen_len, device
            )

            for i in range(len(generated_texts_batch)):
                uid = uids_batch[i]
                reference_text = tokenizer.decode(report_ids_for_gt[i].tolist(), skip_special_tokens=True)
                generated_text = generated_texts_batch[i]

                if uid not in references_for_coco: # Handle multiple references if dataset provides them (not in current dummy)
                    references_for_coco[uid] = []
                references_for_coco[uid].append(reference_text)
                hypotheses_for_coco[uid] = [generated_text] # COCO expects list of strings for hypo

                if batch_idx == 0 and i == 0 : # Print one example from the first val batch
                     print(f"  Val Example UID {uid} - GT: {reference_text}")
                     print(f"  Val Example UID {uid} - Gen: {generated_text}")

    model.train() # Set model back to train mode before heavy computation of metrics

    if not references_for_coco or not hypotheses_for_coco:
        print("No references or hypotheses collected for metric calculation.")
        return {"CIDEr": 0.0, "Bleu_4": 0.0}

    print("Calculating NLP metrics using pycocoevalcap...")
    scores_dict = calculate_nlp_metrics(references_for_coco, hypotheses_for_coco)

    if scores_dict:
        print(f"Validation Metrics: CIDEr: {scores_dict.get('CIDEr', 0.0):.4f}, BLEU-4: {scores_dict.get('Bleu_4', 0.0):.4f}, ROUGE_L: {scores_dict.get('ROUGE_L',0.0):.4f}")
    else:
        print("Metrics calculation returned empty or failed. Defaulting scores to 0.")
        scores_dict = {"CIDEr": 0.0, "Bleu_4": 0.0, "ROUGE_L": 0.0} # Ensure dict for return

    print(f"--- Validation Complete ---\n")
    return scores_dict


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.biobert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.biobert_model_name)
    vocab_size = tokenizer.vocab_size
    sos_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id

    if sos_token_id is None: raise ValueError("SOS token ID not found in tokenizer")
    if pad_token_id is None: raise ValueError("PAD token ID not found in tokenizer")
    print(f"Vocab size: {vocab_size}, SOS ID: {sos_token_id}, PAD ID: {pad_token_id}")

    print("Setting up Training DataLoader...")
    train_dataset = RadiologyDataset(csv_file_path=args.csv_path, image_base_path=args.image_base_path, train_mode=True)
    if len(train_dataset) == 0: print("Training dataset is empty. Exiting."); return
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"Training DataLoader: {len(train_dataset)} samples, {len(train_dataloader)} batches.")

    print("Setting up Validation DataLoader...")
    val_dataset = RadiologyDataset(csv_file_path=args.val_csv_path, image_base_path=args.image_base_path, train_mode=False)
    val_dataloader = None
    if len(val_dataset) == 0: print("Validation dataset is empty. Validation will be skipped.")
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
        print(f"Validation DataLoader: {len(val_dataset)} samples, {len(val_dataloader)} batches.")

    print("Initializing model, optimizer, and loss functions...")
    image_encoder_kwargs = {'model_name': args.swin_model_name, 'pretrained': True}
    text_embedder_kwargs = {'model_name': args.biobert_model_name, 'freeze_bert': args.freeze_bert}
    decoder_kwargs = {'attention_dim': args.attention_dim, 'decoder_h_dim': args.decoder_h_dim, 'dropout_rate': args.dropout_rate}

    model = RadiologyReportGenerator(
        image_encoder_kwargs=image_encoder_kwargs, text_embedder_kwargs=text_embedder_kwargs,
        decoder_kwargs=decoder_kwargs, tokenizer_vocab_size=vocab_size,
        sos_token_id=sos_token_id, pad_token_id=pad_token_id, device=device,
        num_chexpert_labels=args.num_chexpert_labels
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_report = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_chexpert = nn.BCEWithLogitsLoss()

    best_metric_score = 0.0 # Changed from best_bleu_score, using CIDEr as primary for checkpointing
    best_epoch = 0
    print("Setup complete. Starting training loop...")

    for epoch_idx in range(args.epochs):
        model.train()
        epoch_total_loss, epoch_report_loss, epoch_chexpert_loss = 0.0, 0.0, 0.0
        print(f"\n--- Starting Training Epoch {epoch_idx+1}/{args.epochs} ---")
        for batch_idx, batch_data in enumerate(train_dataloader):
            frontal_images = batch_data['frontal_images'].to(device)
            lateral_images = batch_data['lateral_images'].to(device)
            report_input_ids = batch_data['report_input_ids'].to(device)
            true_chexpert_labels = batch_data['chexpert_labels']
            if true_chexpert_labels is not None: true_chexpert_labels = true_chexpert_labels.to(device)

            optimizer.zero_grad()
            outputs_logits, _, chexpert_label_logits = model(
                frontal_images, lateral_images, report_input_ids,
                teacher_forcing_ratio=args.teacher_forcing_ratio
            )
            loss_report = criterion_report(outputs_logits.reshape(-1, vocab_size), report_input_ids[:, 1:].reshape(-1))
            loss_chexpert = torch.tensor(0.0).to(device)
            if chexpert_label_logits is not None and true_chexpert_labels is not None and args.num_chexpert_labels > 0:
                loss_chexpert = criterion_chexpert(chexpert_label_logits, true_chexpert_labels)

            total_loss = loss_report + args.lambda_chexpert * loss_chexpert
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item(); epoch_report_loss += loss_report.item()
            if chexpert_label_logits is not None and true_chexpert_labels is not None: epoch_chexpert_loss += loss_chexpert.item()

            if batch_idx % args.log_interval == 0 or len(train_dataloader) == 1:
                log_msg = (f"Epoch {epoch_idx+1}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                           f"Total Loss: {total_loss.item():.4f}, Report Loss: {loss_report.item():.4f}")
                if chexpert_label_logits is not None and true_chexpert_labels is not None:
                    log_msg += f", CheXpert Loss: {loss_chexpert.item():.4f}"
                print(log_msg)

        avg_epoch_total_loss = epoch_total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        avg_epoch_report_loss = epoch_report_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        avg_epoch_chexpert_loss = epoch_chexpert_loss / (len(train_dataloader) if args.num_chexpert_labels > 0 and epoch_chexpert_loss > 0 else 1)
        print(f"Epoch {epoch_idx+1} Avg Training Losses: Total: {avg_epoch_total_loss:.4f}, Report: {avg_epoch_report_loss:.4f}, CheXpert: {avg_epoch_chexpert_loss:.4f}")

        current_metrics = {"CIDEr": 0.0, "Bleu_4": 0.0} # Default if no val_dataloader
        if val_dataloader:
            current_metrics = validate_epoch(model, val_dataloader, tokenizer, device, args)

        current_cider_score = current_metrics.get('CIDEr', 0.0) # Use CIDEr for checkpointing
        if current_cider_score > best_metric_score: # Checkpoint based on CIDEr
            best_metric_score = current_cider_score
            best_epoch = epoch_idx + 1
            save_path = os.path.join(args.checkpoint_dir, f'best_model_epoch_{best_epoch}_cider_{best_metric_score:.4f}.pth')
            torch.save({
                'epoch': best_epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_metric_score': best_metric_score, # Save CIDEr
                'args': args
            }, save_path)
            print(f"New best model saved to {save_path} with CIDEr: {best_metric_score:.4f}")
        else:
            print(f"Validation CIDEr {current_cider_score:.4f} did not improve from best CIDEr {best_metric_score:.4f} (Epoch {best_epoch}).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Radiology Report Generation Training Script")

    parser.add_argument('--csv_path', type=str, default='data/processed_reports_with_images.csv', help='Path to training CSV file')
    parser.add_argument('--image_base_path', type=str, default='data/dummy_images/', help='Base path for images')
    parser.add_argument('--val_csv_path', type=str, default='data/processed_reports_with_images.csv', help='Path to validation CSV file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--max_gen_len', type=int, default=30, help='Max report length for validation generation')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.75, help='Teacher forcing ratio')
    parser.add_argument('--lambda_chexpert', type=float, default=0.2, help='Weight for CheXpert loss component')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--log_interval', type=int, default=1, help='Log batch loss every N batches')
    parser.add_argument('--swin_model_name', type=str, default='microsoft/swin-tiny-patch4-window7-224', help='Swin model')
    parser.add_argument('--biobert_model_name', type=str, default='dmis-lab/biobert-base-cased-v1.1', help='BioBERT model')
    parser.add_argument('--freeze_bert', type=lambda x: (str(x).lower() == 'true'), default=True, help='Freeze BioBERT')
    parser.add_argument('--decoder_h_dim', type=int, default=512, help='Decoder LSTM hidden dim')
    parser.add_argument('--attention_dim', type=int, default=256, help='Attention internal dim')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Decoder LSTM dropout')
    parser.add_argument('--num_chexpert_labels', type=int, default=14, help='Number of CheXpert labels')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Checkpoint directory')

    args = parser.parse_args([])
    main(args)
