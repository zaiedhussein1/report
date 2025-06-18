import torch
from torch.utils.data import Dataset
import pandas as pd
import ast  # For safely evaluating string representations of lists
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np # Though not directly used here, good to have for context if dealing with np arrays before tensor conversion

class RadiologyDataset(Dataset):
    def __init__(self, csv_file_path, image_base_path="", train_mode=True):
        self.image_base_path = image_base_path

        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"Error: The file {csv_file_path} was not found.")
            self.samples = []
            return
        except Exception as e:
            print(f"Error loading CSV {csv_file_path}: {e}")
            self.samples = []
            return

        # Parse stringified lists into actual lists of numbers/integers
        # For input_ids and attention_mask (text tokens)
        for col in ['input_ids', 'attention_mask']:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(str).apply(ast.literal_eval)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing text token column {col}: {e}. Ensure it contains valid list strings.")
                    self.samples = []
                    return
            # else: # Commented out to reduce noise, as these might not always be present if only evaluating images
                # print(f"Warning: Text token column {col} not found in CSV.")

        # Parse CheXpert labels string
        if 'chexpert_labels_str' in df.columns:
            try:
                df['chexpert_labels'] = df['chexpert_labels_str'].astype(str).apply(ast.literal_eval)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing 'chexpert_labels_str' column: {e}. Ensure it contains valid list strings.")
                self.samples = []
                return
        else:
            print("Warning: 'chexpert_labels_str' column not found in CSV. CheXpert labels will not be available.")
            # Add a dummy column of Nones or empty lists if critical for downstream code expecting the key
            df['chexpert_labels'] = [None] * len(df)


        if train_mode:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                # Cutout(n_holes=1, length=40), # Assuming Cutout is defined if used
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.samples = []
        for uid, group in df.groupby('uid'):
            report_data = group.iloc[0] # Contains 'input_ids', 'attention_mask', 'full_text', 'chexpert_labels'

            # Ensure essential text data is present if needed (might be optional depending on use case)
            # input_ids = report_data.get('input_ids')
            # attention_mask_text = report_data.get('attention_mask') # Renamed to avoid clash if image attention is used later

            frontal_image_filename = None
            lateral_image_filename = None

            for _, row in group.iterrows():
                projection = row.get('projection', '').lower()
                filename = row.get('filename', '')
                if projection == 'frontal': frontal_image_filename = filename
                elif projection == 'lateral': lateral_image_filename = filename

            if frontal_image_filename and lateral_image_filename:
                sample = {
                    'uid': uid,
                    'frontal_image_filename': frontal_image_filename,
                    'lateral_image_filename': lateral_image_filename,
                    'input_ids': report_data.get('input_ids'), # May be None if not present
                    'attention_mask': report_data.get('attention_mask'), # May be None
                    'full_text': report_data.get('full_text', ''),
                    'chexpert_labels': report_data.get('chexpert_labels') # Parsed list of labels
                }
                self.samples.append(sample)
            # else: # Reduce noise, this case is handled by empty self.samples
                # print(f"Warning: UID {uid} is missing image paths. Skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item_dict = {'uid': sample['uid'], 'full_text': sample['full_text']}

        # Image processing
        frontal_image_path = os.path.join(self.image_base_path, sample['frontal_image_filename'])
        lateral_image_path = os.path.join(self.image_base_path, sample['lateral_image_filename'])
        try:
            frontal_image_pil = Image.open(frontal_image_path).convert('RGB')
            lateral_image_pil = Image.open(lateral_image_path).convert('RGB')
            if self.transform:
                item_dict['frontal_image'] = self.transform(frontal_image_pil)
                item_dict['lateral_image'] = self.transform(lateral_image_pil)
        except FileNotFoundError as e:
            print(f"Error: Image file not found for UID {sample['uid']}. Path: {e.filename}")
            # Return images as None or handle as error
            item_dict['frontal_image'] = None
            item_dict['lateral_image'] = None

        # Text data processing (if present)
        if sample['input_ids'] is not None:
            item_dict['input_ids'] = torch.tensor(sample['input_ids'], dtype=torch.long)
        if sample['attention_mask'] is not None: # This is for text embedder
            item_dict['attention_mask'] = torch.tensor(sample['attention_mask'], dtype=torch.long)

        # CheXpert labels processing (if present)
        if sample['chexpert_labels'] is not None:
            try:
                item_dict['chexpert_labels'] = torch.tensor(sample['chexpert_labels'], dtype=torch.float) # Use float for BCEWithLogitsLoss
            except TypeError: # If sample['chexpert_labels'] was None and couldn't be converted
                 print(f"Warning: CheXpert labels for UID {sample['uid']} are None or invalid type, cannot convert to tensor.")
                 item_dict['chexpert_labels'] = None # Or a default tensor: torch.zeros(14, dtype=torch.float)

        return item_dict

if __name__ == '__main__':
    dummy_csv_path = 'data/processed_reports_with_images.csv'
    dummy_image_base_path = 'data/dummy_images/'

    print(f"Attempting to load dataset from: {dummy_csv_path}")
    dataset = RadiologyDataset(csv_file_path=dummy_csv_path, image_base_path=dummy_image_base_path, train_mode=False)

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        print("\nSample 0:")
        try:
            sample_0 = dataset[0]
            for key, value in sample_0.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor of shape {value.shape}, dtype {value.dtype}")
                elif value is None:
                     print(f"  {key}: None")
                else:
                    print(f"  {key}: (Type: {type(value)}) {value}")

            # Specific check for chexpert_labels
            if 'chexpert_labels' in sample_0 and sample_0['chexpert_labels'] is not None:
                print(f"  CheXpert labels (sample 0) value: {sample_0['chexpert_labels']}")
                assert sample_0['chexpert_labels'].shape[0] == 14, "CheXpert labels should have 14 values."
                assert sample_0['chexpert_labels'].dtype == torch.float32, "CheXpert labels should be FloatTensor."
            else:
                print("  CheXpert labels not found or None in sample 0.")

        except Exception as e:
            print(f"Error retrieving or printing sample 0: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Dataset is empty.")
