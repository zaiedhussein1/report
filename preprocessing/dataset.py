import torch
from torch.utils.data import Dataset
import pandas as pd
import ast  # For safely evaluating string representations of lists

class RadiologyDataset(Dataset):
    def __init__(self, csv_file_path, image_base_path=""): # Add image_base_path if needed
        """
        Args:
            csv_file_path (string): Path to the csv file with processed reports and image info.
            image_base_path (string): Base path for image filenames if they are relative.
        """
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

        # Parse stringified lists into actual lists of integers
        for col in ['input_ids', 'attention_mask']:
            if col in df.columns:
                try:
                    # Ensure the column is treated as string before applying ast.literal_eval
                    df[col] = df[col].astype(str).apply(ast.literal_eval)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing column {col}: {e}. Ensure it contains valid list strings.")
                    # Handle cases where some rows might be unparseable, e.g., fill with None or drop
                    # For now, if parsing fails for any row, we might have an issue with the whole column.
                    # Depending on robustness needs, one might add row-specific error handling.
                    self.samples = []
                    return
            else:
                print(f"Warning: Column {col} not found in CSV. It will not be available in the dataset.")


        self.samples = []
        # Group by 'uid' to consolidate report text and pair frontal/lateral images
        for uid, group in df.groupby('uid'):
            # Assuming 'full_text', 'input_ids', 'attention_mask' are the same for all rows of the same uid
            # Take these from the first row of the group
            report_data = group.iloc[0]

            input_ids = report_data.get('input_ids')
            attention_mask = report_data.get('attention_mask')

            if not isinstance(input_ids, list) or not isinstance(attention_mask, list):
                print(f"Warning: UID {uid} has improperly parsed tokenized report data (expected list). Skipping. Type input_ids: {type(input_ids)}, Type attention_mask: {type(attention_mask)}")
                continue

            frontal_image_path = None
            lateral_image_path = None

            for _, row in group.iterrows():
                projection = row.get('projection', '').lower()
                filename = row.get('filename', '')

                if self.image_base_path and filename: # Prepend base path if filename is relative
                    current_image_path = f"{self.image_base_path.rstrip('/')}/{filename}"
                else:
                    current_image_path = filename

                if projection == 'frontal':
                    frontal_image_path = current_image_path
                elif projection == 'lateral':
                    lateral_image_path = current_image_path

            # Only include samples that have both frontal and lateral images
            if frontal_image_path and lateral_image_path:
                sample = {
                    'uid': uid,
                    'frontal_image_path': frontal_image_path,
                    'lateral_image_path': lateral_image_path,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'full_text': report_data.get('full_text', '') # For reference
                }
                self.samples.append(sample)
            else:
                # This case should ideally not happen if prepare_data.py filters correctly
                print(f"Warning: UID {uid} is missing either frontal or lateral image path after grouping. Skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(sample['input_ids'], dtype=torch.long)
        attention_mask_tensor = torch.tensor(sample['attention_mask'], dtype=torch.long)

        # Placeholder for image loading and transformation
        # from PIL import Image
        # frontal_image = Image.open(sample['frontal_image_path']).convert('RGB')
        # lateral_image = Image.open(sample['lateral_image_path']).convert('RGB')
        # if hasattr(self, 'transform') and self.transform:
        #     frontal_image = self.transform(frontal_image)
        #     lateral_image = self.transform(lateral_image)

        return {
            'uid': sample['uid'],
            'frontal_image_path': sample['frontal_image_path'], # Path for now, will be image tensor later
            'lateral_image_path': sample['lateral_image_path'], # Path for now, will be image tensor later
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'full_text': sample['full_text'] # For debugging or reference
            # 'frontal_image': frontal_image, # To be added
            # 'lateral_image': lateral_image, # To be added
        }

if __name__ == '__main__':
    # This assumes 'data/processed_reports_with_images.csv' exists and is populated
    # from the previous steps (using dummy data).
    # The dummy CSV has uid 1 with both Frontal and Lateral images.

    dummy_csv_path = 'data/processed_reports_with_images.csv'

    print(f"Attempting to load dataset from: {dummy_csv_path}")
    dataset = RadiologyDataset(csv_file_path=dummy_csv_path)

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        print("\nSample 0:")
        try:
            sample_0 = dataset[0]
            for key, value in sample_0.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor of shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error retrieving or printing sample 0: {e}")

        if len(dataset) > 1: # If there's more than one unique UID processed
            print("\nSample 1 (if exists):")
            try:
                sample_1 = dataset[1]
                for key, value in sample_1.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: Tensor of shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {value}")
            except IndexError:
                print("  Sample 1 does not exist.")
            except Exception as e:
                print(f"Error retrieving or printing sample 1: {e}")

    else:
        print("Dataset is empty. Check CSV path, content, and parsing logic in __init__.")

    # Example of how to handle a base image path:
    # print("\n--- Testing with a base image path ---")
    # kaggle_image_base = "/mnt/data/dummy_images/" # Replace with a real path if testing image loading
    # # To make this testable, let's assume dummy filenames are just 'filename.png'
    # # and we want to prepend a path.
    # # We would need to ensure the filenames in the dummy CSV are suitable for this.
    # # e.g. '1_IM-0001-4001.dcm.png' would become '/mnt/data/dummy_images/1_IM-0001-4001.dcm.png'

    # dataset_with_base_path = RadiologyDataset(csv_file_path=dummy_csv_path, image_base_path=kaggle_image_base)
    # if len(dataset_with_base_path) > 0:
    #     print("\nSample 0 with base image path:")
    #     sample_0_bp = dataset_with_base_path[0]
    #     print(f"  Frontal Image Path: {sample_0_bp['frontal_image_path']}")
    #     print(f"  Lateral Image Path: {sample_0_bp['lateral_image_path']}")
    # else:
    #     print("Dataset (with base path) is empty.")
