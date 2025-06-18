import pandas as pd
import numpy as np
import ast

# Adjust path to import from parent directory's modules if preprocess_text is not in the same dir
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess_text import tokenize_text # Ensure this import works

def load_and_preprocess_data(reports_path, projections_path, output_path):
    try:
        reports_df = pd.read_csv(reports_path)
        projections_df = pd.read_csv(projections_path)
    except FileNotFoundError as e:
        print(f"Error: One or both CSV files not found. {e}")
        return None

    # print("--- Reports DataFrame ---"); print(reports_df.head(2)) # Verbosity reduced
    # print("\n--- Projections DataFrame ---"); print(projections_df.head(2))

    merged_df = pd.merge(reports_df, projections_df, on='uid', how='inner')
    # print("\n--- Merged DataFrame ---"); print(merged_df.head(2))

    def has_both_projections(group):
        return 'Frontal' in group['projection'].values and 'Lateral' in group['projection'].values

    uids_with_both_projections = merged_df.groupby('uid').filter(has_both_projections)['uid'].unique()
    if len(uids_with_both_projections) == 0:
        print("No UIDs found with both Frontal and Lateral projections. Output will be empty.")
        # Create an empty DataFrame with expected columns to avoid downstream errors if possible
        # Or handle this more gracefully in consuming scripts.
        # For now, let's return None as the original script did for other errors.
        return None

    filtered_df = merged_df[merged_df['uid'].isin(uids_with_both_projections)].copy()
    # print("\n--- Filtered DataFrame (studies with Frontal and Lateral projections) ---"); print(filtered_df.head(2))
    # print(f"Shape after filtering: {filtered_df.shape}")

    if filtered_df.empty:
        print("Filtered DataFrame is empty (after filtering for projections). No data to process.")
        return None

    filtered_df.loc[:, 'findings'] = filtered_df['findings'].fillna('')
    filtered_df.loc[:, 'impression'] = filtered_df['impression'].fillna('')
    filtered_df.loc[:, 'full_text'] = filtered_df['findings'] + ' ' + filtered_df['impression']
    filtered_df.loc[:, 'full_text'] = filtered_df['full_text'].str.replace('XXXX', '', regex=False)
    filtered_df.loc[:, 'full_text'] = filtered_df['full_text'].str.lower()
    filtered_df.loc[:, 'full_text'] = filtered_df['full_text'].str.strip().str.replace(r'\s+', ' ', regex=True)

    # Add BioBERT tokenization outputs (input_ids, attention_mask)
    if not filtered_df['full_text'].empty:
        print("Tokenizing 'full_text' with BioBERT...")
        # Assuming tokenize_text expects a pandas Series
        tokenized_outputs = tokenize_text(filtered_df['full_text'])
        filtered_df.loc[:, 'input_ids'] = [str(ids) for ids in tokenized_outputs['input_ids']]
        filtered_df.loc[:, 'attention_mask'] = [str(mask) for mask in tokenized_outputs['attention_mask']]
        print("Tokenization complete. 'input_ids' and 'attention_mask' columns added.")
    else:
        print("Warning: 'full_text' column is empty or missing. Skipping BioBERT tokenization.")
        filtered_df.loc[:, 'input_ids'] = [str([]) for _ in range(len(filtered_df))] # Empty list string
        filtered_df.loc[:, 'attention_mask'] = [str([]) for _ in range(len(filtered_df))]


    # Add dummy CheXpert labels
    num_chexpert_labels = 14
    chexpert_labels_list_of_lists = [
        list(np.random.randint(0, 2, size=num_chexpert_labels)) for _ in range(len(filtered_df))
    ]
    filtered_df.loc[:, 'chexpert_labels_str'] = [str(labels) for labels in chexpert_labels_list_of_lists]
    print("Dummy CheXpert labels added as 'chexpert_labels_str'.")

    print("\n--- Processed DataFrame (first 2 rows relevant columns) ---")
    cols_to_show = ['uid', 'full_text']
    if 'input_ids' in filtered_df.columns: cols_to_show.append('input_ids')
    if 'chexpert_labels_str' in filtered_df.columns: cols_to_show.append('chexpert_labels_str')
    print(filtered_df[cols_to_show].head(2))
    # print(f"Shape of the finally processed DataFrame: {filtered_df.shape}")

    try:
        filtered_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved processed data to {output_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")
        return None

    return filtered_df

if __name__ == '__main__':
    REPORTS_CSV = 'data/dummy_reports.csv'
    PROJECTIONS_CSV = 'data/dummy_projections.csv'
    OUTPUT_CSV = 'data/processed_reports_with_images.csv'

    print(f"Running prepare_data.py with dummy data: {REPORTS_CSV}, {PROJECTIONS_CSV}")
    processed_data = load_and_preprocess_data(REPORTS_CSV, PROJECTIONS_CSV, OUTPUT_CSV)

    if processed_data is not None:
        print("\n--- Final Check: Head of processed_data from main ---")
        cols_to_show_final = ['uid', 'filename', 'projection', 'full_text']
        if 'input_ids' in processed_data.columns: cols_to_show_final.append('input_ids')
        if 'attention_mask' in processed_data.columns: cols_to_show_final.append('attention_mask')
        if 'chexpert_labels_str' in processed_data.columns: cols_to_show_final.append('chexpert_labels_str')
        print(processed_data[cols_to_show_final].head())
        print(f"Final shape: {processed_data.shape}")
        print(f"Columns: {processed_data.columns.tolist()}")
    else:
        print("Data processing failed.")
