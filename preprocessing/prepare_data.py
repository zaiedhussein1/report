import pandas as pd

def load_and_preprocess_data(reports_path, projections_path, output_path):
    """
    Loads, preprocesses, and merges report and projection data.
    """
    # 1. Read CSV files into pandas DataFrames
    try:
        reports_df = pd.read_csv(reports_path)
        projections_df = pd.read_csv(projections_path)
    except FileNotFoundError as e:
        print(f"Error: One or both CSV files not found. {e}")
        return

    print("--- Reports DataFrame ---")
    print(reports_df.head())
    print(reports_df.info())
    print("\n--- Projections DataFrame ---")
    print(projections_df.head())
    print(projections_df.info())

    # 2. Perform an inner join on the uid column
    merged_df = pd.merge(reports_df, projections_df, on='uid', how='inner')
    print("\n--- Merged DataFrame ---")
    print(merged_df.head())
    print(merged_df.info())

    # 3. Filter the merged DataFrame to keep only studies that have both 'Frontal' and 'Lateral' projections
    # Group by 'uid' and check if both 'Frontal' and 'Lateral' are present in 'projection'
    def has_both_projections(group):
        return 'Frontal' in group['projection'].values and 'Lateral' in group['projection'].values

    # This part needs adjustment: we need to filter the original merged_df based on uids that satisfy the condition
    # First, identify the uids that have both projections
    uids_with_both_projections = merged_df.groupby('uid').filter(has_both_projections)['uid'].unique()

    # Then, filter the merged_df for these uids and create a copy to avoid SettingWithCopyWarning
    filtered_df = merged_df[merged_df['uid'].isin(uids_with_both_projections)].copy()

    # Since the above filtering might still keep separate rows for 'Frontal' and 'Lateral' for the same UID,
    # we might want to aggregate image filenames or just keep one record per UID if specific image data isn't critical at this stage.
    # For now, the task seems to imply keeping the report data, so we can drop duplicates based on 'uid' after ensuring they have both projections.
    # However, the prompt implies "studies that have both", so the `filtered_df` should be correct.
    # Let's re-evaluate if the goal is one row per study or multiple rows if a study has multiple images that fit.
    # The prompt mentions "filter the merged DataFrame", suggesting the rows themselves are filtered.
    # And "Create a new column full_text" implies this is done on the filtered rows.

    # Let's refine the filtering: we want to keep the *report data* for uids that have both projections.
    # The individual projection rows are useful for now.
    print("\n--- Filtered DataFrame (studies with Frontal and Lateral projections) ---")
    print(filtered_df.head())
    print(filtered_df.info())
    print(f"Shape after filtering for both projections: {filtered_df.shape}")


    # 4. Create a new column full_text by concatenating the findings and impression columns
    # Ensure findings and impressions are strings and handle potential NaN values
    filtered_df.loc[:, 'findings'] = filtered_df['findings'].fillna('')
    filtered_df.loc[:, 'impression'] = filtered_df['impression'].fillna('')
    filtered_df.loc[:, 'full_text'] = filtered_df['findings'] + ' ' + filtered_df['impression']

    # 5. Clean the full_text column
    filtered_df.loc[:, 'full_text'] = filtered_df['full_text'].str.replace('XXXX', '', regex=False)
    filtered_df.loc[:, 'full_text'] = filtered_df['full_text'].str.lower()
    # Also clean up extra whitespace that might result from concatenation or missing XXXXX
    filtered_df.loc[:, 'full_text'] = filtered_df['full_text'].str.strip().str.replace(r'\s+', ' ', regex=True)


    print("\n--- Processed DataFrame with full_text ---")
    print(filtered_df.head())
    print(f"Shape of the finally processed DataFrame: {filtered_df.shape}")

    # 6. Save the processed DataFrame
    try:
        filtered_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved processed data to {output_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

    return filtered_df

if __name__ == '__main__':
    # Use these paths for actual Kaggle environment
    # REPORTS_CSV = '/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv'
    # PROJECTIONS_CSV = '/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv'

    # Using dummy paths for local testing
    REPORTS_CSV = 'data/dummy_reports.csv'
    PROJECTIONS_CSV = 'data/dummy_projections.csv'

    OUTPUT_CSV = 'data/processed_reports_with_images.csv'

    # Create the output directory if it doesn't exist (relevant for the main data path)
    # import os
    # os.makedirs('data/', exist_ok=True) # Ensured by previous steps for dummy data

    processed_data_df = load_and_preprocess_data(REPORTS_CSV, PROJECTIONS_CSV, OUTPUT_CSV)

    if processed_data_df is not None:
        print("\n--- Original Processed DataFrame ---")
        print(processed_data_df.head())
        print(f"Shape: {processed_data_df.shape}")

        if not processed_data_df.empty and 'full_text' in processed_data_df.columns:
            from preprocess_text import tokenize_text

            print("\n--- Tokenizing full_text ---")
            # Ensure 'full_text' is not empty and contains strings
            texts_to_tokenize = processed_data_df['full_text'].fillna('').astype(str)

            if not texts_to_tokenize.empty:
                tokenized_outputs = tokenize_text(texts_to_tokenize)

                # Store tokenized outputs as strings of lists in the DataFrame
                processed_data_df['input_ids'] = [str(ids) for ids in tokenized_outputs['input_ids']]
                processed_data_df['attention_mask'] = [str(mask) for mask in tokenized_outputs['attention_mask']]

                print("\n--- DataFrame with Tokenized Outputs (first few rows) ---")
                print(processed_data_df[['uid', 'full_text', 'input_ids', 'attention_mask']].head())
                print(f"Shape after adding token columns: {processed_data_df.shape}")

                # Save the DataFrame with tokenized outputs
                try:
                    processed_data_df.to_csv(OUTPUT_CSV, index=False)
                    print(f"\nSuccessfully saved DataFrame with tokenized outputs to {OUTPUT_CSV}")
                except Exception as e:
                    print(f"Error saving DataFrame with tokenized outputs to CSV: {e}")
            else:
                print("No text found to tokenize.")
        else:
            print("DataFrame is empty or 'full_text' column is missing, skipping tokenization.")

        print("\n--- Final Check: Head of processed_data_df from main (with tokens) ---")
        print(processed_data_df.head())
        print(f"Final shape: {processed_data_df.shape}")
