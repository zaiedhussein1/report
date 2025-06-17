import unittest
import pandas as pd
import os
import sys

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from preprocessing.prepare_data import load_and_preprocess_data

class TestPrepareData(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test data
        self.test_data_dir = "test_temp_data"
        os.makedirs(self.test_data_dir, exist_ok=True)

        self.reports_path = os.path.join(self.test_data_dir, "dummy_reports.csv")
        self.projections_path = os.path.join(self.test_data_dir, "dummy_projections.csv")
        self.output_path = os.path.join(self.test_data_dir, "processed_output.csv")

        # Create dummy reports CSV
        reports_data = {
            'uid': [1, 2, 3, 4],
            'findings': ["Finding 1 XXXX", "Finding 2", None, "Finding 4"],
            'impression': ["Impression 1", "Impression 2 XXXX", "Impression 3", None]
        }
        pd.DataFrame(reports_data).to_csv(self.reports_path, index=False)

        # Create dummy projections CSV
        projections_data = {
            'uid': [1, 1, 2, 3, 3, 5], # UID 1 & 3 have Frontal & Lateral
            'projection': ['Frontal', 'Lateral', 'Frontal', 'Lateral', 'Frontal', 'AP']
        }
        pd.DataFrame(projections_data).to_csv(self.projections_path, index=False)

    def tearDown(self):
        # Clean up created files and directory
        if os.path.exists(self.reports_path):
            os.remove(self.reports_path)
        if os.path.exists(self.projections_path):
            os.remove(self.projections_path)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        # Attempt to remove directory, ensure it's empty first
        if os.path.exists(self.test_data_dir):
            # Check if directory is empty before trying to remove
            if not os.listdir(self.test_data_dir):
                 os.rmdir(self.test_data_dir)
            else:
                # This case should ideally not happen if all files are removed above
                print(f"Warning: Test data directory {self.test_data_dir} is not empty. Manual cleanup might be needed.")


    def test_load_and_preprocess_data_smoke_test(self):
        # Test if the function runs without error and produces an output file
        processed_df = load_and_preprocess_data(self.reports_path, self.projections_path, self.output_path)

        self.assertTrue(os.path.exists(self.output_path), "Output CSV file was not created.")
        self.assertIsNotNone(processed_df, "Processed DataFrame should not be None.")
        self.assertFalse(processed_df.empty, "Processed DataFrame should not be empty.")

        # Check for expected columns (after filtering and processing)
        self.assertIn('uid', processed_df.columns)
        self.assertIn('full_text', processed_df.columns)
        self.assertIn('projection', processed_df.columns) # Still there from merge

        # Check that only UIDs with both Frontal and Lateral projections are present
        # In our dummy data, only UID 1 and 3 should qualify.
        # UID 1 has 2 rows (one for Frontal, one for Lateral)
        # UID 3 has 2 rows (one for Frontal, one for Lateral)
        # So, total 4 rows.
        self.assertEqual(len(processed_df['uid'].unique()), 2, "Should only contain UIDs with both projection types.")
        self.assertEqual(len(processed_df), 4, "Expected 4 rows in the output for UIDs 1 and 3.")


    def test_full_text_cleaning(self):
        processed_df = load_and_preprocess_data(self.reports_path, self.projections_path, self.output_path)

        # Check specific cleaning results for UID 1 (Frontal and Lateral rows)
        # Original finding for UID 1: "Finding 1 XXXX", impression: "Impression 1"
        # Expected full_text: "finding 1 impression 1" (lowercase, XXXX removed, extra space handled)

        # Get the full_text for uid 1. Since there are two rows for uid 1 (Frontal & Lateral),
        # and full_text is generated from findings & impression which are the same for both rows of uid 1.
        uid1_texts = processed_df[processed_df['uid'] == 1]['full_text'].unique()
        self.assertEqual(len(uid1_texts), 1) # Should be only one unique full_text for uid 1
        self.assertEqual(uid1_texts[0], "finding 1 impression 1")

        # Original finding for UID 2: "Finding 2", impression: "Impression 2 XXXX"
        # UID 2 should be filtered out as it only has 'Frontal'
        self.assertNotIn(2, processed_df['uid'].values, "UID 2 should be filtered out.")

if __name__ == '__main__':
    unittest.main()
