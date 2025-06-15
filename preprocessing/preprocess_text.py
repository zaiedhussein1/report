import pandas as pd
from transformers import AutoTokenizer

def tokenize_text(text_series: pd.Series,
                  model_name: str = 'dmis-lab/biobert-base-cased-v1.1',
                  max_length: int = 128) -> dict:
    """
    Tokenizes a pandas Series of text using a specified Hugging Face transformer model.

    Args:
        text_series (pd.Series): A pandas Series containing the text to tokenize.
        model_name (str): The name of the Hugging Face model tokenizer to use.
        max_length (int): The maximum length for padding and truncation.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask' as lists of lists.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer {model_name}: {e}")
        # Fallback to a basic tokenizer if BioBERT is not available or fails.
        # This is mainly for environments without internet or specific model access issues.
        # For this task, we expect BioBERT to be available.
        print("Falling back to bert-base-uncased due to previous error.")
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            print("Successfully loaded bert-base-uncased as fallback.")
        except Exception as e_fallback:
            print(f"Error loading fallback tokenizer bert-base-uncased: {e_fallback}")
            raise

    input_ids_list = []
    attention_mask_list = []

    for text in text_series:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt',  # Return PyTorch tensors (will convert to list for storage)
                       )
        input_ids_list.append(encoded_dict['input_ids'].squeeze().tolist())
        attention_mask_list.append(encoded_dict['attention_mask'].squeeze().tolist())

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list
    }

if __name__ == '__main__':
    # Example Usage for testing
    sample_texts = pd.Series([
        "This is a sample sentence for tokenization.",
        "Another example with different medical terms like XXXXX and YYYY.",
        "Short text."
    ])

    print(f"Using tokenizer: dmis-lab/biobert-base-cased-v1.1")
    tokenized_output = tokenize_text(sample_texts)

    print("\nSample Input Texts:")
    print(sample_texts)
    print("\nTokenized Input IDs (first 5 tokens):")
    for i, ids in enumerate(tokenized_output['input_ids']):
        print(f"Text {i+1}: {ids[:5]}...")
    print("\nAttention Masks (first 5 tokens):")
    for i, mask in enumerate(tokenized_output['attention_mask']):
        print(f"Text {i+1}: {mask[:5]}...")

    # Test with a different tokenizer if needed
    # print(f"\nUsing tokenizer: bert-base-uncased")
    # tokenized_output_bert_base = tokenize_text(sample_texts, model_name='bert-base-uncased')
    # print("\nTokenized Input IDs with bert-base-uncased (first 5 tokens):")
    # for i, ids in enumerate(tokenized_output_bert_base['input_ids']):
    #     print(f"Text {i+1}: {ids[:5]}...")

    # Example of how the output might be added to a DataFrame
    df = pd.DataFrame({'text': sample_texts})
    df['input_ids'] = tokenized_output['input_ids']
    df['attention_mask'] = tokenized_output['attention_mask']
    print("\nDataFrame with tokenized outputs:")
    print(df.head())
