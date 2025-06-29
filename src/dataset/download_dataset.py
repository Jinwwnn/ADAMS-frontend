from datasets import load_dataset
import pandas as pd

try:
    dataset_name = "RAGEVALUATION-HJKMY/TSBC_cleaned"
    dataset = load_dataset(dataset_name)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()


if 'train' in dataset:
    df = dataset['train'].to_pandas()
    print("Dataset loaded successfully to Pandas DataFrame.")
    print("Preview:")
    print(df.head())

    csv_file_path = 'output_data.csv'
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print(f"\nData saved successfully as CSV file: {csv_file_path}")

else:
    print("'train' split not found in dataset.")
