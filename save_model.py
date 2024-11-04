import pandas as pd

def save_to_csv(text, label, file_path='data/data.csv'):
    # Append new data to the CSV file
    new_data = pd.DataFrame([[text, label]], columns=['text', 'label'])
    new_data.to_csv(file_path, mode='a', header=False, index=False)
