import pandas as pd
import numpy as np
import os

def preprocess_data(raw_data_path, processed_data_path):
    """
    Loads, preprocesses, and saves the mushroom dataset.
    """
    print(f"Memuat data dari: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    print("Memulai preprocessing...")
    # Target Encoding
    df['class'] = df['class'].map({'p': 1, 'e': 0})

    # Drop kolom dengan satu nilai unik
    cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Kolom yang dihapus: {cols_to_drop}")

    # One-Hot Encoding
    features_to_encode = [col for col in df.columns if col != 'class' and df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=features_to_encode, drop_first=True)

    # Mengubah semua kolom boolean hasil OHE menjadi integer (0 atau 1)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    print(f"Preprocessing selesai. Dimensi data: {df.shape}")

    # Membuat direktori jika belum ada
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    # Menyimpan data yang sudah diproses
    df.to_csv(processed_data_path, index=False)
    print(f"Data yang sudah diproses disimpan di: {processed_data_path}")

if __name__ == '__main__':
    # Menjalankan fungsi dengan path default
    # Path ini relatif terhadap lokasi script saat dijalankan oleh workflow
    raw_path = '../namadataset_raw/mushrooms.csv'
    processed_path = 'namadataset_preprocessing/mushrooms_preprocessed.csv'
    preprocess_data(raw_path, processed_path)