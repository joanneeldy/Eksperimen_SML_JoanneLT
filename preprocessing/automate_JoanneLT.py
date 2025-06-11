import pandas as pd
import os

def preprocess_data(raw_path, processed_path):
    """
    Fungsi ini memuat data mentah, melakukan preprocessing, dan menyimpannya.
    """
    print(f"Memuat data dari: {raw_path}")
    df = pd.read_csv(raw_path)

    print("Memulai preprocessing...")

    # Target Encoding: Mengubah kolom 'class' menjadi numerik (p=1, e=0)
    df['class'] = df['class'].map({'p': 1, 'e': 0})

    # Drop kolom yang hanya memiliki satu nilai unik (tidak berguna untuk model)
    cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Kolom yang dihapus: {cols_to_drop}")

    # One-Hot Encoding untuk semua fitur kategorikal lainnya
    features_to_encode = [col for col in df.columns if col != 'class' and df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=features_to_encode, drop_first=True)

    # Mengubah semua kolom boolean hasil OHE menjadi integer (0 atau 1)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    print(f"Preprocessing selesai. Dimensi data baru: {df.shape}")

    # Membuat direktori output jika belum ada
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # Menyimpan data yang sudah diproses
    df.to_csv(processed_path, index=False)
    print(f"Data yang sudah diproses disimpan di: {processed_path}")

if __name__ == '__main__':
    # Menentukan path input dan output relatif terhadap lokasi script
    raw_path = '../dataset_raw/mushrooms.csv'
    processed_path = 'mushrooms_preprocessing/mushrooms_preprocessed.csv'
    preprocess_data(raw_path, processed_path)