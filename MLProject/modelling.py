import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# --- PATH AMAN (jalan di lokal & GitHub Actions) ---
BASE_DIR = os.path.dirname(__file__)  # folder tempat modelling.py berada (MLProject/)
DATA_PATH = os.path.join(BASE_DIR, "DataPenjualanMotor_preprocessing.csv")

TARGET_COL = "HARGA JUAL"
EXPERIMENT_NAME = "PenjualanMotor_RF"


def load_data(path: str) -> pd.DataFrame:
    print(f"Membaca data dari: {path}")

    if not os.path.exists(path):
        # debugging biar jelas kalau masih salah lokasi
        print("File tidak ditemukan. Isi BASE_DIR:", os.listdir(BASE_DIR))
        print("Isi root repo:", os.listdir("."))
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {path}")

    df = pd.read_csv(path)

    # Pastikan kolom tanggal dalam format datetime (jika masih string)
    if "TANGGAL" in df.columns and df["TANGGAL"].dtype == "object":
        df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")

    print("Shape data:", df.shape)
    return df
