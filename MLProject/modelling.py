import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


DATA_PATH = "DataPenjualanMotor_preprocessing.csv"
TARGET_COL = "HARGA JUAL"
EXPERIMENT_NAME = "PenjualanMotor_RF"


def load_data(path: str) -> pd.DataFrame:
    print(f"Membaca data dari: {path}")
    df = pd.read_csv(path)

    # Pastikan kolom tanggal dalam format datetime (jika masih string)
    if "TANGGAL" in df.columns:
        if df["TANGGAL"].dtype == "object":
            df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")

    print("Shape data:", df.shape)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Membuat fitur X dan target y dari dataframe.
    Menambahkan fitur tahun & bulan, dan menghapus kolom TANGGAL dari X.
    """
    if "TANGGAL" in df.columns:
        df["TAHUN"] = df["TANGGAL"].dt.year
        df["BULAN"] = df["TANGGAL"].dt.month
        X = df.drop(columns=[TARGET_COL, "TANGGAL"])
    else:
        X = df.drop(columns=[TARGET_COL])

    y = df[TARGET_COL]

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    return X, y


def build_pipeline(X):
    """
    Membangun pipeline:
    - OneHotEncoder untuk fitur kategorikal
    - passthrough untuk numerik
    - RandomForestRegressor sebagai model
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("Fitur numerik :", numeric_features)
    print("Fitur kategori:", categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf_model),
        ]
    )

    return model


def train_and_log():
    """
    Melatih model dan mencatat hasilnya ke MLflow.
    Menggunakan autolog + manual logging untuk RMSE & R2.
    """
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Siapkan fitur & target
    X, y = prepare_features(df)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Bangun pipeline model
    model = build_pipeline(X)

    # 5. Set experiment MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 6. Aktifkan autolog (syarat Basic)
    mlflow.sklearn.autolog()

    # 7. Mulai run MLflow
    with mlflow.start_run(run_name="RandomForest_PenjualanMotor"):
        # Train
        model.fit(X_train, y_train)

        # Prediksi di test set
        y_pred = model.predict(X_test)

        # sklearn versi lama belum punya argumen squared=False,
        # jadi kita hitung RMSE dari MSE secara manual
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"R2  : {r2:.4f}")

        # Log metrik manual (tambahan di luar autolog)
        mlflow.log_metric("rmse_manual", rmse)
        mlflow.log_metric("r2_manual", r2)

        # Simpan model sebagai artifact tambahan
        mlflow.sklearn.log_model(model, "model_manual")

    print("Training selesai. Cek MLflow UI untuk melihat hasil run.")


if __name__ == "__main__":
    train_and_log()
