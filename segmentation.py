import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(path: str):
    df = pd.read_csv(path)
    return df


# -----------------------------
# 2. Preprocess Data
# -----------------------------
def preprocess(df: pd.DataFrame):
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])

    # Ensure TotalAmount exists
    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df


# -----------------------------
# 3. Create RFM Features
# -----------------------------
def create_rfm(df: pd.DataFrame):
    snapshot_date = df["OrderDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "OrderDate": lambda x: (snapshot_date - x.max()).days,
        "OrderID": "count",
        "TotalAmount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm.reset_index(inplace=True)

    return rfm


# -----------------------------
# 4. Scale Features
# -----------------------------
def scale_features(rfm: pd.DataFrame):
    scaler = StandardScaler()
    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaled = scaler.fit_transform(features)
    return scaled, scaler


# -----------------------------
# 5. Train KMeans Model
# -----------------------------
def train_kmeans(scaled_data, k=3):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)
    return model, labels


# -----------------------------
# 6. Run Full Pipeline
# -----------------------------
def run_pipeline(input_path, output_path="clusters.csv", k=3):

    # Load + preprocess
    df = load_data(input_path)
    df = preprocess(df)

    # RFM
    rfm = create_rfm(df)

    # Scaling
    scaled, scaler = scale_features(rfm)

    # Clustering
    model, labels = train_kmeans(scaled, k)

    rfm["Cluster"] = labels

    # Save output
    rfm.to_csv(output_path, index=False)

    print("Pipeline executed successfully!")
    print(f"Output saved to {output_path}")

    return rfm, model, scaler


# -----------------------------
# 7. Main Execution
# -----------------------------
if __name__ == "__main__":
    input_file = "data/customers.csv"
    output_file = "outputs/clusters.csv"

    run_pipeline(input_file, output_file, k=3)