"""
Data utilities for METR-LA traffic flow dataset.
Handles: loading .h5 / .npz, normalization, sliding window sequences,
train/val/test splits, and PyTorch DataLoader creation.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


# â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NUM_SENSORS  = 207          # METR-LA sensor count
T_IN         = 12           # 12 Ã— 5-min = 1-hour look-back
T_OUT        = 1            # single-step prediction (next 5 min)
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.1
# TEST_RATIO = 0.2  (remainder)


# â”€â”€â”€ Loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _read_h5_raw(h5_path: str) -> pd.DataFrame:
    """
    Read METR-LA HDF5 using h5py directly (compatible with all pandas/h5py versions).
    The file stores:
      df/block0_values  : (34272, 207) float64  â€” speed readings
      df/axis0          : (207,) bytes          â€” sensor IDs
      df/axis1          : (34272,) int64        â€” nanosecond timestamps
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        values     = f["df"]["block0_values"][:]          # (T, N)
        sensor_ids = [s.decode("utf-8") for s in f["df"]["axis0"][:]]
        timestamps = f["df"]["axis1"][:]                  # nanoseconds since epoch
    index = pd.to_datetime(timestamps, unit="ns")
    df = pd.DataFrame(values, index=index, columns=sensor_ids)
    return df


def load_metr_la(data_dir: str):
    """
    Load METR-LA raw speed data.
    Supports (case-insensitive search for .h5 / .npz / .csv):
      â€¢ METR-LA.h5   (DataFrame, key='df', index=timestamps, columns=sensor_ids)
      â€¢ metr-la.npz  (array key 'data')
      â€¢ any .csv

    Returns:
        df : pd.DataFrame  shape (34272, 207), values in mph
    """
    # Try common filenames (Kaggle uses uppercase 'METR-LA.h5')
    candidates_h5  = ["METR-LA.h5", "metr-la.h5", "metr_la.h5"]
    candidates_npz = ["METR-LA.npz", "metr-la.npz"]
    candidates_csv = ["METR-LA.csv", "metr-la.csv"]

    for name in candidates_h5:
        h5_path = os.path.join(data_dir, name)
        if os.path.exists(h5_path):
            df = _read_h5_raw(h5_path)
            print(f"[load] Loaded {name} â†’ shape {df.shape}")
            return df

    for name in candidates_npz:
        npz_path = os.path.join(data_dir, name)
        if os.path.exists(npz_path):
            arr = np.load(npz_path)["data"]      # (T, N, F)
            arr = arr[:, :, 0]                    # speed feature
            df  = pd.DataFrame(arr)
            print(f"[load] Loaded {name} â†’ shape {df.shape}")
            return df

    for name in candidates_csv:
        csv_path = os.path.join(data_dir, name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            print(f"[load] Loaded {name} â†’ shape {df.shape}")
            return df

    raise FileNotFoundError(
        f"No METR-LA data file found in '{data_dir}'.\n"
        "Expected one of: METR-LA.h5 | metr-la.h5 | METR-LA.npz | METR-LA.csv"
    )


def load_adj_mx(data_dir: str):
    """
    Load adjacency matrix for METR-LA (207 Ã— 207).
    Kaggle file: adj_METR-LA.pkl â€” structure: [sensor_id_list, id_to_idx_dict, adj_matrix]
    Returns numpy array of shape (207, 207) or None if not found.
    """
    # Try both common filenames
    candidates = ["adj_METR-LA.pkl", "adj_mx.pkl", "adj_metr_la.pkl"]
    path = None
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            path = p
            break

    if path is None:
        print("[load_adj] No adjacency matrix file found â€” spatial features unavailable.")
        return None

    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # Kaggle METR-LA structure: [list_of_sensor_ids, dict_id_to_idx, ndarray(207,207)]
    if isinstance(obj, (list, tuple)):
        # Find the ndarray element (the adjacency matrix)
        for item in obj:
            if isinstance(item, np.ndarray) and item.ndim == 2:
                adj_mx = item
                break
        else:
            adj_mx = np.array(obj[-1])
    elif isinstance(obj, np.ndarray):
        adj_mx = obj
    else:
        adj_mx = np.array(obj)

    print(f"[load_adj] Loaded {os.path.basename(path)} â†’ adjacency matrix shape: {adj_mx.shape}")
    return adj_mx.astype(np.float32)


# â”€â”€â”€ Preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill missing sensor readings."""
    df = df.ffill().bfill()
    remaining = df.isna().sum().sum()
    if remaining:
        df = df.fillna(df.mean())
        print(f"[fill] Filled {remaining} remaining NaN values with column means.")
    return df


def split_data(data: np.ndarray, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Temporal (non-shuffled) split.
    data : shape (T, N)
    Returns: train, val, test â€” each shape (T_i, N)
    """
    T      = len(data)
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)
    train = data[:n_train]
    val   = data[n_train: n_train + n_val]
    test  = data[n_train + n_val:]
    print(f"[split] Train:{len(train)}  Val:{len(val)}  Test:{len(test)}")
    return train, val, test


def normalize(train, val, test):
    """
    Fit MinMaxScaler on training set, apply to all splits.
    Returns: (train_norm, val_norm, test_norm, scaler)
    Each array shape unchanged (T_i, N).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)
    return (
        scaler.transform(train).astype(np.float32),
        scaler.transform(val).astype(np.float32),
        scaler.transform(test).astype(np.float32),
        scaler,
    )


# â”€â”€â”€ Sliding Window â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def create_sequences(data: np.ndarray, t_in: int = T_IN, t_out: int = T_OUT):
    """
    Create (X, y) sliding window sequences.
    data : (T, N)
    X    : (samples, t_in, N)
    y    : (samples, t_out, N)  â€” squeezed to (samples, N) if t_out==1
    """
    X, y = [], []
    for i in range(len(data) - t_in - t_out + 1):
        X.append(data[i: i + t_in])
        y.append(data[i + t_in: i + t_in + t_out])
    X = np.array(X, dtype=np.float32)          # (S, t_in, N)
    y = np.array(y, dtype=np.float32)          # (S, t_out, N)
    if t_out == 1:
        y = y.squeeze(1)                        # (S, N)
    return X, y


# â”€â”€â”€ DataLoaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_loaders(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    batch_size_train=32,
    batch_size_eval=64,
):
    """
    Convert numpy arrays â†’ PyTorch TensorDatasets â†’ DataLoaders.
    """
    def _loader(X, y, bs, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          pin_memory=True, num_workers=0)

    train_loader = _loader(X_train, y_train, batch_size_train, shuffle=True)
    val_loader   = _loader(X_val,   y_val,   batch_size_eval,  shuffle=False)
    test_loader  = _loader(X_test,  y_test,  batch_size_eval,  shuffle=False)

    print(f"[loaders] Train:{len(train_loader.dataset)}  "
          f"Val:{len(val_loader.dataset)}  Test:{len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader


# â”€â”€â”€ Save / Load Processed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_processed(save_dir, X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "processed_data.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "scaler":  scaler,
        }, f)
    print(f"[save] Processed data saved â†’ {path}")


def load_processed(save_dir):
    path = os.path.join(save_dir, "processed_data.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"[load] Loaded processed data from {path}")
    return (d["X_train"], d["y_train"],
            d["X_val"],   d["y_val"],
            d["X_test"],  d["y_test"],
            d["scaler"])


# â”€â”€â”€ Inverse transform â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def inverse_transform(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse MinMax on a (N_samples, N_sensors) array or (N_samples,) 1-sensor.
    """
    shape = arr.shape
    arr2d = arr.reshape(-1, scaler.scale_.shape[0]) if arr.ndim > 1 else arr.reshape(-1, 1)
    return scaler.inverse_transform(arr2d).reshape(shape)

