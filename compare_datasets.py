"""
Inspect HDF5 file structure and compare datasets
"""
import h5py
import numpy as np
import pandas as pd

print("=" * 80)
print("COMPARING METR-LA DATASETS")
print("=" * 80)

# ============================================================================
# PART 1: Inspect Current Local Dataset
# ============================================================================
print("\n[1] INSPECTING LOCAL HDF5 FILE...")
print("-" * 80)

DATA_DIR = "AIML_Traffic_Flow_Prediction/data/raw"
h5_file = f"{DATA_DIR}/METR-LA.h5"

try:
    with h5py.File(h5_file, "r") as f:
        print(f"✓ Opened: {h5_file}")
        print(f"  Available keys in HDF5:")
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"    '{key}': Dataset, shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"    '{key}': Group")
                for subkey in item.keys():
                    subitem = item[subkey]
                    print(f"      - '{subkey}': shape={subitem.shape if hasattr(subitem, 'shape') else 'N/A'}")
        
        # Try to load the data with correct key
        for key in f.keys():
            if 'speed' in key.lower() or 'data' in key.lower() or key in ['x', 'y', 'data', 'traffic']:
                print(f"\n✓ Loading key '{key}'...")
                data_local = f[key][()]
                print(f"  Shape: {data_local.shape}")
                print(f"  Data type: {data_local.dtype}")
                print(f"  Value range: [{data_local.min():.2f}, {data_local.max():.2f}]")
                print(f"  Mean: {data_local.mean():.2f}, Std: {data_local.std():.2f}")
                print(f"  Missing values: {np.isnan(data_local).sum()}")
                print(f"  First 3x5 values:\n{data_local[:3, :5]}")
                break
                
except Exception as e:
    print(f"✗ Error: {e}")

# Load adjacency matrix
print("\n" + "-" * 80)
print("CHECKING ADJACENCY MATRIX...")
print("-" * 80)

adj_file = f"{DATA_DIR}/../processed/adj_mx.npy"
try:
    adj_local = np.load(adj_file)
    print(f"✓ Loaded: {adj_file}")
    print(f"  Shape: {adj_local.shape}")
    print(f"  Non-zero edges: {np.count_nonzero(adj_local)}")
    print(f"  Density: {np.count_nonzero(adj_local) / (adj_local.shape[0] * adj_local.shape[1]) * 100:.2f}%")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# PART 2: Attempt HF Dataset
# ============================================================================
print("\n" + "=" * 80)
print("[2] ATTEMPTING HUGGING FACE DATASET...")
print("-" * 80)

try:
    from datasets import load_dataset
    print("✓ Importing Hugging Face datasets library...")
    print("  Downloading witgaw/METR-LA...")
    ds_hf = load_dataset("witgaw/METR-LA", trust_remote_code=True)
    print(f"✓ Success!")
    print(f"  Dataset: {ds_hf}")
except ImportError:
    print("✗ datasets library not installed")
    print("  Install with: pip install datasets")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
