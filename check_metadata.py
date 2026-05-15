"""
Check for location/metadata in HDF5 file
"""
import h5py
import numpy as np

print("=" * 80)
print("CHECKING FOR LOCATION & METADATA IN HDF5 FILE")
print("=" * 80)

h5_file = "AIML_Traffic_Flow_Prediction/data/raw/METR-LA.h5"

with h5py.File(h5_file, "r") as f:
    print("\n[1] ALL GROUPS AND DATASETS IN HDF5:")
    print("-" * 80)
    
    def print_structure(name, obj):
        """Recursively print HDF5 structure"""
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"  Dataset: '{name}'")
            print(f"    Shape: {shape}, Type: {dtype}")
            # Show first few values if small enough
            if obj.size < 20:
                print(f"    Values: {obj[()]}")
        elif isinstance(obj, h5py.Group):
            print(f"  Group: '{name}'")
    
    f.visititems(print_structure)

print("\n" + "=" * 80)
print("[2] CHECKING FOR COMMON LOCATION ATTRIBUTES:")
print("-" * 80)

with h5py.File(h5_file, "r") as f:
    # Check for various possible location data
    possible_keys = [
        'coordinates', 'coords', 'location', 'locations', 'lat', 'lon', 
        'latitude', 'longitude', 'distance', 'distances', 'geo', 'geometry',
        'sensor_info', 'metadata', 'info', 'attributes'
    ]
    
    found_keys = []
    for key in possible_keys:
        if key in f:
            found_keys.append(key)
            print(f"✓ Found: '{key}'")
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"  Shape: {item.shape}, Type: {item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  Group with keys: {list(item.keys())}")
    
    if not found_keys:
        print("✗ No location/metadata found in HDF5 file")

print("\n" + "=" * 80)
print("[3] CHECKING ADJACENCY MATRIX INSIGHTS:")
print("-" * 80)

adj = np.load("AIML_Traffic_Flow_Prediction/data/raw/../processed/adj_mx.npy")
print(f"Adjacency matrix shape: {adj.shape}")
print(f"Non-zero values: {np.count_nonzero(adj)}")
print(f"Min non-zero value: {adj[adj > 0].min():.4f}")
print(f"Max value: {adj.max():.4f}")
print(f"Mean of non-zero values: {adj[adj > 0].mean():.4f}")
print("\nNote: Non-zero values in adjacency matrix might represent distances!")
print("If values are small (< 10), they likely represent normalized distances")

print("\n" + "=" * 80)
print("[4] RECOMMENDATION:")
print("-" * 80)
print("""
The current dataset is MISSING:
  ❌ Sensor coordinates (latitude/longitude)
  ❌ Real distances between sensors
  
OPTIONS TO GET THIS DATA:

1. PUBLIC SOURCES (METR-LA is well-documented):
   - METR-LA detector locations available from LA Metro
   - Coordinates available in research papers
   - GitHub repos with sensor coordinates
   
2. HUGGING FACE DATASET:
   - witgaw/METR-LA might include sensor metadata
   - Check if location data is in that version
   
3. DERIVE FROM ADJACENCY MATRIX:
   - If values represent distances, use them directly
   - Create synthetic coordinates from network topology
   
4. EXTERNAL DATA:
   - Match sensor IDs with public transportation databases
   - Use freeway corridor information from LA Metro
""")

print("=" * 80)
