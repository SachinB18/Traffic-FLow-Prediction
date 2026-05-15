"""
Phase 1.1: Unified METR-LA Dataset Loader
Combines speed data + sensor locations + distances + ID mappings
"""
import h5py
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

class UnifiedMETRLA:
    """Load and merge METR-LA data with spatial information"""
    
    def __init__(self, 
                 h5_file="AIML_Traffic_Flow_Prediction/data/raw/METR-LA.h5",
                 metr_dir="AIML_Traffic_Flow_Prediction/METR-LA/sensor_graph"):
        self.h5_file = h5_file
        self.metr_dir = metr_dir
        self.data = None
        self.coordinates = None
        self.distances = None
        self.sensor_mapping = None
        
    def load_speed_data(self):
        """Load traffic speed data from HDF5"""
        print("[1/4] Loading speed data...")
        with h5py.File(self.h5_file, "r") as f:
            # Data is stored in f['df']['block0_values']
            self.data = f["df"]["block0_values"][()]  # (34272, 207)
        
        print(f"✓ Speed data: {self.data.shape}")
        print(f"  Range: [{self.data.min():.2f}, {self.data.max():.2f}] mph")
        print(f"  Mean: {self.data.mean():.2f} mph")
        print(f"  Missing: {np.isnan(self.data).sum()}")
        return self
    
    def load_sensor_locations(self):
        """Load GPS coordinates from METR-LA folder"""
        print("\n[2/4] Loading sensor locations...")
        locations_file = f"{self.metr_dir}/sensor_locations.csv"
        locations_df = pd.read_csv(locations_file)
        
        print(f"✓ Loaded {len(locations_df)} sensors")
        print(f"  Latitude:  [{locations_df['latitude'].min():.4f}, {locations_df['latitude'].max():.4f}]")
        print(f"  Longitude: [{locations_df['longitude'].min():.4f}, {locations_df['longitude'].max():.4f}]")
        
        # Load mapping from JSON
        mapping_file = f"{self.metr_dir}/adj_mx_mapping.json"
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        sensor_ids = mapping_data['sensor_ids']
        print(f"✓ Loaded mapping for {len(sensor_ids)} sensors")
        
        # Create mapping: index → sensor_id, sensor_id → index
        self.index_to_id = {i: int(sid) for i, sid in enumerate(sensor_ids)}
        self.id_to_index = {v: k for k, v in self.index_to_id.items()}
        
        # Map locations to indices
        locations_df['index'] = locations_df['sensor_id'].map(self.id_to_index)
        locations_df = locations_df.sort_values('index')
        
        self.coordinates = locations_df[['index', 'sensor_id', 'latitude', 'longitude']].values
        
        print(f"✓ Mapped coordinates to indices")
        print(f"  Sample (first 5):\n{locations_df.head(5).to_string()}")
        
        return self
    
    def load_distances(self):
        """Load distance matrix from METR-LA folder"""
        print("\n[3/4] Loading distances...")
        distances_file = f"{self.metr_dir}/distances.csv"
        distances_df = pd.read_csv(distances_file)
        
        print(f"✓ Loaded {len(distances_df):,} distance pairs")
        print(f"  Range: [{distances_df['cost'].min():.1f}, {distances_df['cost'].max():.1f}] meters")
        print(f"  Mean: {distances_df[distances_df['cost'] > 0]['cost'].mean():.1f} meters")
        
        self.distances = distances_df
        return self
    
    def validate_consistency(self):
        """Validate data consistency across all sources"""
        print("\n[4/4] Validating consistency...")
        
        # Check 1: Speed data shape
        n_timesteps, n_sensors = self.data.shape
        assert n_sensors == 207, f"Expected 207 sensors, got {n_sensors}"
        print(f"✓ Speed data: {n_timesteps} timesteps × {n_sensors} sensors")
        
        # Check 2: Coordinates shape
        assert len(self.coordinates) == 207, f"Expected 207 coordinates, got {len(self.coordinates)}"
        print(f"✓ Coordinates: {len(self.coordinates)} sensors")
        
        # Check 3: Mapping consistency
        assert len(self.index_to_id) == 207, f"Expected 207 mappings, got {len(self.index_to_id)}"
        assert len(self.id_to_index) == 207, f"Expected 207 reverse mappings, got {len(self.id_to_index)}"
        print(f"✓ ID mappings: 207 entries (bidirectional)")
        
        # Check 4: Coordinates are sorted by index
        indices = self.coordinates[:, 0]
        assert np.array_equal(indices, np.arange(207)), "Coordinates not sorted by index"
        print(f"✓ Coordinates sorted by index")
        
        print("\n✅ ALL VALIDATION CHECKS PASSED!")
        return self
    
    def get_metadata(self):
        """Return metadata dictionary"""
        metadata = {
            'n_timesteps': self.data.shape[0],
            'n_sensors': self.data.shape[1],
            'time_period': 'Mar 1 - Jun 27, 2012',
            'time_resolution': '5 minutes',
            'speed_range': [float(self.data.min()), float(self.data.max())],
            'speed_mean': float(self.data.mean()),
            'speed_std': float(self.data.std()),
            'missing_values': int(np.isnan(self.data).sum()),
            'coordinate_bounds': {
                'latitude': [float(self.coordinates[:, 2].min()), float(self.coordinates[:, 2].max())],
                'longitude': [float(self.coordinates[:, 3].min()), float(self.coordinates[:, 3].max())],
            },
            'n_distance_pairs': len(self.distances),
            'distance_range': [float(self.distances['cost'].min()), float(self.distances['cost'].max())],
        }
        return metadata
    
    def save(self, output_dir="AIML_Traffic_Flow_Prediction/data/processed"):
        """Save unified dataset and metadata"""
        print(f"\n📁 Saving unified dataset to {output_dir}...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save speed data
        speed_file = f"{output_dir}/speed_data.npy"
        np.save(speed_file, self.data)
        print(f"  ✓ {speed_file} ({self.data.nbytes / 1e6:.1f} MB)")
        
        # Save coordinates
        coords_file = f"{output_dir}/sensor_coordinates.npy"
        np.save(coords_file, self.coordinates)
        print(f"  ✓ {coords_file}")
        
        # Save distances
        distances_file = f"{output_dir}/sensor_distances.csv"
        self.distances.to_csv(distances_file, index=False)
        print(f"  ✓ {distances_file}")
        
        # Save ID mappings as JSON
        mappings = {
            'index_to_id': {str(k): v for k, v in self.index_to_id.items()},
            'id_to_index': {str(k): v for k, v in self.id_to_index.items()},
        }
        mappings_file = f"{output_dir}/sensor_id_mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"  ✓ {mappings_file}")
        
        # Save metadata
        metadata = self.get_metadata()
        metadata_file = f"{output_dir}/unified_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ {metadata_file}")
        
        # Create summary file
        summary = {
            'speed_data': speed_file,
            'coordinates': coords_file,
            'distances': distances_file,
            'mappings': mappings_file,
            'metadata': metadata_file,
        }
        
        print("\n" + "=" * 80)
        print("UNIFIED DATASET SUMMARY")
        print("=" * 80)
        print(f"Speed Data:     {self.data.shape} (timesteps × sensors)")
        print(f"Coordinates:    {len(self.coordinates)} sensors with lat/lon")
        print(f"Distances:      {len(self.distances):,} sensor-pair distances")
        print(f"Sensor IDs:     207 mappings (index ↔ real detector ID)")
        print(f"\nMetadata:")
        print(f"  Time period: {metadata['time_period']}")
        print(f"  Resolution:  {metadata['time_resolution']}")
        print(f"  Speed range: [{metadata['speed_range'][0]:.1f}, {metadata['speed_range'][1]:.1f}] mph")
        print(f"  Geographic:  Lat [{metadata['coordinate_bounds']['latitude'][0]:.4f}, {metadata['coordinate_bounds']['latitude'][1]:.4f}]")
        print(f"               Lon [{metadata['coordinate_bounds']['longitude'][0]:.4f}, {metadata['coordinate_bounds']['longitude'][1]:.4f}]")
        print("=" * 80)
        
        return summary


def main():
    """Execute Phase 1.1: Data Integration"""
    print("=" * 80)
    print("PHASE 1.1: UNIFIED METR-LA DATASET")
    print("=" * 80)
    
    try:
        # Create unified dataset
        loader = UnifiedMETRLA()
        
        # Load all data
        loader.load_speed_data()
        loader.load_sensor_locations()
        loader.load_distances()
        
        # Validate
        loader.validate_consistency()
        
        # Save
        summary = loader.save()
        
        print("\n✅ PHASE 1.1 COMPLETE!")
        print("\nOutput files created:")
        for key, path in summary.items():
            print(f"  • {key}: {path}")
        
        return loader
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    loader = main()
