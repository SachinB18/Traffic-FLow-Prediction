"""
PHASE 1.2: Spatial Utilities for METR-LA Dataset
=================================================
Functions for spatial analysis and feature computation:
- Distance-based adjacency
- K-nearest neighbors
- Spatial correlation
- Spatial-temporal feature engineering
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SpatialAnalyzer:
    """Comprehensive spatial analysis for METR-LA sensor network."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize spatial analyzer with unified dataset.
        
        Args:
            data_dir: Path to processed data directory
        """
        self.data_dir = Path(data_dir)
        self._load_data()
    
    def _load_data(self):
        """Load unified dataset files."""
        # Load coordinates (index, sensor_id, lat, lon)
        self.coordinates = np.load(self.data_dir / "sensor_coordinates.npy")
        
        # Load distance matrix
        dist_df = pd.read_csv(self.data_dir / "sensor_distances.csv")
        self.distance_df = dist_df
        
        # Load ID mappings
        with open(self.data_dir / "sensor_id_mappings.json", "r") as f:
            self.id_mapping = json.load(f)
        
        # Load metadata
        with open(self.data_dir / "unified_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        self.n_sensors = len(self.coordinates)
        print(f"✓ Loaded spatial data for {self.n_sensors} sensors")
    
    # ========== DISTANCE OPERATIONS ==========
    
    def compute_pairwise_distances(self) -> np.ndarray:
        """
        Compute pairwise Euclidean distances from coordinates.
        Coordinates are in (lat, lon) format.
        
        Returns:
            dist_matrix: (n_sensors, n_sensors) distance matrix in meters
        """
        # Extract lat/lon (columns 2, 3)
        coords = self.coordinates[:, [2, 3]]  # (207, 2)
        
        # Haversine distance for lat/lon
        # For short distances, Euclidean approximation OK
        # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
        lat = coords[:, 0]
        lon = coords[:, 1]
        
        # Convert to meters (Euclidean approximation for LA area)
        mean_lat = np.mean(lat)
        m_per_deg_lat = 111000  # meters per degree latitude
        m_per_deg_lon = 111000 * np.cos(np.radians(mean_lat))  # meters per degree lon
        
        # Scale coordinates
        scaled_coords = coords.copy()
        scaled_coords[:, 0] *= m_per_deg_lat
        scaled_coords[:, 1] *= m_per_deg_lon
        
        # Compute Euclidean distance
        dist_matrix = cdist(scaled_coords, scaled_coords, metric='euclidean')
        
        return dist_matrix
    
    def build_distance_matrix_sparse(self, threshold: float = 5000.0) -> csr_matrix:
        """
        Build sparse distance matrix with threshold.
        
        Args:
            threshold: Maximum distance (meters) to consider as connected
            
        Returns:
            Sparse matrix of distances (connections only)
        """
        dist_matrix = self.compute_pairwise_distances()
        dist_matrix[dist_matrix > threshold] = 0  # Cut off distant pairs
        
        return csr_matrix(dist_matrix)
    
    # ========== K-NEAREST NEIGHBORS ==========
    
    def get_k_nearest_neighbors(self, sensor_idx: int, k: int = 5) -> Dict:
        """
        Get k nearest neighbors for a sensor.
        
        Args:
            sensor_idx: Sensor index (0-206)
            k: Number of neighbors
            
        Returns:
            dict with neighbor indices, distances, and sensor IDs
        """
        dist_matrix = self.compute_pairwise_distances()
        distances = dist_matrix[sensor_idx]
        
        # Get indices of k smallest distances (excluding self)
        nearest_indices = np.argsort(distances)[1:k+1]
        nearest_distances = distances[nearest_indices]
        
        # Map to sensor IDs
        nearest_ids = self.coordinates[nearest_indices, 1]  # Column 1 is sensor_id
        
        return {
            "sensor_idx": sensor_idx,
            "sensor_id": int(self.coordinates[sensor_idx, 1]),
            "k_nearest_indices": nearest_indices.tolist(),
            "k_nearest_ids": nearest_ids.astype(int).tolist(),
            "distances_m": nearest_distances.tolist(),
        }
    
    def get_all_k_nearest_neighbors(self, k: int = 5) -> Dict:
        """Get k-nearest neighbors for all sensors."""
        all_neighbors = {}
        dist_matrix = self.compute_pairwise_distances()
        
        for i in range(self.n_sensors):
            distances = dist_matrix[i]
            nearest_indices = np.argsort(distances)[1:k+1]
            all_neighbors[str(i)] = nearest_indices.tolist()
        
        return all_neighbors
    
    # ========== SPATIAL ADJACENCY ==========
    
    def build_adjacency_from_distance(self, threshold: float = 5000.0) -> np.ndarray:
        """
        Build adjacency matrix from distance threshold.
        
        Args:
            threshold: Maximum distance to consider adjacent (meters)
            
        Returns:
            Binary adjacency matrix (207, 207)
        """
        dist_matrix = self.compute_pairwise_distances()
        adj_matrix = (dist_matrix > 0) & (dist_matrix <= threshold)
        
        return adj_matrix.astype(np.float32)
    
    def normalize_adjacency(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix (symmetric normalization).
        
        Args:
            adj_matrix: Binary or weighted adjacency matrix
            
        Returns:
            Normalized adjacency matrix D^(-1/2) A D^(-1/2)
        """
        # Compute degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        degrees[degrees == 0] = 1  # Avoid division by zero
        
        # D^(-1/2)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        
        # Normalized: D^(-1/2) A D^(-1/2)
        normalized = d_inv_sqrt @ adj_matrix @ d_inv_sqrt
        
        return normalized
    
    # ========== SPATIAL FEATURES ==========
    
    def compute_spatial_moving_average(self, speed_data: np.ndarray, 
                                      k: int = 5, k_type: str = "distance") -> np.ndarray:
        """
        Compute spatial moving average using k nearest neighbors.
        
        Args:
            speed_data: Traffic speed data (T, N) - timesteps × sensors
            k: Number of neighbors to average
            k_type: "distance" for k-nearest, "threshold" for distance threshold
            
        Returns:
            Spatial MA data (T, N, k+1) - includes self + k neighbors
        """
        T, N = speed_data.shape
        spatial_features = []
        
        # Get k-NN for all sensors
        knn_map = self.get_all_k_nearest_neighbors(k=k)
        
        for t in range(T):
            time_features = []
            for sensor_i in range(N):
                # Get k nearest neighbors
                neighbors = [sensor_i] + knn_map[str(sensor_i)]
                values = speed_data[t, neighbors]
                time_features.append(values)
            
            spatial_features.append(time_features)
        
        # Stack: (T, N, k+1)
        return np.array(spatial_features)
    
    def compute_spatial_variance(self, speed_data: np.ndarray, 
                                k: int = 5) -> np.ndarray:
        """
        Compute spatial variance (spread of speeds among neighbors).
        
        Args:
            speed_data: Traffic speed data (T, N)
            k: Number of neighbors
            
        Returns:
            Spatial variance (T, N)
        """
        T, N = speed_data.shape
        variance = np.zeros((T, N))
        
        knn_map = self.get_all_k_nearest_neighbors(k=k)
        
        for t in range(T):
            for i in range(N):
                neighbors = [i] + knn_map[str(i)]
                neighbor_speeds = speed_data[t, neighbors]
                variance[t, i] = np.var(neighbor_speeds)
        
        return variance
    
    def compute_spatial_correlation(self, speed_data: np.ndarray,
                                   k: int = 5) -> np.ndarray:
        """
        Compute spatial correlation with neighbors.
        
        Args:
            speed_data: Traffic speed data (T, N)
            k: Number of neighbors
            
        Returns:
            Mean correlation with neighbors (N,)
        """
        N = speed_data.shape[1]
        correlations = np.zeros(N)
        
        knn_map = self.get_all_k_nearest_neighbors(k=k)
        
        for i in range(N):
            neighbors = knn_map[str(i)]
            # Compute correlation between sensor i and its neighbors
            correlations[i] = np.mean([
                np.corrcoef(speed_data[:, i], speed_data[:, j])[0, 1]
                for j in neighbors
            ])
        
        return np.nan_to_num(correlations, nan=0.0)
    
    # ========== GRAPH METRICS ==========
    
    def compute_graph_metrics(self, adj_matrix: np.ndarray = None) -> Dict:
        """
        Compute network metrics (degree, clustering, etc).
        
        Args:
            adj_matrix: Adjacency matrix (if None, computed from distances)
            
        Returns:
            Dictionary of metrics
        """
        if adj_matrix is None:
            adj_matrix = self.build_adjacency_from_distance(threshold=5000.0)
        
        # Degree
        degrees = np.sum(adj_matrix, axis=1)
        
        # Clustering coefficient (approximate)
        clustering = []
        for i in range(self.n_sensors):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 1:
                edges_in_neighborhood = np.sum(adj_matrix[np.ix_(neighbors, neighbors)])
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                c_i = edges_in_neighborhood / (2 * possible_edges) if possible_edges > 0 else 0
                clustering.append(c_i)
            else:
                clustering.append(0)
        
        clustering = np.array(clustering)
        
        return {
            "mean_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "min_degree": int(np.min(degrees)),
            "density": float(np.sum(adj_matrix) / (self.n_sensors * (self.n_sensors - 1))),
            "mean_clustering": float(np.mean(clustering)),
            "max_clustering": float(np.max(clustering)),
            "degree_distribution": degrees.tolist(),
        }
    
    # ========== SAVE UTILITIES ==========
    
    def save_spatial_features(self, speed_data: np.ndarray, 
                             output_dir: str = "data/processed"):
        """
        Compute and save all spatial features.
        
        Args:
            speed_data: Traffic speed data (T, N)
            output_dir: Directory to save features
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print("Computing spatial features...")
        
        # 1. K-NN indices
        knn_map = self.get_all_k_nearest_neighbors(k=5)
        with open(out_path / "knn_neighbors.json", "w") as f:
            json.dump(knn_map, f, indent=2)
        print("  ✓ k-NN indices saved")
        
        # 2. Adjacency matrix
        adj_matrix = self.build_adjacency_from_distance(threshold=5000.0)
        np.save(out_path / "adjacency_distance.npy", adj_matrix)
        print("  ✓ Adjacency matrix saved")
        
        # 3. Normalized adjacency
        adj_norm = self.normalize_adjacency(adj_matrix)
        np.save(out_path / "adjacency_normalized.npy", adj_norm)
        print("  ✓ Normalized adjacency saved")
        
        # 4. Spatial moving average
        spatial_ma = self.compute_spatial_moving_average(speed_data, k=5)
        np.save(out_path / "spatial_moving_average.npy", spatial_ma)
        print(f"  ✓ Spatial MA saved: {spatial_ma.shape}")
        
        # 5. Spatial variance
        spatial_var = self.compute_spatial_variance(speed_data, k=5)
        np.save(out_path / "spatial_variance.npy", spatial_var)
        print("  ✓ Spatial variance saved")
        
        # 6. Spatial correlation
        spatial_corr = self.compute_spatial_correlation(speed_data, k=5)
        np.save(out_path / "spatial_correlation.npy", spatial_corr)
        print("  ✓ Spatial correlation saved")
        
        # 7. Graph metrics
        metrics = self.compute_graph_metrics(adj_matrix)
        with open(out_path / "graph_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("  ✓ Graph metrics saved")


def main():
    """Execute Phase 1.2: Spatial utilities creation and feature extraction."""
    import sys
    
    # Adjust path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("PHASE 1.2: SPATIAL UTILITIES")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = SpatialAnalyzer(data_dir="AIML_Traffic_Flow_Prediction/data/processed")
    
    # Test basic operations
    print("\n[1/5] Testing distance computation...")
    dist_matrix = analyzer.compute_pairwise_distances()
    print(f"✓ Distance matrix: {dist_matrix.shape}")
    print(f"  Range: [{np.min(dist_matrix):.1f}, {np.max(dist_matrix):.1f}] meters")
    
    print("\n[2/5] Testing k-nearest neighbors...")
    knn = analyzer.get_k_nearest_neighbors(sensor_idx=0, k=5)
    print(f"✓ Sensor {knn['sensor_id']} neighbors:")
    for idx, dist in zip(knn['k_nearest_indices'], knn['distances_m']):
        print(f"    Sensor {analyzer.coordinates[idx, 1]:.0f}: {dist:.0f}m")
    
    print("\n[3/5] Testing adjacency matrices...")
    adj = analyzer.build_adjacency_from_distance(threshold=5000.0)
    print(f"✓ Binary adjacency: {adj.shape}, edges: {int(np.sum(adj) // 2)}")
    
    adj_norm = analyzer.normalize_adjacency(adj)
    print(f"✓ Normalized adjacency: {adj_norm.shape}")
    
    print("\n[4/5] Testing graph metrics...")
    metrics = analyzer.compute_graph_metrics(adj)
    print(f"✓ Mean degree: {metrics['mean_degree']:.2f}")
    print(f"✓ Network density: {metrics['density']:.4f}")
    print(f"✓ Mean clustering: {metrics['mean_clustering']:.4f}")
    
    print("\n[5/5] Computing spatial features...")
    # Load speed data
    speed_data = np.load("AIML_Traffic_Flow_Prediction/data/processed/speed_data.npy")
    analyzer.save_spatial_features(speed_data, 
                                   output_dir="AIML_Traffic_Flow_Prediction/data/processed")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1.2 COMPLETE!")
    print("=" * 80)
    print("\nSpatial features created:")
    print("  • knn_neighbors.json (5-NN indices for all sensors)")
    print("  • adjacency_distance.npy (binary adjacency, 5km threshold)")
    print("  • adjacency_normalized.npy (normalized D^(-1/2)AD^(-1/2))")
    print("  • spatial_moving_average.npy (T, N, 6) - sensor + 5 neighbors")
    print("  • spatial_variance.npy (T, N) - variance among neighbors")
    print("  • spatial_correlation.npy (N,) - correlation with neighbors")
    print("  • graph_metrics.json (network statistics)")


if __name__ == "__main__":
    main()
