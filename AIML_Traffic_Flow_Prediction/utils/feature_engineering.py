"""
PHASE 3: Feature Engineering for METR-LA Dataset
===================================================
Create spatial-temporal features for enhanced model training:
- Temporal features (hour, day of week, holidays)
- Spatial features (k-NN aggregations, graph-based)
- Lag features (multi-scale temporal patterns)
- Normalization and sequence generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import json
from typing import Tuple, Dict, List, Optional


class SpatialTemporalFeatureEngineer:
    """Create spatial-temporal features for traffic prediction."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """Initialize feature engineer with unified dataset."""
        self.data_dir = Path(data_dir)
        self._load_data()
        
    def _load_data(self):
        """Load all necessary data files."""
        self.speed_data = np.load(self.data_dir / 'speed_data.npy')  # (T, N)
        self.coordinates = np.load(self.data_dir / 'sensor_coordinates.npy')
        self.spatial_ma = np.load(self.data_dir / 'spatial_moving_average.npy')
        self.spatial_var = np.load(self.data_dir / 'spatial_variance.npy')
        self.spatial_corr = np.load(self.data_dir / 'spatial_correlation.npy')
        self.adjacency_norm = np.load(self.data_dir / 'adjacency_normalized.npy')
        
        with open(self.data_dir / 'unified_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.n_timesteps = self.speed_data.shape[0]  # First dimension
        self.n_sensors = self.speed_data.shape[1]    # Second dimension
        
        print(f"✓ Data loaded: {self.n_timesteps} timesteps × {self.n_sensors} sensors")
    
    # ========== TEMPORAL FEATURES ==========
    
    def create_temporal_features(self) -> np.ndarray:
        """
        Create temporal cyclical features (hour, day of week).
        
        Returns:
            temporal_features: (T, 4) - sin(hour), cos(hour), sin(dow), cos(dow)
        """
        # Time period: Mar 1 - Jun 27, 2012, 5-minute resolution
        start_date = datetime(2012, 3, 1, 0, 0)
        
        temporal_features = []
        
        for t in range(self.n_timesteps):
            # Current timestamp
            current_time = start_date + timedelta(minutes=5*t)
            
            # Hour of day (0-23)
            hour = current_time.hour + current_time.minute / 60.0
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Day of week (0-6, Monday-Sunday)
            dow = current_time.weekday()
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)
            
            temporal_features.append([hour_sin, hour_cos, dow_sin, dow_cos])
        
        return np.array(temporal_features)  # (T, 4)
    
    def create_lag_features(self, lags: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Create lagged speed features at multiple time scales.
        
        Args:
            lags: List of lag steps (e.g., [1, 12, 288] for 5min, 1hr, 24hr)
            
        Returns:
            Dictionary of lag features {lag: (T-max_lag, N)}
        """
        if lags is None:
            lags = [1, 12, 288]  # 5-min, 1-hour, 24-hour
        
        lag_features = {}
        max_lag = max(lags)
        
        for lag in lags:
            # Shift speed data backward by 'lag' timesteps
            lagged = np.roll(self.speed_data, lag, axis=0)
            lagged[:lag, :] = np.nan  # Mark initial values as NaN
            lag_features[lag] = lagged
        
        return lag_features
    
    # ========== SPATIAL FEATURES ==========
    
    def create_graph_convolution_features(self, k_hops: int = 2) -> np.ndarray:
        """
        Create graph convolution features: A^k @ speed_data.T
        Captures information from k-hop neighbors.
        
        Args:
            k_hops: Number of hops in graph convolution
            
        Returns:
            gc_features: (T, N, k_hops+1) - speed + 1-hop + 2-hop, etc.
        """
        A = self.adjacency_norm  # Normalized adjacency (207, 207)
        X = self.speed_data.T  # (207, T)
        
        gc_features = []
        A_pow = np.eye(A.shape[0])  # A^0 = I
        
        for hop in range(k_hops + 1):
            # Current hop: A^hop @ X
            feature_hop = (A_pow @ X).T  # (T, 207)
            gc_features.append(feature_hop)
            
            # Update for next hop
            A_pow = A_pow @ A  # A^(hop+1)
        
        return np.stack(gc_features, axis=2)  # (T, N, k_hops+1)
    
    def create_attention_weights(self, method: str = 'distance') -> np.ndarray:
        """
        Create attention weights for spatial modeling.
        
        Args:
            method: 'distance' (from adjacency) or 'correlation' (correlation-based)
            
        Returns:
            attention: (N, N) - weight matrix for spatial attention
        """
        if method == 'distance':
            # Use pre-computed adjacency distance matrix as attention
            # Normalize rows to sum to 1
            attention = self.adjacency_norm.copy()
            # Ensure non-negative and normalized
            attention = np.maximum(attention, 0)
            attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-8)
            
        elif method == 'correlation':
            # Compute sensor-wise correlations (not timestep-wise)
            # (T, N).T @ (T, N) / T = (N, N)
            speed_corr = np.corrcoef(self.speed_data.T, rowvar=True)  # (207, 207)
            # Ensure non-negative and normalized
            attention = np.maximum(speed_corr, 0)
            attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-8)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return attention
    
    # ========== NORMALIZATION & SEQUENCES ==========
    
    def normalize_data(self, data: np.ndarray, 
                      fit_on: Optional[np.ndarray] = None) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Normalize data using MinMaxScaler (0-1 range).
        
        Args:
            data: Data to normalize
            fit_on: Data to fit scaler on (if None, use 'data')
            
        Returns:
            normalized_data, scaler
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        if fit_on is not None:
            # Fit on one dataset, transform another (prevents leakage)
            scaler.fit(fit_on.reshape(-1, 1) if fit_on.ndim == 1 else fit_on)
        
        normalized = scaler.fit_transform(
            data.reshape(-1, 1) if data.ndim == 1 else data
        )
        
        return normalized, scaler
    
    def create_sequences(self, data: np.ndarray, 
                        seq_len_in: int = 12, 
                        seq_len_out: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for supervised learning.
        
        Args:
            data: Input data (T, N) or (T, N, F)
            seq_len_in: Lookback window
            seq_len_out: Lookahead window
            
        Returns:
            X: (num_sequences, seq_len_in, N) or (num_sequences, seq_len_in, N, F)
            y: (num_sequences, seq_len_out, N)
        """
        X, y = [], []
        
        for i in range(len(data) - seq_len_in - seq_len_out + 1):
            X.append(data[i:i+seq_len_in])
            y.append(data[i+seq_len_in:i+seq_len_in+seq_len_out])
        
        return np.array(X), np.array(y)
    
    def split_train_val_test(self, data: np.ndarray,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test chronologically (no shuffling).
        
        Args:
            data: Input data (T, ...)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (rest goes to test)
            
        Returns:
            train, val, test
        """
        n = len(data)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        return data[:train_idx], data[train_idx:val_idx], data[val_idx:]
    
    # ========== COMPREHENSIVE FEATURE PIPELINE ==========
    
    def create_enhanced_dataset(self, 
                               seq_len_in: int = 12,
                               seq_len_out: int = 1,
                               normalize: bool = True) -> Dict:
        """
        Create complete enhanced dataset with all features.
        
        Args:
            seq_len_in: Lookback window
            seq_len_out: Lookahead window
            normalize: Whether to normalize speed data
            
        Returns:
            Dictionary with train/val/test data and metadata
        """
        print("Creating enhanced spatial-temporal dataset...")
        
        # 1. Temporal features
        print("[1/6] Temporal features...")
        temporal_feat = self.create_temporal_features()
        print(f"  ✓ Shape: {temporal_feat.shape}")
        
        # 2. Lag features
        print("[2/6] Lag features...")
        lag_feats = self.create_lag_features()
        print(f"  ✓ Lags: {list(lag_feats.keys())}")
        
        # 3. Graph convolution features
        print("[3/6] Graph convolution features...")
        gc_feat = self.create_graph_convolution_features(k_hops=2)
        print(f"  ✓ Shape: {gc_feat.shape}")
        
        # 4. Attention weights
        print("[4/6] Attention weights...")
        attn_dist = self.create_attention_weights(method='distance')
        attn_corr = self.create_attention_weights(method='correlation')
        print(f"  ✓ Distance attention: {attn_dist.shape}")
        print(f"  ✓ Correlation attention: {attn_corr.shape}")
        
        # 5. Normalization
        print("[5/6] Normalization...")
        if normalize:
            # Split first to prevent leakage
            train_data, val_data, test_data = self.split_train_val_test(self.speed_data)
            speed_norm, scaler = self.normalize_data(
                np.vstack([train_data, val_data, test_data]),
                fit_on=train_data
            )
        else:
            speed_norm = self.speed_data
            scaler = None
        
        print(f"  ✓ Speed normalized to [0, 1]")
        
        # 6. Sequence generation
        print("[6/6] Sequence generation...")
        X_seq, y_seq = self.create_sequences(
            speed_norm.T.reshape(self.n_timesteps, self.n_sensors, 1),
            seq_len_in=seq_len_in,
            seq_len_out=seq_len_out
        )
        
        # Split sequences
        X_train, X_val, X_test = self.split_train_val_test(X_seq)
        y_train, y_val, y_test = self.split_train_val_test(y_seq)
        
        print(f"  ✓ Sequences created:")
        print(f"    Train: X {X_train.shape}, y {y_train.shape}")
        print(f"    Val:   X {X_val.shape}, y {y_val.shape}")
        print(f"    Test:  X {X_test.shape}, y {y_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'temporal_features': temporal_feat,
            'lag_features': lag_feats,
            'gc_features': gc_feat,
            'attention_distance': attn_dist,
            'attention_correlation': attn_corr,
            'spatial_ma': self.spatial_ma,
            'spatial_variance': self.spatial_var,
            'spatial_correlation': self.spatial_corr,
            'adjacency_normalized': self.adjacency_norm,
            'scaler': scaler,
            'coordinates': self.coordinates,
            'metadata': self.metadata,
        }
    
    def save_enhanced_dataset(self, dataset: Dict, output_dir: str = "data/processed"):
        """Save enhanced dataset files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving enhanced dataset to {out_path}...")
        
        # Save sequences
        np.save(out_path / 'X_train.npy', dataset['X_train'])
        np.save(out_path / 'y_train.npy', dataset['y_train'])
        np.save(out_path / 'X_val.npy', dataset['X_val'])
        np.save(out_path / 'y_val.npy', dataset['y_val'])
        np.save(out_path / 'X_test.npy', dataset['X_test'])
        np.save(out_path / 'y_test.npy', dataset['y_test'])
        
        # Save features
        np.save(out_path / 'temporal_features.npy', dataset['temporal_features'])
        np.save(out_path / 'gc_features.npy', dataset['gc_features'])
        np.save(out_path / 'attention_distance.npy', dataset['attention_distance'])
        np.save(out_path / 'attention_correlation.npy', dataset['attention_correlation'])
        
        print("✓ All files saved successfully")


def main():
    """Execute Phase 3: Feature Engineering."""
    import sys
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("PHASE 3: SPATIAL-TEMPORAL FEATURE ENGINEERING")
    print("=" * 80)
    
    # Initialize engineer
    engineer = SpatialTemporalFeatureEngineer(
        data_dir="AIML_Traffic_Flow_Prediction/data/processed"
    )
    
    # Create enhanced dataset
    dataset = engineer.create_enhanced_dataset(
        seq_len_in=12,
        seq_len_out=1,
        normalize=True
    )
    
    # Save dataset
    engineer.save_enhanced_dataset(
        dataset,
        output_dir="AIML_Traffic_Flow_Prediction/data/processed"
    )
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3 COMPLETE!")
    print("=" * 80)
    print("\nFeature engineering outputs:")
    print("  • X_train, y_train: Training sequences")
    print("  • X_val, y_val: Validation sequences")
    print("  • X_test, y_test: Test sequences")
    print("  • temporal_features: Cyclical hour/day features")
    print("  • gc_features: Graph convolution (0, 1, 2 hops)")
    print("  • attention_distance: Distance-based attention weights")
    print("  • attention_correlation: Correlation-based attention weights")


if __name__ == "__main__":
    main()
