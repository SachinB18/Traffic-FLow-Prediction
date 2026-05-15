"""
PHASE 4: Model Enhancement - Spatial-Temporal Models
======================================================
Implement enhanced models with spatial attention:
- LSTM + Spatial Attention (SpatialAttention-LSTM)
- GRU + Spatial Attention (SpatialAttention-GRU)
- Graph-based models (proof-of-concept)
- Training pipeline with validation monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt


class SpatialAttention(nn.Module):
    """Multi-head spatial attention mechanism for sensor networks."""
    
    def __init__(self, n_sensors: int, d_model: int, n_heads: int = 8):
        """
        Args:
            n_sensors: Number of sensors (207)
            d_model: Hidden dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            X: (batch, time, n_sensors, d_model) - sensor values
            A: (n_sensors, n_sensors) - adjacency/attention matrix
            
        Returns:
            output: (batch, time, n_sensors, d_model) - attended features
        """
        batch, time, n_sensors, d_model = X.shape
        
        # Project to Q, K, V
        Q = self.W_q(X)  # (batch, time, n_sensors, d_model)
        K = self.W_k(X)
        V = self.W_v(X)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch, time, n_sensors, self.n_heads, self.d_head)
        K = K.reshape(batch, time, n_sensors, self.n_heads, self.d_head)
        V = V.reshape(batch, time, n_sensors, self.n_heads, self.d_head)
        
        # Transpose to (batch, time, n_heads, n_sensors, d_head)
        Q = Q.permute(0, 1, 3, 2, 4)
        K = K.permute(0, 1, 3, 2, 4)
        V = V.permute(0, 1, 3, 2, 4)
        
        # Attention scores: Q @ K^T / sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        # Apply spatial adjacency mask
        if A is not None:
            A_mask = A.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, n_sensors, n_sensors)
            scores = scores * A_mask
        
        # Softmax normalization
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, time, n_heads, n_sensors, d_head)
        
        # Reshape back
        context = context.permute(0, 1, 3, 2, 4).contiguous()
        context = context.reshape(batch, time, n_sensors, d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output


class SpatialAttentionLSTM(nn.Module):
    """LSTM enhanced with spatial attention mechanism."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_sensors: int,
                 n_layers: int = 2,
                 attention_heads: int = 4,
                 attention_matrix: Optional[np.ndarray] = None,
                 dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimension (1)
            hidden_dim: LSTM hidden dimension
            output_dim: Output dimension (1)
            n_sensors: Number of sensors
            n_layers: Number of LSTM layers
            attention_heads: Heads for spatial attention
            attention_matrix: (n_sensors, n_sensors) adjacency matrix
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_sensors = n_sensors
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Spatial attention
        self.spatial_attn = SpatialAttention(n_sensors, hidden_dim, attention_heads)
        
        # Register attention matrix as buffer
        if attention_matrix is not None:
            A = torch.from_numpy(attention_matrix).float()
            self.register_buffer('attention_matrix', A)
        else:
            self.register_buffer('attention_matrix', None)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            X: (batch, seq_len, n_sensors, input_dim)
            
        Returns:
            output: (batch, 1, n_sensors, output_dim)
        """
        batch, seq_len, n_sensors, input_dim = X.shape
        
        # Process each sensor's sequence through LSTM
        X_flat = X.reshape(batch * n_sensors, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(X_flat)  # (batch*n_sensors, seq_len, hidden_dim)
        lstm_out = lstm_out.reshape(batch, seq_len, n_sensors, self.hidden_dim)
        
        # Apply spatial attention
        attn_out = self.spatial_attn(lstm_out, self.attention_matrix)
        
        # Use final timestep
        final = attn_out[:, -1, :, :]  # (batch, n_sensors, hidden_dim)
        
        # Output projection
        output = self.output_layer(final)  # (batch, n_sensors, output_dim)
        output = output.unsqueeze(1)  # (batch, 1, n_sensors, output_dim)
        
        return output


class SpatialAttentionGRU(nn.Module):
    """GRU enhanced with spatial attention mechanism."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_sensors: int,
                 n_layers: int = 2,
                 attention_heads: int = 4,
                 attention_matrix: Optional[np.ndarray] = None,
                 dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimension (1)
            hidden_dim: GRU hidden dimension
            output_dim: Output dimension (1)
            n_sensors: Number of sensors
            n_layers: Number of GRU layers
            attention_heads: Heads for spatial attention
            attention_matrix: (n_sensors, n_sensors) adjacency matrix
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_sensors = n_sensors
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Spatial attention
        self.spatial_attn = SpatialAttention(n_sensors, hidden_dim, attention_heads)
        
        # Register attention matrix as buffer
        if attention_matrix is not None:
            A = torch.from_numpy(attention_matrix).float()
            self.register_buffer('attention_matrix', A)
        else:
            self.register_buffer('attention_matrix', None)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            X: (batch, seq_len, n_sensors, input_dim)
            
        Returns:
            output: (batch, 1, n_sensors, output_dim)
        """
        batch, seq_len, n_sensors, input_dim = X.shape
        
        # Process each sensor's sequence through GRU
        X_flat = X.reshape(batch * n_sensors, seq_len, input_dim)
        gru_out, _ = self.gru(X_flat)  # (batch*n_sensors, seq_len, hidden_dim)
        gru_out = gru_out.reshape(batch, seq_len, n_sensors, self.hidden_dim)
        
        # Apply spatial attention
        attn_out = self.spatial_attn(gru_out, self.attention_matrix)
        
        # Use final timestep
        final = attn_out[:, -1, :, :]  # (batch, n_sensors, hidden_dim)
        
        # Output projection
        output = self.output_layer(final)  # (batch, n_sensors, output_dim)
        output = output.unsqueeze(1)  # (batch, 1, n_sensors, output_dim)
        
        return output


class ModelTrainer:
    """Training pipeline for spatial-temporal models."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001):
        """
        Args:
            model: PyTorch model
            device: torch.device
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            pred = self.model(X)
            loss = self.criterion(pred, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                loss = self.criterion(pred, y)
                mae = torch.mean(torch.abs(pred - y))
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        return val_loss / len(val_loader), val_mae / len(val_loader)
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50,
            patience: int = 10) -> Dict:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            
            # Print every epoch
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MAE: {val_mae:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        print("=" * 60)
        print("Training complete!")
        
        return self.history


def load_enhanced_dataset(data_dir: str = "data/processed",
                         batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Load enhanced dataset with spatial features.
    
    Returns:
        train_loader, val_loader, test_loader, attention_matrix
    """
    data_dir = Path(data_dir)
    
    # Load sequences
    X_train = torch.from_numpy(np.load(data_dir / 'X_train.npy')).float()
    y_train = torch.from_numpy(np.load(data_dir / 'y_train.npy')).float()
    X_val = torch.from_numpy(np.load(data_dir / 'X_val.npy')).float()
    y_val = torch.from_numpy(np.load(data_dir / 'y_val.npy')).float()
    X_test = torch.from_numpy(np.load(data_dir / 'X_test.npy')).float()
    y_test = torch.from_numpy(np.load(data_dir / 'y_test.npy')).float()
    
    # Load attention matrix
    attention = np.load(data_dir / 'attention_correlation.npy')
    
    # Create data loaders
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Dataset loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Attention: {attention.shape}")
    
    return train_loader, val_loader, test_loader, attention


def main():
    """Execute Phase 4: Model Enhancement."""
    import sys
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("PHASE 4: SPATIAL-TEMPORAL MODEL ENHANCEMENT")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load data
    print("\n[1/3] Loading enhanced dataset...")
    train_loader, val_loader, test_loader, attn_matrix = load_enhanced_dataset(
        data_dir="AIML_Traffic_Flow_Prediction/data/processed",
        batch_size=32
    )
    
    # Build model
    print("\n[2/3] Building SpatialAttention-LSTM model...")
    model = SpatialAttentionLSTM(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        n_sensors=207,
        n_layers=2,
        attention_heads=4,
        attention_matrix=attn_matrix,
        dropout=0.2
    )
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\n[3/3] Training model...")
    trainer = ModelTrainer(model, device, learning_rate=0.001)
    history = trainer.fit(train_loader, val_loader, epochs=50, patience=10)
    
    # Save results
    results_dir = Path("AIML_Traffic_Flow_Prediction/results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training history
    with open(results_dir / 'training_history_phase4.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), results_dir / 'spatial_attention_lstm_best.pth')
    
    print("\n" + "=" * 80)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved:")
    print(f"  • Model: spatial_attention_lstm_best.pth")
    print(f"  • History: training_history_phase4.json")


if __name__ == "__main__":
    main()
