# 📋 Project Enhancement Plan: METR-LA Traffic Prediction with Spatial Integration

## Executive Summary
Integrate real sensor location data from METR-LA folder to enhance current traffic prediction models from basic time-series to spatial-temporal architectures. This enables publication-ready research with complete dataset documentation.

---

## Phase 1: Data Integration (Foundation)
**Duration:** 1-2 hours | **Effort:** Low

### 1.1 Unify Datasets
**Goal:** Create single source of truth combining speed data + locations + distances

Tasks:
- [ ] Create `data/processed/unified_metr_la.py` - dataset class
  - Load current speed data (34,272 × 207)
  - Load sensor locations from METR-LA folder
  - Create mapping: index (0-206) ↔ sensor_id (e.g., 773869)
  - Merge coordinates into dataset
  - Save as `unified_data.pkl`

- [ ] Validation
  - Verify 207 sensors match exactly
  - Check coordinate ranges: lat [34.04, 34.22], lon [-118.54, -118.18]
  - Confirm adjacency matrix (1,722 edges) matches distance pairs

**Output Files:**
- `data/processed/unified_data.pkl` - Combined dataset
- `data/processed/sensor_metadata.json` - ID mappings
- `data/processed/coordinate_mapping.csv` - Index↔ID↔Lat/Lon

### 1.2 Create Enhanced Data Utilities
**Goal:** Reusable functions for spatial features

Tasks:
- [ ] Create `utils/spatial_utils.py`
  ```python
  - compute_distance_matrix(coordinates) → (207, 207)
  - get_k_nearest_neighbors(sensor_idx, k=5) → indices + distances
  - normalize_distances(dist_matrix) → weighted adjacency
  - compute_spatial_features(data, coordinates) → features
  ```

- [ ] Create `utils/visualization_utils.py`
  ```python
  - plot_sensor_network(coordinates, adjacency) → map visualization
  - plot_sensor_locations(coordinates) → scatter on LA map
  - animate_traffic_flow(data, coordinates, timestamps) → video
  ```

**Output Files:**
- `utils/spatial_utils.py` (200 lines)
- `utils/visualization_utils.py` (300 lines)

---

## Phase 2: Enhanced EDA with Location Data
**Duration:** 2-3 hours | **Effort:** Medium

### 2.1 Spatial Distribution Analysis
**Goal:** Show sensor coverage of LA freeway network

Tasks:
- [ ] Create `notebooks/02_Spatial_EDA.ipynb`
  
  **Section 1: Sensor Network Visualization**
  - Plot 207 sensors on LA map with coordinates
  - Color by mean traffic speed
  - Show network connectivity (1,722 edges)
  - Highlight highway corridors (I-10, I-101, I-405)
  
  **Section 2: Spatial Clustering**
  - K-means clustering (k=5) → freeway segments
  - Analyze speed patterns by cluster
  - Find bottleneck areas
  
  **Section 3: Distance-based Correlation**
  - Correlation vs. distance scatter plot
  - Show nearby sensors are correlated
  - Justify spatial dependencies in models
  
  **Section 4: Geographic Heatmaps**
  - Mean speed heatmap on LA map
  - Volatility (std dev) heatmap
  - Peak congestion zones

**Output Files:**
- `results/spatial_eda_network.png` - Sensor network map
- `results/spatial_eda_clusters.png` - Highway segments
- `results/spatial_eda_correlation.png` - Distance correlation
- `results/spatial_eda_heatmaps.png` - Speed/volatility heatmaps
- `results/sensor_locations.html` - Interactive Folium map

### 2.2 Update Main EDA Notebook
**Goal:** Incorporate location insights into Section 1 (Dataset Fundamentals)

Tasks:
- [ ] Add to `001_Data_Loading_and_EDA.ipynb`
  - Section 3A: Map showing all 207 sensors
  - Section 3B: Geographic coverage metrics
  - Section 3C: Spatial homogeneity analysis

---

## Phase 3: Spatial Feature Engineering
**Duration:** 2-3 hours | **Effort:** Medium

### 3.1 Create Spatial Features
**Goal:** Enrich dataset with location-based features

Tasks:
- [ ] Create `notebooks/03_Feature_Engineering.ipynb`

  **Spatial Features:**
  - [ ] Distance to nearest k neighbors (k=1,5,10)
  - [ ] Spatial smoothness (gradient between adjacent sensors)
  - [ ] Regional averages (cluster mean speed)
  - [ ] Centrality metrics (degree, betweenness, closeness)
  
  **Spatio-Temporal Features:**
  - [ ] Lagged spatial diffusion (flow from neighbors at t-1, t-2)
  - [ ] Spatial gradient of speed change
  - [ ] Congestion propagation patterns
  
  **Feature Selection:**
  - [ ] Correlation analysis
  - [ ] Feature importance ranking
  - [ ] Select top 10-15 spatial features

- [ ] Save feature-engineered dataset
  - `data/processed/data_with_spatial_features.pkl`
  - Shape: (34,272, 207, ~20 features)

**Output Files:**
- `notebooks/03_Feature_Engineering.ipynb`
- `results/feature_importance.png`
- `data/processed/data_with_spatial_features.pkl`

---

## Phase 4: Model Enhancement (Priority)
**Duration:** 4-6 hours | **Effort:** High | **Impact:** High

### 4.1 Baseline Model Integration (Use Location Data)
**Goal:** Enhance current LSTM/GRU with spatial awareness

Tasks:
- [ ] Update model training notebooks
  - Modify `02_LSTM_Scratch.ipynb`
  - Modify `03_GRU_Scratch.ipynb`
  - **Add spatial features to input**
  - Keep architecture unchanged
  - Compare performance: baseline vs. spatial features

  Changes:
  ```python
  # Before: X_train shape (23978, 12, 207)
  # After:  X_train shape (23978, 12, 207, 20)  # 20 = spatial features
  ```

- [ ] Train and evaluate
  - [ ] MAE/RMSE/MAPE on test set
  - [ ] Compare to baselines
  - [ ] Document improvement %

### 4.2 Advanced Spatial Model: Spatial-Temporal Attention
**Goal:** New model leveraging spatial structure

Tasks:
- [ ] Create `notebooks/04B_SpatialAttention_Scratch.ipynb`
  
  Architecture:
  ```
  Input: (12, 207, 20)  # timesteps, sensors, features
  
  Temporal Encoder:
    - LSTM(12) → (207, 64)  # per-sensor temporal pattern
  
  Spatial Attention:
    - Multi-head attention(207) → (207, 64)
    - Attention weights based on:
      * Learned spatial relationships
      * Distance-based kernel
  
  Fusion:
    - Combine temporal + spatial
    - Decoder: LSTM(1) → 207 predictions
  ```

  Benefits vs current STFormer:
  - Explicit spatial attention weights
  - Distance-aware interaction
  - Interpretable spatial dependencies

- [ ] Training
  - 20 epochs on full dataset
  - Same evaluation metrics
  - Expected: 5-10% improvement over baseline

**Output Files:**
- `notebooks/04B_SpatialAttention_Scratch.ipynb`
- `models/spatial_attention_best.pth`
- `results/spatial_attention_training.png`
- `results/spatial_attention_comparison.png`

### 4.3 Graph Neural Network Baseline
**Goal:** Introduce GNN as future direction

Tasks:
- [ ] Create `notebooks/04C_GraphNeural_Baseline.ipynb`
  
  Simple GCN architecture:
  ```
  Input: X (207, 12, features) + Adjacency (207, 207)
  
  GCN Layer 1:
    - A' = normalize(Adjacency + I)
    - H' = σ(A' X W1)  → (207, 64)
  
  GCN Layer 2:
    - H'' = σ(A' H' W2) → (207, 32)
  
  Temporal:
    - LSTM on H'' → predictions
  ```

  Focus: Proof of concept + documentation
  - Can use existing libraries (PyG, DGL)
  - Expected: Similar performance to STFormer
  - Value: Shows scalability for future work

**Output Files:**
- `notebooks/04C_GraphNeural_Baseline.ipynb` (proof of concept)
- Documentation on GNN potential

---

## Phase 5: Visualization & Interpretability
**Duration:** 2-3 hours | **Effort:** Medium

### 5.1 Spatial Prediction Visualization
**Goal:** Show which sensors influence prediction

Tasks:
- [ ] Create `notebooks/05_Prediction_Analysis.ipynb`
  
  For each model (LSTM, GRU, STFormer, SpatialAttention):
  - [ ] Attention/weight heatmap
    - Show which sensors influence each prediction
    - Visualize spatial attention patterns
  
  - [ ] Prediction error by location
    - Map errors to sensor coordinates
    - Identify problematic regions
  
  - [ ] Spatial error patterns
    - Do certain highway segments have higher errors?
    - Correlation with traffic conditions?

**Output Files:**
- `results/attention_heatmaps_all_models.png`
- `results/prediction_error_map.png`
- `results/spatial_error_patterns.png`

### 5.2 Interactive Dashboard (Optional)
**Goal:** Exploratory tool for model inspection

Tasks:
- [ ] Create Streamlit/Plotly dashboard
  - Live traffic map (sensor locations + predictions)
  - Model selection (LSTM, GRU, STFormer, SpatialAttention)
  - Time slider for different timestamps
  - Error visualization

**Output:**
- `app/traffic_dashboard.py` (optional, time permitting)

---

## Phase 6: Comparative Analysis & Paper Writing
**Duration:** 3-4 hours | **Effort:** Medium-High | **Impact:** High

### 6.1 Model Comparison Framework
**Goal:** Comprehensive benchmark

Tasks:
- [ ] Create `notebooks/06_Model_Comparison.ipynb`
  
  Models to compare:
  1. Baseline (no spatial features)
  2. LSTM with spatial features
  3. GRU with spatial features
  4. STFormer (current)
  5. STFormer + spatial features
  6. Spatial Attention (new)
  7. GCN baseline
  
  Metrics:
  - MAE, RMSE, MAPE
  - By time of day (peak vs off-peak)
  - By location (freeway segment)
  - Computational cost (FLOPs, memory, inference time)

- [ ] Statistical testing
  - [ ] Paired t-tests for significance
  - [ ] Confidence intervals
  - [ ] Performance scaling with horizon (1-step to 12-step)

**Output Files:**
- `results/model_comparison_comprehensive.csv`
- `results/model_comparison_metrics.png`
- `results/model_comparison_by_location.png`
- `results/model_comparison_by_time.png`
- `results/computational_cost_comparison.png`

### 6.2 Paper Outline & Results Summary
**Goal:** Publication-ready document structure

Tasks:
- [ ] Create `PROJECT_REPORT_ENHANCED.md`

  Sections:
  ```markdown
  1. Introduction
     - Traffic prediction importance
     - Spatial-temporal challenges
     - METR-LA dataset (with real sensor network)
  
  2. Related Work
     - Time-series models (ARIMA, LSTM, GRU)
     - Spatial-temporal architectures (STFormer, ST-ResNet)
     - Graph neural networks for traffic
  
  3. Dataset & Preprocessing
     - METR-LA: 34,272 × 207 sensor network
     - Real GPS coordinates: LA freeway network
     - Distance matrix: 1,722 connected pairs
     - Feature engineering approach
  
  4. Methodology
     - Baseline models (LSTM, GRU)
     - Proposed spatial-temporal models
     - Attention mechanisms
     - GNN baseline
  
  5. Experiments
     - Train/val/test split strategy
     - Hyperparameters
     - Training procedures
  
  6. Results
     - Comparative analysis table
     - Visualization: attention maps, error maps
     - Statistical significance
  
  7. Analysis & Discussion
     - Spatial patterns learned by models
     - Which regions benefit from spatial features?
     - Failure modes and limitations
  
  8. Future Work
     - Multi-horizon forecasting
     - Event-aware prediction
     - Transfer learning to other cities
  
  9. Reproducibility
     - Dataset: witgaw/METR-LA (Hugging Face)
     - Code: GitHub repository
     - Results: Saved models + predictions
  ```

**Output Files:**
- `PROJECT_REPORT_ENHANCED.md`
- `results/all_comparison_tables.md`

---

## Phase 7: Documentation & Reproducibility
**Duration:** 1-2 hours | **Effort:** Low-Medium

### 7.1 Complete Documentation
**Goal:** Enable others to reproduce

Tasks:
- [ ] Update `README.md`
  - Dataset description (with locations)
  - Model architectures explained
  - How to run each notebook
  - Results reproduction steps

- [ ] Create `REPRODUCIBILITY.md`
  - Exact package versions
  - Random seeds for reproducibility
  - Expected computational requirements
  - Output file structure

- [ ] Create `DATA_README.md`
  - METR-LA sensor network details
  - Location coordinate reference
  - Distance computation method
  - Data quality notes

- [ ] Code cleanup
  - [ ] Add docstrings to all functions
  - [ ] Type hints for Python 3.8+
  - [ ] Comments on complex sections
  - [ ] Requirements.txt with exact versions

**Output Files:**
- `README_ENHANCED.md`
- `REPRODUCIBILITY.md`
- `DATA_README.md`
- `requirements.txt` (updated)

---

## Phase 8: Optional Enhancements
**Duration:** Variable | **Effort:** Medium-High | **Impact:** Medium

### 8.1 Multi-Horizon Forecasting
- Predict 1, 3, 6, 12 steps ahead
- Compare with single-step models
- Analyze forecast degradation over horizon

### 8.2 Anomaly Detection Layer
- Detect unusual traffic patterns
- Separate normal vs anomaly predictions
- Evaluate on congestion events

### 8.3 Transfer Learning
- Pre-train on METR-LA
- Fine-tune on PEMS-BAY dataset (if available)
- Demonstrate spatial feature transferability

### 8.4 Attention Visualization Tool
- Interactive web tool to explore attention weights
- Show "what the model sees"
- Educational value for publications

---

## Implementation Timeline

```
WEEK 1:
  ├─ Mon-Tue (Phase 1): Data Integration
  ├─ Wed-Thu (Phase 2): Spatial EDA
  └─ Fri (Phase 3): Feature Engineering [START]

WEEK 2:
  ├─ Mon-Tue (Phase 3 CONT): Feature Engineering
  ├─ Wed-Fri (Phase 4): Model Enhancement
  │  ├─ Enhance LSTM/GRU with spatial
  │  ├─ NEW Spatial Attention model
  │  └─ GNN baseline (optional)
  └─ Fri Eve: Phase 5 prep

WEEK 3:
  ├─ Mon (Phase 5): Prediction Visualizations
  ├─ Tue-Wed (Phase 6): Comprehensive Comparison
  ├─ Thu (Phase 7): Documentation
  └─ Fri: Polish & Final Results

WEEK 4 (BUFFER):
  ├─ Optional enhancements
  ├─ Paper writing
  └─ Code cleanup & submission
```

---

## Success Metrics

✅ **Phase 1-2:** Can load + visualize 207 sensors on LA map
✅ **Phase 3:** 5+ new spatial features engineered
✅ **Phase 4:** New SpatialAttention model 5-10% better than baseline
✅ **Phase 5:** Attention maps show meaningful spatial patterns
✅ **Phase 6:** Comprehensive comparison of 7 models
✅ **Phase 7:** Reproducible with exact package versions
✅ **Phase 8:** Bonus enhancements (time permitting)

---

## Deliverables

### Code
- [ ] 10+ new/updated Jupyter notebooks
- [ ] Enhanced utilities (spatial_utils.py, visualization_utils.py)
- [ ] New models (SpatialAttention, GCN baseline)
- [ ] Cleaned, documented codebase

### Data
- [ ] Unified dataset with location information
- [ ] Feature-engineered version with spatial features
- [ ] Trained models (5+ variants)

### Visualizations
- [ ] 15+ high-quality figures for paper
- [ ] Interactive maps and dashboards
- [ ] Attention weight visualizations

### Documentation
- [ ] Enhanced README with spatial analysis
- [ ] Reproducibility guide
- [ ] Data documentation
- [ ] Research paper outline

### Results
- [ ] Comprehensive model comparison table
- [ ] Statistical analysis with significance tests
- [ ] Spatial pattern analysis
- [ ] Performance by location/time

---

## Resource Requirements

**Computational:**
- GPU: RTX 4060 (✓ you have this)
- Memory: 16GB RAM (sufficient)
- Storage: 100GB for models + outputs
- Training time: ~5-10 hours total for all models

**Software:**
- PyTorch 2.7.1 + CUDA 11.8 ✓
- Pandas, NumPy, SciPy ✓
- Scikit-learn (feature engineering)
- Matplotlib, Seaborn (visualizations)
- Folium or Plotly (map visualizations)
- Optional: PyTorch Geometric (GNN)

**Knowledge:**
- Spatial-temporal deep learning ✓ (STFormer paper covers)
- Attention mechanisms ✓ (implemented in STFormer)
- Graph neural networks ~ (basic GCN sufficient)

---

## Risk & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Spatial features don't improve performance | Medium | Multiple spatial representations tested; GNN provides alternative |
| Long training times | Low | GPU available; hyperparameter tuning for efficiency |
| Coordinate mapping mismatch | High | Comprehensive validation in Phase 1.1 |
| Reproducibility issues | Low | Detailed documentation; frozen dependencies |
| Paper complexity | Medium | Start with clear narrative; iterative refinement |

---

## Next Immediate Action

**START Phase 1.1 Today:**

```python
# Create: utils/dataset_loader.py

def load_unified_metr_la():
    """Load traffic data + sensor locations + distances"""
    # Load speed data from HDF5
    # Load locations from METR-LA/sensor_graph/
    # Create mapping JSON
    # Validate consistency
    return data, metadata, coordinates
```

**EXPECTED OUTPUT:** `data/processed/unified_data.pkl`

---

## Questions to Guide Implementation

1. **Data Integration (Phase 1):** How to best merge indices with real sensor IDs?
2. **Spatial Features (Phase 3):** Which spatial representations most important?
3. **Model Design (Phase 4):** Should spatial attention be input-level or layer-level?
4. **Comparison (Phase 6):** Which baselines are most relevant for your paper?
5. **Paper (Phase 6):** What's your target venue/journal?

---

**Status:** Ready to implement  
**Total Estimated Time:** 3-4 weeks for full plan  
**MVP Time:** 1-2 weeks (Phases 1-4)
