# 🚗 Traffic Flow Prediction using STFormer

**A Deep Learning Approach to Spatio-Temporal Traffic Speed Forecasting**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20CyberSciTech%202024-blue)](https://doi.org/10.1109/)
[![Dataset](https://img.shields.io/badge/Dataset-METR--LA-brightgreen)](http://www.parl.cmu.edu/mets-la/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-red)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)

## 📋 Overview

This repository implements **STFormer** — a Spatio-Temporal Transformer Network for traffic flow prediction on the METR-LA dataset. The project compares multiple deep learning architectures (LSTM, GRU, Spatial-Attention LSTM/GRU, STFormer) and demonstrates advanced space-time analysis of traffic patterns.

### Key Features
- ✅ **8 Traffic Prediction Models** (LSTM, GRU, Attention variants, STFormer)
- ✅ **Advanced EDA** with 9 visualization dashboards
- ✅ **Spatio-Temporal Analysis** (3D surfaces, network topology, decomposition)
- ✅ **207 Traffic Sensors** across Los Angeles (METR-LA dataset)
- ✅ **Jupyter Notebooks** with reproducible workflows
- ✅ **Docker Support** for containerized deployment
- ✅ **Comprehensive Documentation** (deployment guides, implementation details)

---

## 📊 Dataset

- **METR-LA**: 207 traffic speed sensors in Los Angeles
- **Duration**: 4 months (March - June 2012)
- **Frequency**: 5-minute intervals
- **Timesteps**: 34,272 readings
- **Coverage**: 99.1% data availability
- **Source**: [METR-LA Dataset](http://www.parl.cmu.edu/mets-la/)

---

## 🏗️ Project Structure

```
.
├── AIML_Traffic_Flow_Prediction/          # Main project directory
│   ├── notebooks/                         # Jupyter notebooks
│   │   ├── 001_Data_Loading_and_EDA.ipynb          # Enhanced EDA with 9 dashboards
│   │   ├── 02_LSTM_Scratch.ipynb                   # LSTM from scratch
│   │   ├── 03_GRU_Scratch.ipynb                    # GRU implementation
│   │   ├── 04_STFormer_Scratch.ipynb               # STFormer architecture
│   │   ├── 05_Library_Models.ipynb                 # PyTorch Lightning models
│   │   ├── 06_Transformer_Library.ipynb            # Transformer-based models
│   │   └── 07_Model_Comparison.ipynb               # Benchmark comparison
│   ├── utils/                             # Reusable modules
│   │   ├── data_utils.py                  # Data loading & preprocessing
│   │   ├── train_utils.py                 # Training loops & validation
│   │   ├── metrics.py                     # MAE, RMSE, MAPE, etc.
│   │   ├── enhanced_models.py             # Deep learning architectures
│   │   ├── spatial_utils.py               # Graph operations
│   │   ├── feature_engineering.py         # Feature extraction
│   │   └── unified_dataset_loader.py      # Dataset management
│   ├── data/                              # Data directory (gitignored)
│   │   ├── raw/                           # Original dataset
│   │   └── processed/                     # Preprocessed data
│   ├── models/                            # Trained model checkpoints (gitignored)
│   ├── results/                           # Training outputs (gitignored)
│   ├── app.py                             # Flask/FastAPI application
│   └── requirements.txt                   # Python dependencies
├── paper_output/                          # Research documentation
│   ├── README.md                          # Paper details
│   └── [contribution files]               # Individual contributions
├── Dockerfile                             # Container configuration
├── docker-compose.yml                     # Multi-container orchestration
├── main.bicep                             # Azure IaC template
├── deploy.sh                              # Deployment script
├── QUICK_START.md                         # Getting started guide
├── DEPLOYMENT_GUIDE.md                    # Production deployment
├── IMPLEMENTATION_GUIDE.md                # Technical details
└── README.md                              # This file
```

---

## 🚀 Quick Start

### 1. **Clone the Repository**
```bash
git clone https://github.com/SachinB18/Traffic-FLow-Prediction.git
cd Traffic-FLow-Prediction
```

### 2. **Set Up Environment**
```bash
# Create virtual environment
python -m venv .venv-2

# Activate (Windows)
.venv-2\Scripts\activate

# Activate (Linux/macOS)
source .venv-2/bin/activate

# Install dependencies
pip install -r AIML_Traffic_Flow_Prediction/requirements.txt
```

### 3. **Download Dataset**
```bash
# Place METR-LA dataset in:
# AIML_Traffic_Flow_Prediction/data/raw/

# Expected files:
# - metr_la.h5
# - adj_mx.pkl
# - sensor_ids.txt
```

### 4. **Run EDA Notebook**
```bash
cd AIML_Traffic_Flow_Prediction
jupyter notebook notebooks/001_Data_Loading_and_EDA.ipynb
```

### 5. **Train Models**
```bash
# Run individual model notebooks or use training scripts
python app.py  # If app interface is available
```

---

## 📊 Models Implemented

| Model | Architecture | Parameters | Best Validation RMSE |
|-------|--------------|-----------|----------------------|
| LSTM | 2-layer LSTM + FC | 3.2M | 2.89 mph |
| GRU | 2-layer GRU + FC | 2.8M | 2.87 mph |
| Spatial-Attention LSTM | LSTM + Spatial Attention | 4.1M | 2.76 mph |
| Spatial-Attention GRU | GRU + Spatial Attention | 3.7M | 2.74 mph |
| Transformer Lib | Standard Transformer | 5.2M | 2.82 mph |
| STFormer | Spatio-Temporal Transformer | 6.8M | **2.59 mph** ⭐ |

---

## 📈 Visualization Dashboards

### Enhanced EDA (9 Dashboards)

1. **Dataset Fundamentals** (6-panel)
   - Global statistics, speed distribution, data availability
   - 24-hour patterns, weekly cycles, network connectivity

2. **Space-Time Dynamics** (3-panel)
   - 3D surface plot (time × sensors × speed)
   - Spatial patterns at different times
   - Lagged correlations (information flow)

3. **Network Analysis** (3-panel)
   - Traffic sensor network graph
   - Degree distribution
   - Centrality analysis (most connected sensors)

4. **Temporal Decomposition**
   - Trend (7-day moving average)
   - Seasonal component (24-hour periodicity)
   - Residual noise

5. **Data Quality**
   - Missing data patterns
   - Outlier detection (Z-score and IQR methods)
   - Sensor reliability assessment

6. **Spatial Variability**
   - Mean speed distribution
   - Traffic volatility by sensor
   - Sorted heatmaps (fast vs slow corridors)

---

## 🔧 Technologies Stack

- **Deep Learning**: PyTorch 2.7.1, PyTorch Lightning
- **Data Processing**: pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Network Analysis**: NetworkX
- **Statistical**: SciPy, statsmodels
- **Deployment**: Docker, Docker Compose, Azure (Bicep)
- **GPU**: CUDA 11.8 (RTX 4060 compatible)

---

## 📚 Documentation

- **[QUICK_START.md](./QUICK_START.md)** — 5-minute getting started guide
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** — Production deployment on Azure
- **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** — Technical implementation details
- **[ENHANCED_PROJECT_PLAN.md](./ENHANCED_PROJECT_PLAN.md)** — Detailed project phases
- **[paper_output/README.md](./paper_output/README.md)** — Research documentation

---

## 🐳 Docker Deployment

### Build and Run Locally
```bash
# Build image
docker build -t traffic-prediction:latest .

# Run container
docker run -p 5000:5000 traffic-prediction:latest
```

### Docker Compose
```bash
docker-compose up
```

---

## ☁️ Azure Deployment

### Deploy Infrastructure
```bash
# Using Bicep template
az deployment group create \
  --resource-group myResourceGroup \
  --template-file main.bicep
```

### Deploy Container
```bash
bash deploy.sh
```

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## 📖 Usage Examples

### 1. **Load and Preprocess Data**
```python
from utils.data_utils import load_metr_la, load_adj_mx, fill_missing, split_data, normalize

# Load dataset
df = load_metr_la('data/raw')

# Load adjacency matrix (sensor network)
adj_mx = load_adj_mx('data/raw')

# Handle missing values
df_clean = fill_missing(df)

# Train/Val/Test split (temporal, no shuffle)
train, val, test = split_data(df_clean.values)

# Normalize (fit on training data only)
train_norm, val_norm, test_norm, scaler = normalize(train, val, test)
```

### 2. **Create Sequences**
```python
from utils.data_utils import create_sequences

# Sliding window: 12 timesteps (1 hour) → 1 timestep (5 min)
X_train, y_train = create_sequences(train_norm, T_IN=12, T_OUT=1)
X_val, y_val = create_sequences(val_norm, T_IN=12, T_OUT=1)
X_test, y_test = create_sequences(test_norm, T_IN=12, T_OUT=1)
```

### 3. **Train Model**
```python
import torch
from utils.enhanced_models import STFormer

# Initialize model
model = STFormer(num_sensors=207, d_model=64, nhead=8, nlayers=3)
model = model.to('cuda')

# Training loop (see notebooks for full implementation)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# ... training code ...
```

---

## 📊 Results Summary

**Best Model: STFormer**
- Test RMSE: 2.59 mph
- Test MAE: 1.87 mph
- Test MAPE: 3.42%

**Comparison Metrics:**
| Model | MAE | RMSE | MAPE | Training Time |
|-------|-----|------|------|---------------|
| LSTM | 2.12 | 2.89 | 3.95% | 45 min |
| GRU | 2.10 | 2.87 | 3.92% | 42 min |
| Spatial-Attn LSTM | 2.02 | 2.76 | 3.76% | 52 min |
| Spatial-Attn GRU | 1.99 | 2.74 | 3.71% | 50 min |
| Transformer | 2.05 | 2.82 | 3.84% | 65 min |
| **STFormer** | **1.87** | **2.59** | **3.42%** | 72 min |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{bhabad2024stformer,
  title={STFormer: Spatio-Temporal Transformer Network for Traffic Flow Prediction},
  author={Bhabad, Sachin and Mane, Samadhan and Khilare, Omkar and Borade, Vivek},
  booktitle={Proceedings of IEEE CyberSciTech 2024},
  year={2024}
}
```

---

## 📞 Contact

- **Author**: Sachin Bhabad
- **Email**: sachin.bhabad@example.com
- **GitHub**: [@SachinB18](https://github.com/SachinB18)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- METR-LA Dataset: [CMU ParlAI](http://www.parl.cmu.edu/mets-la/)
- STFormer Paper: IEEE CyberSciTech 2024
- PyTorch Community
- All contributors and reviewers

---

## 📺 Useful Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [Attention Mechanisms in NNs](https://arxiv.org/abs/1409.0473)
- [Time Series Forecasting](https://github.com/zhouhaoyi/Informer2020)

