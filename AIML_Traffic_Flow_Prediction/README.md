---
title: Traffic Flow Prediction — METR-LA
emoji: 🚦
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🚦 Traffic Flow Prediction Dashboard

Deep learning models for traffic speed forecasting on the **METR-LA** dataset.

## Models
| Model | MAE | R² |
|-------|-----|-----|
| XGBoost | 3.44 | 0.895 |
| STFormer (Scratch) | 4.55 | 0.856 |
| GRU (Scratch) | 4.99 | 0.813 |
| LSTM (Scratch) | 5.08 | 0.810 |

## Dataset
- **207** road sensors across Los Angeles
- **34,272** timesteps at 5-minute intervals (~4 months)
- Input: 12 timesteps (1 hour) → Output: 15/30/60 min ahead

## Architecture
Spatial Attention LSTM — combines LSTM temporal modelling with multi-head attention across the sensor graph.
