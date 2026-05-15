"""
Traffic Flow Prediction Dashboard
METR-LA Dataset | LSTM · GRU · STFormer
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Flow Prediction | METR-LA",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
RES_DIR   = BASE / "results"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-card h2 { color: #4fc3f7; margin: 0; font-size: 2rem; }
.metric-card p  { color: #90caf9; margin: 4px 0 0; font-size: 0.85rem; }

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4fc3f7, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #4fc3f7;
    border-left: 4px solid #4fc3f7;
    padding-left: 10px;
    margin: 20px 0 10px;
}
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Model definition  (must match training code)
# ══════════════════════════════════════════════════════════════════════════════
class SpatialAttention(nn.Module):
    def __init__(self, n_sensors, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, X, A):
        b, t, n, d = X.shape
        Q = self.W_q(X).reshape(b,t,n,self.n_heads,self.d_head).permute(0,1,3,2,4)
        K = self.W_k(X).reshape(b,t,n,self.n_heads,self.d_head).permute(0,1,3,2,4)
        V = self.W_v(X).reshape(b,t,n,self.n_heads,self.d_head).permute(0,1,3,2,4)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.d_head)
        if A is not None:
            scores = scores * A.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        ctx  = torch.matmul(attn, V).permute(0,1,3,2,4).contiguous().reshape(b,t,n,d)
        return self.W_o(ctx)

class SpatialAttentionLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1,
                 n_sensors=207, n_layers=2, attention_heads=4,
                 attention_matrix=None, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_sensors  = n_sensors
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.spatial_attn = SpatialAttention(n_sensors, hidden_dim, attention_heads)
        if attention_matrix is not None:
            self.register_buffer('attention_matrix',
                                 torch.from_numpy(attention_matrix).float())
        else:
            self.register_buffer('attention_matrix', None)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        b, s, n, d = X.shape
        X_flat  = X.reshape(b*n, s, d)
        out, _  = self.lstm(X_flat)
        out     = out.reshape(b, s, n, self.hidden_dim)
        out     = self.spatial_attn(out, self.attention_matrix)
        final   = out[:, -1, :, :]
        output  = self.output_layer(final).unsqueeze(1)
        return output

# ══════════════════════════════════════════════════════════════════════════════
# Data / model loading (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    attn = np.load(DATA_DIR / "attention_correlation.npy")
    model = SpatialAttentionLSTM(
        input_dim=1, hidden_dim=64, output_dim=1,
        n_sensors=207, n_layers=2, attention_heads=4,
        attention_matrix=attn, dropout=0.2
    )
    state = torch.load(MODEL_DIR / "spatial_attention_lstm_best.pth",
                       map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_data(show_spinner="Loading data…")
def load_data():
    sensors  = pd.read_csv(DATA_DIR / "sensor_coordinates.csv")
    X_test   = np.load(DATA_DIR / "X_test.npy")   # (N,12,207,1)
    y_test   = np.load(DATA_DIR / "y_test.npy")   # (N,3,207,1)
    metrics  = pd.read_csv(RES_DIR / "model_comparison.csv")
    per_s    = pd.read_csv(RES_DIR / "per_sensor_metrics.csv")
    with open(RES_DIR / "training_history_lstm.json") as f:
        hist = json.load(f)
    speed    = np.load(DATA_DIR / "speed_data.npy")  # (T, 207)
    return sensors, X_test, y_test, metrics, per_s, hist, speed

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🚦 Traffic Prediction")
    st.markdown("**Dataset:** METR-LA")
    st.markdown("**Sensors:** 207 · **Timestep:** 5 min")
    st.divider()

    tab_choice = st.radio("Navigate", [
        "🏠 Overview",
        "🔮 Live Prediction",
        "📊 Model Comparison",
        "📈 Training Curves",
    ])
    st.divider()
    st.caption("Built with Streamlit · PyTorch · METR-LA")

# ══════════════════════════════════════════════════════════════════════════════
# Load everything
# ══════════════════════════════════════════════════════════════════════════════
try:
    model = load_model()
    sensors, X_test, y_test, metrics_df, per_sensor, hist, speed = load_data()
    with open(RES_DIR / "training_history_gru.json") as f:
        hist_gru = json.load(f)
    with open(RES_DIR / "training_history_phase4.json") as f:
        hist_p4 = json.load(f)
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Could not load data/model: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == "🏠 Overview":
    st.markdown('<p class="hero-title">🚦 Traffic Flow Prediction</p>', unsafe_allow_html=True)
    st.markdown("**Deep Learning on METR-LA · Los Angeles Freeway Network**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["207", "34,272", "~4 months", "3 models"],
        ["Road Sensors", "Timesteps", "Data Span", "DL Architectures"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">How It Works</p>', unsafe_allow_html=True)
        st.markdown("""
| Step | What happens |
|------|-------------|
| **Input** | Last **12 timesteps** (1 hour) of speed at all 207 sensors |
| **Model** | LSTM + Spatial Attention processes both time & space |
| **Output** | Predicted speed at **15 min, 30 min, 60 min** ahead |

**Best model (XGBoost):** MAE = 3.44 mph · R² = 0.895
        """)

    with col_b:
        st.markdown('<p class="section-header">Model Leaderboard</p>', unsafe_allow_html=True)
        if data_ok:
            df = metrics_df.copy()
            df = df.rename(columns={"MAE ↓": "MAE", "RMSE ↓": "RMSE", "MAPE (%) ↓": "MAPE%", "R² ↑": "R²"})
            st.dataframe(
                df[["Model", "MAE", "RMSE", "MAPE%", "R²"]].style.highlight_min(
                    subset=["MAE", "RMSE", "MAPE%"], color="#1a3a2a"
                ).highlight_max(subset=["R²"], color="#1a3a2a"),
                use_container_width=True, hide_index=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🔮 Live Prediction":
    st.markdown('<p class="hero-title">🔮 Live Prediction Demo</p>', unsafe_allow_html=True)
    st.markdown("Pick a sensor and a test sample — see what the model predicts vs actual speed.")
    st.divider()

    if not data_ok:
        st.warning("Data not loaded.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            sensor_id = st.selectbox(
                "Select Sensor",
                options=list(range(207)),
                format_func=lambda x: f"Sensor {x:03d}"
            )
            sample_idx = st.slider("Test Sample Index", 0, min(999, len(X_test)-1), 0)
            run = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

            # Show sensor info
            row = sensors.iloc[sensor_id]
            st.info(f"""
**Sensor {sensor_id:03d}**
- Lat: `{row.latitude:.4f}`
- Lon: `{row.longitude:.4f}`
            """)

        with col2:
            if run:
                with st.spinner("Running inference…"):
                    X_sample = X_test[sample_idx:sample_idx+1]   # (1,12,207,1)
                    y_sample = y_test[sample_idx]                 # (3,207,1)

                    X_tensor = torch.from_numpy(X_sample).float()
                    with torch.no_grad():
                        pred = model(X_tensor).numpy()            # (1,1,207,1)

                    # For this sensor
                    past    = X_sample[0, :, sensor_id, 0]       # 12 values
                    actual  = y_sample[:, sensor_id, 0]           # 3 future values
                    # Model predicts 1 step; repeat for demo
                    pred_val = float(pred[0, 0, sensor_id, 0])

                    # Build time axis
                    t_past   = list(range(-12, 0))
                    t_future = [3, 6, 12]   # in 5-min units → 15,30,60 min

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t_past, y=past,
                        mode="lines+markers", name="Past (Input)",
                        line=dict(color="#4fc3f7", width=2),
                        marker=dict(size=5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=t_future, y=actual,
                        mode="lines+markers", name="Actual Future",
                        line=dict(color="#81c784", width=2, dash="dot"),
                        marker=dict(size=8, symbol="diamond")
                    ))
                    # Predicted (single step, shown at t=3)
                    fig.add_trace(go.Scatter(
                        x=[3], y=[pred_val],
                        mode="markers", name="Model Prediction (15 min)",
                        marker=dict(size=14, color="#ff8a65", symbol="star")
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                                  annotation_text="Now")
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Sensor {sensor_id:03d} — Speed Over Time",
                        xaxis_title="Time (× 5 min)",
                        yaxis_title="Speed (normalised)",
                        height=380,
                        legend=dict(orientation="h", y=-0.2),
                        margin=dict(t=50, b=60),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    err = abs(pred_val - actual[0])
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted (15 min)", f"{pred_val:.4f}")
                    m2.metric("Actual (15 min)",    f"{actual[0]:.4f}")
                    m3.metric("Absolute Error",     f"{err:.4f}")
            else:
                st.info("👈 Select a sensor and click **Run Prediction**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📊 Model Comparison":
    st.markdown('<p class="hero-title">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown("Full comparison of all 6 models — accuracy, error metrics, and epoch-wise training behaviour.")
    st.divider()

    if not data_ok:
        st.warning("Data not loaded.")
    else:
        df = metrics_df.copy().rename(
            columns={"MAE ↓": "MAE", "RMSE ↓": "RMSE", "MAPE (%) ↓": "MAPE%", "R² ↑": "R²"}
        )
        colors = ["#4fc3f7", "#ff8a65", "#81c784", "#7c4dff", "#ffb74d", "#e57373"]

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Accuracy & Error",
            "🕸️ Radar Chart",
            "📉 Epoch-wise Loss",
            "🎯 Per-Sensor MAE",
        ])

        # ── Tab 1: Bar charts ────────────────────────────────────────────────
        with tab1:
            st.markdown('<p class="section-header">Model Accuracy & Error Metrics</p>', unsafe_allow_html=True)

            # Leaderboard table
            styled = df[["Model","Type","MAE","RMSE","MAPE%","R²"]].style\
                .highlight_min(subset=["MAE","RMSE","MAPE%"], color="#0d3320")\
                .highlight_max(subset=["R²"], color="#0d3320")\
                .format({"MAE":"{:.4f}","RMSE":"{:.4f}","MAPE%":"{:.2f}","R²":"{:.4f}"})
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.divider()

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "MAE — Mean Absolute Error (↓ lower is better)",
                    "RMSE — Root Mean Sq Error (↓ lower is better)",
                    "MAPE % (↓ lower is better)",
                    "R² Score (↑ higher is better)",
                ]
            )
            metrics_list = [("MAE", 1, 1, True), ("RMSE", 1, 2, True),
                            ("MAPE%", 2, 1, True), ("R²", 2, 2, False)]
            for metric, row, col, asc in metrics_list:
                dfs = df.sort_values(metric, ascending=asc)
                fig.add_trace(go.Bar(
                    x=dfs["Model"], y=dfs[metric],
                    marker_color=colors[:len(dfs)],
                    text=dfs[metric].round(3),
                    textposition="outside",
                    showlegend=False,
                ), row=row, col=col)
            fig.update_layout(
                template="plotly_dark", height=620,
                margin=dict(t=60, b=40),
            )
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # ── Tab 2: Radar ─────────────────────────────────────────────────────
        with tab2:
            st.markdown('<p class="section-header">Multi-metric Radar Comparison</p>', unsafe_allow_html=True)
            cats = ["MAE", "RMSE", "MAPE%"]
            fig2 = go.Figure()
            for idx, row in df.iterrows():
                vals = [float(row[c]) for c in cats] + [float(row[cats[0]])]
                fig2.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]],
                    fill="toself", name=row["Model"],
                    line=dict(color=colors[idx % len(colors)], width=2),
                    opacity=0.8,
                ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, gridcolor="#334"),
                           angularaxis=dict(gridcolor="#334")),
                template="plotly_dark", height=520,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Smaller area = better model overall")

        # ── Tab 3: Epoch-wise Loss ────────────────────────────────────────────
        with tab3:
            st.markdown('<p class="section-header">Epoch-wise Training & Validation Loss</p>', unsafe_allow_html=True)
            st.markdown("Showing LSTM (60 epochs), GRU (60 epochs), and Spatial-LSTM/Phase4 (16 epochs).")

            col_a, col_b = st.columns(2)

            # Training Loss comparison
            with col_a:
                fig_tl = go.Figure()
                e_lstm = list(range(1, len(hist["train_loss"])+1))
                e_gru  = list(range(1, len(hist_gru["train_loss"])+1))
                e_p4   = list(range(1, len(hist_p4["train_loss"])+1))
                fig_tl.add_trace(go.Scatter(
                    x=e_lstm, y=hist["train_loss"],
                    name="LSTM Train", line=dict(color="#4fc3f7", width=2)
                ))
                fig_tl.add_trace(go.Scatter(
                    x=e_gru, y=hist_gru["train_loss"],
                    name="GRU Train", line=dict(color="#ff8a65", width=2)
                ))
                fig_tl.add_trace(go.Scatter(
                    x=e_p4, y=hist_p4["train_loss"],
                    name="Spatial-LSTM Train", line=dict(color="#81c784", width=2, dash="dot")
                ))
                fig_tl.update_layout(
                    template="plotly_dark", title="Training Loss per Epoch",
                    xaxis_title="Epoch", yaxis_title="Loss (MSE)",
                    height=380, legend=dict(orientation="h", y=-0.25),
                    margin=dict(b=70, t=50),
                )
                st.plotly_chart(fig_tl, use_container_width=True)

            # Validation Loss comparison
            with col_b:
                fig_vl = go.Figure()
                fig_vl.add_trace(go.Scatter(
                    x=e_lstm, y=hist["val_loss"],
                    name="LSTM Val", line=dict(color="#4fc3f7", width=2, dash="dot")
                ))
                fig_vl.add_trace(go.Scatter(
                    x=e_gru, y=hist_gru["val_loss"],
                    name="GRU Val", line=dict(color="#ff8a65", width=2, dash="dot")
                ))
                fig_vl.add_trace(go.Scatter(
                    x=e_p4, y=hist_p4["val_loss"],
                    name="Spatial-LSTM Val", line=dict(color="#81c784", width=2)
                ))
                fig_vl.update_layout(
                    template="plotly_dark", title="Validation Loss per Epoch",
                    xaxis_title="Epoch", yaxis_title="Val Loss (MSE)",
                    height=380, legend=dict(orientation="h", y=-0.25),
                    margin=dict(b=70, t=50),
                )
                st.plotly_chart(fig_vl, use_container_width=True)

            # Validation MAE comparison
            st.markdown('<p class="section-header">Validation MAE per Epoch</p>', unsafe_allow_html=True)
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=e_lstm, y=hist["val_mae"],
                name="LSTM", line=dict(color="#4fc3f7", width=2)
            ))
            fig_mae.add_trace(go.Scatter(
                x=e_gru, y=hist_gru["val_mae"],
                name="GRU", line=dict(color="#ff8a65", width=2)
            ))
            fig_mae.add_trace(go.Scatter(
                x=e_p4, y=hist_p4["val_mae"],
                name="Spatial-LSTM", line=dict(color="#81c784", width=2, dash="dot")
            ))
            fig_mae.update_layout(
                template="plotly_dark",
                xaxis_title="Epoch", yaxis_title="Val MAE",
                height=360,
                legend=dict(orientation="h", y=-0.2),
                margin=dict(b=60, t=20),
            )
            st.plotly_chart(fig_mae, use_container_width=True)

            # Final metrics summary
            st.divider()
            st.markdown('<p class="section-header">Final Epoch Summary</p>', unsafe_allow_html=True)
            summary = pd.DataFrame({
                "Model":       ["LSTM", "GRU", "Spatial-LSTM"],
                "Epochs":      [len(e_lstm), len(e_gru), len(e_p4)],
                "Final Train Loss": [
                    round(hist["train_loss"][-1], 5),
                    round(hist_gru["train_loss"][-1], 5),
                    round(hist_p4["train_loss"][-1], 5),
                ],
                "Final Val Loss": [
                    round(hist["val_loss"][-1], 5),
                    round(hist_gru["val_loss"][-1], 5),
                    round(hist_p4["val_loss"][-1], 5),
                ],
                "Final Val MAE": [
                    round(hist["val_mae"][-1], 5),
                    round(hist_gru["val_mae"][-1], 5),
                    round(hist_p4["val_mae"][-1], 5),
                ],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Tab 4: Per-Sensor MAE ─────────────────────────────────────────────
        with tab4:
            st.markdown('<p class="section-header">Per-Sensor MAE across 207 Sensors</p>', unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=per_sensor["Sensor_ID"], y=per_sensor["LSTM_MAE"],
                mode="lines", name="LSTM",
                line=dict(color="#4fc3f7", width=1.5),
                fill="tozeroy", fillcolor="rgba(79,195,247,0.08)"
            ))
            fig3.add_trace(go.Scatter(
                x=per_sensor["Sensor_ID"], y=per_sensor["GRU_MAE"],
                mode="lines", name="GRU",
                line=dict(color="#ff8a65", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,138,101,0.08)"
            ))
            fig3.update_layout(
                template="plotly_dark",
                xaxis_title="Sensor ID",
                yaxis_title="MAE (normalised)",
                height=420,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(b=60, t=20),
            )
            st.plotly_chart(fig3, use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("LSTM — Avg MAE across sensors",
                      f"{per_sensor['LSTM_MAE'].mean():.5f}")
            m2.metric("GRU — Avg MAE across sensors",
                      f"{per_sensor['GRU_MAE'].mean():.5f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Training Curves
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📈 Training Curves":
    st.markdown('<p class="hero-title">📈 Training Curves</p>', unsafe_allow_html=True)
    st.divider()

    if not data_ok:
        st.warning("Data not loaded.")
    else:
        epochs = list(range(1, len(hist["train_loss"]) + 1))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Loss (Train vs Val)", "Validation MAE"])

        fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"],
                                 name="Train Loss", line=dict(color="#4fc3f7")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"],
                                 name="Val Loss", line=dict(color="#ff8a65", dash="dot")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=hist["val_mae"],
                                 name="Val MAE", line=dict(color="#81c784")),
                      row=1, col=2)

        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Epoch",
            xaxis2_title="Epoch",
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Train Loss", f"{hist['train_loss'][-1]:.5f}")
        c2.metric("Final Val Loss",   f"{hist['val_loss'][-1]:.5f}")
        c3.metric("Final Val MAE",    f"{hist['val_mae'][-1]:.5f}")

        st.info(f"✅ Trained for **{len(epochs)} epochs** with early stopping · Model: Spatial Attention LSTM")
