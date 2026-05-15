# 🚗 Traffic Flow Prediction - Showcase Deployment Guide

## Quick Start (Local)

### 1. Prerequisites
- Docker & Docker Compose installed
- Python 3.11+ (for running without Docker)

### 2. Run with Docker Compose
```bash
docker-compose up --build
```
Visit: http://localhost:8000

### 3. Run locally (no Docker)
```bash
# Install dependencies
pip install -r requirements-deploy.txt

# Run server
python app.py
```

---

## Deploy to Azure

### Option A: Azure Container Apps (Recommended)

#### 1. Setup
```bash
# Login to Azure
az login

# Create resource group
az group create --name traffic-pred-rg --location eastus

# Deploy infrastructure
az deployment group create \
  --resource-group traffic-pred-rg \
  --template-file main.bicep
```

#### 2. Build and Push Image
```bash
# Get registry name from deployment output
REGISTRY_NAME="trafficpredregistry..."
REGISTRY_URL="${REGISTRY_NAME}.azurecr.io"

# Build image
docker build -t ${REGISTRY_URL}/traffic-prediction:latest .

# Push to registry
az acr login --name ${REGISTRY_NAME}
docker push ${REGISTRY_URL}/traffic-prediction:latest
```

#### 3. Deploy Container App
```bash
# The Container App is already created by Bicep
# Get the URL
az containerapp show \
  --name traffic-prediction-app \
  --resource-group traffic-pred-rg \
  --query properties.configuration.ingress.fqdn
```

---

### Option B: Azure App Service (Simpler)

```bash
# Create App Service Plan
az appservice plan create \
  --name traffic-pred-plan \
  --resource-group traffic-pred-rg \
  --sku B1 \
  --is-linux

# Create Web App
az webapp create \
  --resource-group traffic-pred-rg \
  --plan traffic-pred-plan \
  --name traffic-prediction-app \
  --deployment-container-image-name ${REGISTRY_URL}/traffic-prediction:latest

# Configure continuous deployment
az webapp deployment container config \
  --name traffic-prediction-app \
  --resource-group traffic-pred-rg \
  --enable-cd true \
  --registry-server-url ${REGISTRY_URL} \
  --registry-username $(az acr credential show -n ${REGISTRY_NAME} --query username -o tsv) \
  --registry-password $(az acr credential show -n ${REGISTRY_NAME} --query 'passwords[0].value' -o tsv)
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### List Models
```bash
curl http://localhost:8000/models
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "stformer"}'
```

### Interactive UI
http://localhost:8000

### API Documentation (Swagger)
http://localhost:8000/docs

---

## Architecture

```
User Browser
    ↓
   [UI - /index.html]
    ↓
   FastAPI Server (port 8000)
    ├─ /health (health check)
    ├─ /predict (inference)
    ├─ /models (list models)
    └─ /stats (dataset info)
    ↓
  Pre-trained Models (PyTorch)
  ├─ STFormer (best)
  ├─ Transformer
  ├─ LSTM
  └─ GRU
```

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server with prediction endpoints |
| `public/index.html` | Web UI for testing |
| `Dockerfile` | Container image definition |
| `requirements-deploy.txt` | Minimal dependencies (CPU-only PyTorch) |
| `docker-compose.yml` | Local multi-container setup |
| `main.bicep` | Azure Infrastructure as Code |

---

## Customization

### Switch Default Model
Edit `app.py`, line ~30:
```python
MODEL_NAME = "lstm"  # Change to: lstm, gru, transformer_lib, stformer
```

### Change Container Resources
Edit `main.bicep`:
```bicep
resources: {
  cpu: json('1.0')      // Increase CPU
  memory: '2.0Gi'       // Increase memory
}
```

### Enable GPU Support
Edit `Dockerfile`:
```dockerfile
# Replace CPU PyTorch with CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Monitoring & Logs

### View logs locally
```bash
docker-compose logs -f
```

### Azure Container Apps
```bash
az containerapp logs show \
  --name traffic-prediction-app \
  --resource-group traffic-pred-rg
```

### Azure App Service
```bash
az webapp log tail \
  --resource-group traffic-pred-rg \
  --name traffic-prediction-app
```

---

## Cost Estimate (Monthly)

**Azure Container Apps:**
- Compute (0.5 CPU): ~$8-15
- Memory (1 GB): Included
- **Total: ~$10-20/month**

**Azure App Service (B1):**
- App Service Plan: ~$8-12/month
- **Total: ~$8-15/month**

---

## Cleanup

```bash
# Delete entire resource group
az group delete --name traffic-pred-rg
```

---

## Troubleshooting

**Container won't start:**
```bash
docker build -t traffic-prediction:latest .
docker run -p 8000:8000 traffic-prediction:latest
```

**Model loading fails:**
- Verify `./AIML_Traffic_Flow_Prediction/models/stformer_best.pth` exists
- Check file permissions

**Slow predictions:**
- Switch to smaller model (LSTM, GRU)
- Enable GPU support in Dockerfile
- Increase container CPU/memory

---

## Next Steps

1. ✅ Test locally with Docker Compose
2. 🔧 Deploy to Azure Container Apps
3. 📊 Add monitoring & alerts
4. 🔄 Setup CI/CD pipeline (GitHub Actions)
5. 🚀 Scale to production

---

Happy forecasting! 🚗📈
