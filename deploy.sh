#!/bin/bash

# Quick deployment script for Traffic Flow Prediction to Azure

set -e

echo "🚀 Traffic Flow Prediction - Azure Deployment Script"
echo "======================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"
command -v az &> /dev/null || { echo -e "${RED}❌ Azure CLI not found${NC}"; exit 1; }
command -v docker &> /dev/null || { echo -e "${RED}❌ Docker not found${NC}"; exit 1; }
echo -e "${GREEN}✓ All prerequisites met${NC}"

# Variables
RESOURCE_GROUP="traffic-pred-rg"
LOCATION="eastus"
REGISTRY_NAME="trafficpredregistry$RANDOM"
APP_NAME="traffic-prediction-app"

# Step 1: Login
echo -e "\n${YELLOW}Step 1: Logging into Azure...${NC}"
az login

# Step 2: Create resource group
echo -e "\n${YELLOW}Step 2: Creating resource group...${NC}"
az group create --name ${RESOURCE_GROUP} --location ${LOCATION}

# Step 3: Deploy infrastructure
echo -e "\n${YELLOW}Step 3: Deploying Azure infrastructure...${NC}"
DEPLOYMENT=$(az deployment group create \
  --resource-group ${RESOURCE_GROUP} \
  --template-file main.bicep \
  --query properties.outputs)

REGISTRY_URL=$(echo $DEPLOYMENT | jq -r '.containerRegistryUrl.value')
echo -e "${GREEN}✓ Infrastructure deployed${NC}"
echo "  Registry: ${REGISTRY_URL}"

# Step 4: Build Docker image
echo -e "\n${YELLOW}Step 4: Building Docker image...${NC}"
docker build -t ${REGISTRY_URL}/${APP_NAME}:latest .
echo -e "${GREEN}✓ Image built${NC}"

# Step 5: Push to registry
echo -e "\n${YELLOW}Step 5: Pushing to Azure Container Registry...${NC}"
az acr login --name $(echo ${REGISTRY_URL} | cut -d. -f1)
docker push ${REGISTRY_URL}/${APP_NAME}:latest
echo -e "${GREEN}✓ Image pushed${NC}"

# Step 6: Get app URL
echo -e "\n${YELLOW}Step 6: Getting app URL...${NC}"
APP_URL=$(az containerapp show \
  --name ${APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null || echo "pending...")

echo -e "\n${GREEN}✅ Deployment Complete!${NC}"
echo "====================================="
echo -e "🌐 App URL: https://${APP_URL}"
echo -e "📚 API Docs: https://${APP_URL}/docs"
echo -e "🎨 Web UI: https://${APP_URL}"
echo -e "\n💡 To delete: az group delete --name ${RESOURCE_GROUP}"
