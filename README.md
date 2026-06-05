# MedAI Diagnostics: Multi-Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Deployment](https://img.shields.io/badge/Deployed_on-Render-black.svg)

An end-to-end, AI-powered diagnostic web application capable of detecting four major diseases from medical imagery using custom-trained convolutional neural networks (CNNs). The system features a modern, responsive frontend, a high-performance FastAPI backend, and integrates concepts of Federated Learning.

## 🩺 Supported Diagnostics
1. **Brain Tumor Detection** (MRI Scans)
2. **Skin Cancer** (Dermoscopy Images) - *Benign vs. Malignant*
3. **Retinal Eye Disease (OCT)** - *CNV, DME, DRUSEN, NORMAL*
4. **Tuberculosis** (Chest X-Rays)

---

## ✨ Key Features
- **Four Integrated CNN Models**: State-of-the-art accuracy trained via PyTorch.
- **Lazy Loading Architecture**: Models are instantiated only when called, allowing the entire application to run within extremely strict memory constraints (e.g., Render's 512MB free tier limit) without triggering Out-Of-Memory (OOM) errors.
- **Premium Glassmorphism UI**: Beautiful, medical-themed frontend with live image preview, loading states, and dynamic confidence progress bars.
- **Federated Learning Ready**: Exposes `/federated/info` and `/federated/status` endpoints to log global model hashes and training rounds simulated via FedAvg.
- **Full-Stack Single-Server Deployment**: FastAPI natively mounts and serves the static frontend, meaning the entire stack runs flawlessly on a single web service.

---

## 🏗️ Architecture

```text
├── backend/
│   ├── main.py                 # FastAPI router and static file mounter
│   ├── download_models.py      # Automated script to pull PyTorch models from cloud storage
│   ├── requirements.txt        # Python dependencies
│   ├── services/               # Inference logic (transforms & PyTorch model definitions)
│   │   ├── brain_predict.py
│   │   ├── skin_predict.py
│   │   ├── oct_predict.py
│   │   ├── tb_predict.py
│   │   └── federated_info.py   # Federated learning metadata endpoints
│   └── models/                 # Directory where .pt files are downloaded
├── frontend/
│   ├── index.html              # Core UI structure
│   ├── style.css               # Styling and animations
│   └── script.js               # Async fetch calls to FastAPI
└── Model/                      # Jupyter notebooks used for initial model training & Federated Learning simulation
```

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME/backend
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the Pre-trained Models
Because PyTorch model weights (`.pt` files) are too large for standard GitHub repositories (over 100MB), we host them securely on Google Drive. 
Run the automated script to pull them down into the `backend/models` directory:
```bash
python download_models.py
```

### 4. Start the Application
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to `http://127.0.0.1:8000` to interact with the web app!

---

## ☁️ Deployment (Render.com)

This project is fully configured for automated deployment on **Render's Free Tier**.

1. Connect your GitHub repository to Render.
2. Create a new **Web Service**.
3. **Root Directory**: `backend`
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `python download_models.py && uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Health Check Path**: `/health`

Render will automatically install the environment, run the python script to download the PyTorch weights into memory, and launch the full-stack application!

---

## 🔬 Federated Learning Context
This application integrates the results of a **Federated Learning** simulation. In the `/Model` directory, Jupyter notebooks demonstrate how these CNNs can be trained locally on edge devices (simulated clients) and aggregated using the `FedAvg` algorithm. This ensures raw patient imagery never leaves the hospital, preserving patient privacy while still contributing to a highly accurate global model.

The API endpoints `/federated/info` and `/federated/status` provide transparency into this process, outputting the MD5 hash of the global models and the total number of aggregation rounds completed.
