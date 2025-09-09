 # Load Prediction on PDV Electric Power

## Predicting electric power load from time series data using PyTorch GRUs on Mac systems with MPS acceleration.


### 📖 Project Overview

This project builds and trains a time series forecasting model for electric load prediction.
The dataset is sourced from Kaggle.
The model is based on GRU (Gated Recurrent Unit) networks, which are well-suited for sequential data.
Training is accelerated using MPS (Metal Performance Shaders) on Apple Silicon (M1/M2/M3) Macs.

Why this matters:
Accurate load prediction helps in managing power demand, reducing operational costs, and ensuring grid stability.


### 🗂️ Project Structure
.
├── data/                # datasets
├── notebooks/           # exploratory notebooks
├── model/               # saved model
├── requirements.txt     # dependencies
└── README.md            # this file


### ⚙️ Installation
1.	Clone this repo:
`git clone https://github.com/Calaabdul/Energy-consumption-with-GRU.git
cd Energy-consumption-with-GRU`

2.	Create a virtual environment and install requirements:
`conda create -n dl_project Python=3.10 -y
cobda activate dl_project
pip install -r requirements.txt`

3.	Verify PyTorch with MPS:

`import torch
print(torch.backends.mps.is_available())  # should return True`


### 🚀 Training the Model

Run training with:
python src/train.py --epochs 30 --batch_size 32 --lr 0.001

Default device selection automatically picks MPS if available, otherwise falls back to CPU.

### Model Details
•	Architecture: GRU → Fully Connected Layer
•	Input shape: (batch_size, sequence_length, features)
•	Loss function: MSELoss (regression task)
•	Optimizer: Adam (with betas for momentum-like behavior)



### ⚡ Key Features

✅ Handles sequence length + batch training
✅ Compatible with MPS on Mac (fast training vs CPU)
✅ Prevents device mismatch errors by ensuring hidden states are initialized on mps



### Results (example)
	•	Loss decreasing steadily across epochs
	•	Predicted load closely follows actual load curve
