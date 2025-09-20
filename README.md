# 🧪 GNN Molecular Property Prediction

A Graph Neural Network project for predicting molecular energies using PyTorch Geometric.

## 📖 Overview

This project implements and benchmarks multiple GNN architectures for molecular property prediction, with a focus on invariance to rotations and translations.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirement.txt
```
### Basic usage
```python
from models import CGCNN
from training import MolecularTrainer

# Create model
model = CGCNN(node_fea_dim=64, invariant=True, num_layers=3)

# Train model
trainer = MolecularTrainer(model)
history = trainer.train(train_loader, val_loader, epochs=100)
```
### 📁 Project Structure
- `src/`
    -`data/ `         # Data loading and preprocessing
    -`models/`        # GNN architectures (CGCNN, Attention GNN, MPNN)
    -`training/`     # Training utilities and hyperparameter tuning
    -`visualization/` # Molecular visualization tools
    -`notebooks/`     # Example Jupyter notebooks

## 🧠 Available Models
-CGCNN - Crystal Graph Convolutional Neural Network
-Enhanced CGCNN - With attention mechanisms
-Graph Attention Networks - Multi-head attention
-MPNN - Message Passing Neural Network
## 🎯 Key Features
-Invariance studies (rotation/translation)
-Hyperparameter optimization with Optuna
-Molecular visualization with ASE
-Comprehensive benchmarking suite
-Reproducible experiment configurations
## 📊 Example Results
```python
# Compare model performance
from training import BenchmarkingSuite

benchmark = BenchmarkingSuite(test_loader)
benchmark.evaluate_model(model, "CGCNN")
# Output: MAE: 0.123, RMSE: 0.156, R²: 0.945
```
## 👨‍💻 Author

- LinkedIn: [Junming Ma](https://linkedin.com/in/junmingma)
yesterday

- Email: louisama778@gmail.com

*This project was completed as part of the Deep Learning (1RT720) course at Uppsala University.*
