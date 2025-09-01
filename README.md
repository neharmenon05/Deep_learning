# Neural Network Implementation & Comparative Study

## 📌 Overview
This project implements a **scratch-built Multi-Layer Perceptron (MLP)** using NumPy and compares its performance against other models, including **Keras Neural Networks, XGBoost, Linear Regression, and Random Forest**.  

The study explores how different **architectures, activation functions, and optimizers** influence performance and robustness under varying data conditions.

---

## 🎯 Goals
- Implement a **neural network from scratch** with forward and backward propagation.
- Compare optimizers: **SGD, Momentum, Adam**.
- Study the effect of **depth and activation functions** (ReLU, Tanh, LeakyReLU).
- Analyze robustness with:
  - **Noisy training data**
  - **Reduced training samples**
- Benchmark against:
  - **Linear Regression**
  - **Random Forest**
  - **XGBoost**
  - **Keras Neural Network**

---

## ⚙️ Implemented Components

### 🔹 Optimizers
- **SGD** – simple gradient descent update.  
- **Momentum** – accelerated updates using past gradients.  
- **Adam** – adaptive learning rate + momentum.

### 🔹 Activation Functions
- **ReLU** – efficient, avoids vanishing gradients.  
- **Tanh** – zero-centered, outputs in [-1, 1].  
- **Leaky ReLU** – variant of ReLU with small negative slope to prevent dead neurons.

### 🔹 Loss Function
- **MSE with L2 regularization** – balances error minimization with generalization.

### 🔹 Evaluation Metrics
- **MSE** – Mean Squared Error  
- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **R²** – Coefficient of Determination

---

## 📊 Workflow
1. **Forward Propagation** – compute layer outputs.  
2. **Loss Calculation** – quantify prediction error.  
3. **Backpropagation** – compute gradients.  
4. **Optimizer Step** – update weights.  
5. **Training Loop** – repeat across epochs.  
6. **Evaluation** – compare performance across models.

---

## 🔍 Key Insights
- Custom MLP achieves competitive performance but depends on optimizer and activation choice.  
- **Adam** generally provides the most stable and fastest convergence.  
- **ReLU and Leaky ReLU** perform better than Tanh in deeper architectures.  
- Traditional ML models like **Random Forest** and **XGBoost** can rival neural networks on structured/tabular data.

---

## 📂 Repository Structure
.
├── NN_uupdates.ipynb # Main Jupyter notebook (implementation & experiments)
├── data/ # Dataset(s) used
├── results/ # Metrics, plots, and comparison outputs
└── README.md # Project overview


---

## 🛠️ Dependencies
- Python 3.x  
- NumPy  
- Matplotlib  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras  

Install required packages:
```bash
pip install numpy matplotlib scikit-learn xgboost tensorflow


## Run instruction
- Clone the repository and navigate to the project directory.
- Open the Jupyter notebook:
    jupyter notebook NN_uupdates.ipynb
- Run all cells to train models and view results.
