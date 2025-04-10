# 🏠 House Price Prediction using PyTorch

This project implements a deep learning model using PyTorch to predict house prices based on a cleaned housing dataset. The dataset is first preprocessed and scaled before being passed through a custom-built neural network. The model is trained to minimize Mean Squared Error (MSE) and evaluated using MSE and R² Score.

> 📝 I have also created a **Data Wrangling file** that was used to clean and prepare the dataset before model training.

---

## 📁 Dataset

The dataset used for this project is `housing_clean.csv`, which was preprocessed to remove unnecessary columns and normalize features.

---

## 🛠️ Technologies Used

- Python 🐍  
- PyTorch 🔥  
- Scikit-learn 🧪  
- Matplotlib 📊  
- NumPy 🧮  
- Pandas 📑  

---

## 📈 Model Architecture

```python
HousePriceModel(
    (model): Sequential(
        (0): Linear(in_features=..., out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=1, bias=True)
    )
)
```

---

## 🚀 How It Works

**Load and Clean Data:**
Load the housing dataset and remove unwanted columns.

**Split and Scale:**
The data is split into training and testing sets and scaled using StandardScaler.

**Convert to PyTorch Tensors:**
Data is converted into tensors for use in the PyTorch model.

**Model Definition:**
A simple feedforward neural network with one hidden layer and ReLU activation is used.

**Training Loop:**
The model is trained using MSE loss and the Adam optimizer.

**Evaluation:**
MSE and R² Score are used to evaluate the model's performance.

**Visualization:**
 - Plots are generated to show:
 - Actual vs Predicted Prices
 - First 50 samples: Actual vs Predicted
 - Training Loss over Epochs

---

## 📊 Evaluation Metrics

 - Mean Squared Error (MSE): 0.547227680683136
 - R² Score: 0.46674455303521833

---

## 📷 Visualizations

📍 Actual vs Predicted Prices
Scatter plot showing the actual and predicted house prices.

📍 First 50 Samples
Line plot of the actual vs predicted prices for the first 50 test samples.

📍 Training Loss Over Epochs
Line plot showing how the training loss decreased over 100 epochs.

---

## 📝 Author Note
I have also created a Data Wrangling file that was used to clean and prepare the dataset before model training.

---

## 🔍 Future Improvements

 - Hyperparameter tuning
 - Cross-validation
 - Deeper network architectures
 - Save and load trained models
