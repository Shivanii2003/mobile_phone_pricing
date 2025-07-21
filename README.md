# 📱 Mobile Phone Price Range Prediction

This project uses a machine learning model to predict the price range (Low, Medium, High, Very High) of mobile phones based on various features like RAM, internal memory, battery power, camera specs, and more.

## 🧠 Objective

To build a classifier that can accurately predict the **price range** of a mobile phone (0–3) using its features. This helps companies and customers make data-driven decisions.

## 🗂️ Dataset

The dataset contains 21 features and a target variable `price_range`:
- `battery_power`
- `blue`
- `clock_speed`
- `dual_sim`
- `fc` (front camera megapixels)
- `four_g`
- `int_memory`
- `m_dep` (mobile depth)
- `mobile_wt`
- `n_cores`
- `pc` (primary camera)
- `px_height`, `px_width`
- `ram`
- `sc_h`, `sc_w` (screen height & width)
- `talk_time`
- `three_g`
- `touch_screen`
- `wifi`
- `price_range` (Target: 0=Low, 1=Medium, 2=High, 3=Very High)

## 📊 Exploratory Data Analysis

- Checked data types, missing values.
- Visualized correlations between RAM, battery power, and price range.
- Performed feature selection and scaling.

## 🧪 Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## ✅ Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report

## 📌 Highlights

- Achieved **high accuracy** using Random Forest and SVM.
- Performed data cleaning, feature scaling (`StandardScaler`), and model comparison.
- Provided predictions for new samples.

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivanii2003/mobile_phone_pricing
   cd mobile-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook UM_mobile_pricing.ipynb
   ```

## 📎 Sample Prediction Code

```python
import numpy as np
sample = np.array([[1043, 1, 2.5, 1, 5, 1, 10, 0.6, 145, 4, 13, 980, 1240, 2549, 16, 8, 15, 1, 1, 1]])
sample_scaled = scaler.transform(sample)
model.predict(sample_scaled)
```

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/iabhishekofficial/mobile-price-classification)
- Scikit-learn Documentation
