# Heart Disease Prediction

An AI-powered heart disease prediction system using machine learning. The entire data science pipeline -- from exploratory data analysis to model training -- was developed in a [Google Colab notebook](Heart_Disease_prediction_.ipynb) using AI-assisted code generation, then deployed as an interactive web app with Streamlit.

**Live App:** [heart-failure-prediction0.streamlit.app](https://heart-failure-prediction0.streamlit.app/)

---

## About

Cardiovascular disease is one of the leading causes of death worldwide. This project builds a binary classification model to predict the presence of heart disease based on patient medical data. The core ML workflow (EDA, preprocessing, model comparison, and export) was developed entirely in the accompanying Jupyter notebook with AI assistance, demonstrating how AI tools can accelerate the data science process.

---

## Notebook Pipeline (AI-Assisted)

The notebook `Heart_Disease_prediction_.ipynb` covers the full ML workflow:

### 1. Dataset
- **Source:** [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** 918 samples, 12 features
- **Target:** `HeartDisease` (0 = No, 1 = Yes)

### 2. Exploratory Data Analysis
- Class distribution: 508 positive vs 410 negative cases
- Identified 172 zero-values in `Cholesterol` and zero-values in `RestingBP` -- replaced with column means
- Statistical summaries and data type inspection

### 3. Data Preprocessing
- Replaced invalid zero values with column means
- One-hot encoded categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) using `pd.get_dummies`
- Final feature set: **15 features** (after `drop_first=True`)

### 4. Model Training & Comparison

Five classifiers were trained with an 80/20 train-test split (`random_state=42`) and `StandardScaler` applied to all features:

| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| Logistic Regression  | 86.41%   | 87.92%   |
| SVM                  | 85.87%   | 87.62%   |
| KNN                  | 85.33%   | 87.08%   |
| Naive Bayes          | 84.78%   | 86.14%   |
| Decision Tree        | 80.43%   | 82.18%   |

**Best model:** Logistic Regression (86.41% accuracy, 87.92% F1 score)

### 5. Model Export
- `LR_heart.pkl` -- Trained Logistic Regression model
- `scaler.pkl` -- Fitted StandardScaler
- `columns.pkl` -- Feature column names

All exported using `joblib`.

---

## Streamlit Web App

The deployed app (`app.py`) loads the exported model artifacts and provides:

- **Sidebar inputs** for patient data (age, blood pressure, cholesterol, ECG results, etc.)
- **Prediction output** with a binary High/Low risk classification
- **Gauge chart** showing the probability score (Plotly)
- **Risk factor bar chart** visualizing input values
- **Detailed metrics** (age, BP, cholesterol, heart rate)
- **Health recommendations** based on input thresholds

---

## Project Structure

```
heart/
├── app.py                  # Streamlit web application
├── Heart_Disease_prediction_.ipynb  # Full ML pipeline (AI-generated)
├── LR_heart.pkl            # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler for feature normalization
├── columns.pkl             # Feature column names
└── requirements.txt        # Python dependencies
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- **ML:** scikit-learn (Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree)
- **Data:** pandas, numpy
- **Web App:** Streamlit
- **Visualization:** Plotly
- **Development:** Google Colab, AI-assisted notebook generation

---

## Disclaimer

This tool is for educational purposes only and is not a substitute for professional medical diagnosis or advice.
