# Student Performance Analysis Using Hybrid Algorithm in Machine Learning

## 📌 Project Overview
This project presents a **hybrid machine learning framework** for predicting student performance by combining:

- **DBSCAN Clustering** (unsupervised learning)
- **Random Forest Regressor** (for score prediction)
- **Logistic Regression Classifier** (for performance category prediction)

The goal is to improve student performance prediction by discovering **hidden behavioral patterns** among students and using those patterns as an additional feature in supervised models. :contentReference[oaicite:1]{index=1}

---

## 🚀 Key Features

- Automated **DBSCAN hyperparameter tuning** using silhouette score
- Detection of **student behavioral clusters**
- Hybrid **Random Forest Regression** for predicting exam scores
- Hybrid **Logistic Regression** for classifying student performance categories
- Feature importance analysis
- Cross-validation for model robustness
- Statistical significance testing using paired t-test

---

## 🧠 Problem Statement
Traditional machine learning models usually predict student performance directly from raw features such as attendance, study hours, and previous scores.

This project improves that approach by:

- Finding **natural student groups (clusters)** using DBSCAN
- Using the cluster label as a **new engineered feature**
- Feeding this enriched dataset into supervised models

This helps the model capture **hidden behavioral patterns** that may not be visible from individual features alone. :contentReference[oaicite:2]{index=2}

---

## 📂 Dataset
The project uses the **StudentPerformanceFactors** dataset.

### Main Features
- `Hours_Studied`
- `Attendance`
- `Previous_Scores`
- `Tutoring_Sessions`
- `Sleep_Hours`
- `Motivation_Level`
- `Family_Income`
- `Teacher_Quality`
- `Exam_Score` (Target Variable)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handle missing values
- Encode categorical features
- Standardize numerical features
- Train-test split (80:20)

### 2. DBSCAN Clustering
- Applied on standardized data
- Used for discovering hidden student behavior groups
- Cluster labels appended as a new feature: **`DBSCAN_Cluster`**

### 3. Automated Hyperparameter Tuning
DBSCAN parameters tested:

- `eps = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]`
- `min_samples = [3, 5, 7, 10, 15]`

Best result:
- **eps = 1.0**
- **min_samples = 5**
- **Silhouette Score = 0.102** :contentReference[oaicite:3]{index=3}

### 4. Hybrid Models
#### Hybrid Random Forest Regressor
- Predicts exact exam score
- Uses original features + `DBSCAN_Cluster`

#### Hybrid Logistic Regression Classifier
- Predicts performance category
- Uses original features + `DBSCAN_Cluster`

---

## 📊 Results

### 🔹 DBSCAN Best Parameters
- **eps = 1.0**
- **min_samples = 5**
- **Silhouette Score = 0.102**

### 🔹 Random Forest Regression Performance
| Model | MSE | RMSE | R² Score |
|------|-----|------|----------|
| Standalone Random Forest | 4.92 | 2.22 | 0.81 |
| Hybrid Random Forest (with DBSCAN) | 4.45 | 2.11 | 0.83 |

**Improvement:**
- ΔMSE = **0.47**
- ΔRMSE = **0.11**
- ΔR² = **0.02** :contentReference[oaicite:4]{index=4}

### 🔹 Logistic Regression Classification Performance
- **Accuracy = 82.3%**
- **Precision (Macro) = 0.81**
- **Recall (Macro) = 0.80**
- **F1-Score (Macro) = 0.80** :contentReference[oaicite:5]{index=5}

### 🔹 Cross Validation
- **5-Fold CV MSE = 4.88 ± 0.12** :contentReference[oaicite:6]{index=6}

---

## 📈 Feature Importance
Top important features in the hybrid model:

1. **Attendance** – 33.4%
2. **Hours_Studied** – 23.9%
3. **Previous_Scores** – 9.8%
4. **DBSCAN_Cluster** – 7.2%
5. **Tutoring_Sessions** – 3.9%
6. **Sleep_Hours** – 3.5%
7. **Motivation_Level** – 3.2% :contentReference[oaicite:7]{index=7}

---

## 🏷️ Student Clusters Identified
DBSCAN identified meaningful student groups:

- **Cluster 0** → High Engagement Students
- **Cluster 1** → Moderate Engagement Students
- **Cluster 2** → Low Engagement Students
- **Noise (-1)** → Unique / Outlier students

These clusters help in understanding student behavior and planning targeted interventions. :contentReference[oaicite:8]{index=8}

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn** (optional for visualization)
- **Jupyter Notebook / Google Colab**

---
