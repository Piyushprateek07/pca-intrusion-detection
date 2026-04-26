# PCA-Based Intrusion Detection System (IDS)

## 📌 Overview
This project presents a machine learning-based Intrusion Detection System (IDS) that uses Principal Component Analysis (PCA) for dimensionality reduction combined with multiple classification algorithms. The system focuses not only on detection accuracy but also on cost-sensitive decision making and prediction stability.

The proposed framework is evaluated on benchmark datasets (CICIDS and UNSW) and introduces a Decision Stability Index (DSI) to measure consistency across different data clusters.

---

## 🎯 Objectives
- Improve intrusion detection performance using PCA
- Reduce feature dimensionality while preserving information
- Minimize misclassification cost using cost-sensitive threshold tuning
- Evaluate model stability using Decision Stability Index (DSI)
- Analyze performance across multiple train-test splits (0.2, 0.4, 0.6)

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handling missing values
- Removing low-variance features
- Feature scaling using StandardScaler

### 2. Dimensionality Reduction
- Principal Component Analysis (PCA)
- Retains 95% variance
- Reduces computational complexity

### 3. Machine Learning Models
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Logistic Regression
- XGBoost

### 4. Cost-Sensitive Thresholding
- Assigns higher penalty to False Negatives
- Optimizes decision threshold based on cost + MCC

### 5. Clustering-Based Analysis
- KMeans clustering
- Evaluates predictions across clusters

### 6. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)
- Decision Stability Index (DSI)

---

## 📊 Datasets Used

### 🔹 CICIDS Dataset
- Real-world network traffic dataset
- Contains various attack types and normal traffic

### 🔹 UNSW-NB15 Dataset
- Modern intrusion dataset
- More complex and challenging patterns

---

## 📈 Key Results

- PCA reduced dimensionality significantly (e.g., 71 → 23 features)
- Random Forest consistently achieved best performance
- High detection accuracy (>99% on CICIDS)
- Strong MCC and DSI values indicating robustness
- Cost-sensitive tuning reduced false negatives effectively

---

## 🧪 Experimental Setup

- Train-test splits: 0.2, 0.4, 0.6
- PCA variance retention: 95%
- Multiple classifiers evaluated
- Cost-sensitive threshold optimization applied

---

## 🔍 Key Findings

- PCA effectively improves model efficiency and performance
- Accuracy alone is not sufficient for IDS evaluation
- Cost and MCC provide better insight into model performance
- DSI reveals model stability across different data distributions
- Model performance depends on dataset complexity and training size

---

## 📌 Contribution

- PCA-based IDS framework for efficient feature reduction
- Cost-sensitive decision mechanism for realistic deployment
- Decision Stability Index (DSI) for robustness evaluation
- Multi-split experimental analysis for reliability assessment

---

## 📄 Paper

This repository is associated with a research paper on PCA-based intrusion detection.

---

## ⚙️ Requirements

- Python 3.x
- Scikit-learn
- NumPy
- Pandas
- XGBoost

---

## 🚀 Future Work

- Real-time intrusion detection system
- Deep learning integration (CNN, LSTM)
- Adaptive threshold optimization
- Deployment in production environments

---

## 👨‍💻 Author

Piyush Prateek  
MTech Computer Science Engineering  

---

## 📜 License

This project is open-source and available under the MIT License
