# 🏦 Loan Default Prediction with Machine Learning

This project uses machine learning models (Random Forest & Logistic Regression) to predict whether a loan application should be approved or not based on applicant financial and demographic features.

---

## 📂 Dataset

The dataset used in this project is sourced from Kaggle (https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset).
**Files:**
- `loan-train.csv`: Training data with target labels
- `loan-test.csv`: Test data without target labels

## 📊 Features used 
- Gender
- Marital Status
- Dependents
- Education
- Employment Status
- Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

---

## 🛠️ Tools
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Pickle (model serialization)
- Streamlit (for future deployment)

---

## ✅ Models
| Model               | Accuracy | ROC AUC |
|---------------------|----------|---------|
| Logistic Regression | 85.4%    | 0.81    |
| Random Forest       | 82.1%    | 0.82    |

Random Forest performed slightly better in terms of AUC, but Logistic Regression had a higher accuracy overall.

---

## 🚀 Future Work
- Streamlit app deployment
- Hyperparameter tuning
- Better imputation strategies

## 📁 Files
- `loan_predictions_rf.csv`: Predicted loan statuses
- `random_forest_model.pkl`: Trained model
- `notebook.ipynb`: Model training & evaluation

