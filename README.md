# 💳 Fraud Detection with Machine Learning

This project focuses on detecting fraudulent financial transactions using supervised machine learning. The dataset used includes over 6 million anonymized transaction records. A balanced and interpretable model was developed using SMOTE and a Random Forest Classifier.

---

## 📂 Project Structure

- `fraud_detection_solution.ipynb` – Jupyter Notebook containing all steps
- `fraud.csv` – Dataset file (not included in repo for size/privacy)
- `images/` – Folder containing output visualizations (ROC Curve, Confusion Matrix, Feature Importances)
- `LICENSE` – MIT License

---

## 📊 Dataset Summary

- **Records**: 6,362,620 transactions
- **Target**: `isFraud` (1: Fraud, 0: Not Fraud)
- **Features**:  
  - `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`,  
    `oldbalanceDest`, `newbalanceDest`, etc.

---

## ⚙️ Workflow

1. **Data Cleaning**
   - Removed irrelevant columns (`nameOrig`, `nameDest`)
   - One-hot encoded `type`
   - Checked for null values

2. **Feature Engineering**
   - No additional features added in this version
   - Considered correlation matrix to examine multicollinearity

3. **Data Balancing**
   - Applied SMOTE to balance the minority class (fraudulent transactions)

4. **Modeling**
   - Algorithm: `RandomForestClassifier`
   - Train-test split: 70/30
   - StandardScaler used for normalization

5. **Evaluation**
   - Classification Report (Precision, Recall, F1)
   - Confusion Matrix
   - ROC-AUC Score
   - Feature Importance Plot

---

## 🔍 Model Performance
After running the fraud detection pipeline on the dataset, the model achieved outstanding performance:


### 📊 Confusion Matrix

[[29789 38]
[ 10 30094]]


- **True Negatives (TN):** 29,789  
- **False Positives (FP):** 38  
- **False Negatives (FN):** 10  
- **True Positives (TP):** 30,094  

---

### 🧮 Classification Report
```
                 precision  recall  f1-score   support

           0       1.00      1.00      1.00     29827
           1       1.00      1.00      1.00     30104

    accuracy                           1.00     59931
   macro avg       1.00      1.00      1.00     59931
weighted avg       1.00      1.00      1.00     59931
```

### 📈 ROC-AUC Score

0.99999

---

### 🔍 Top 5 Important Features

| Feature           | Importance |
|------------------|------------|
| `step`           | 0.2327     |
| `oldbalanceOrg`  | 0.1638     |
| `type_TRANSFER`  | 0.1487     |
| `newbalanceOrig` | 0.1270     |
| `type_PAYMENT`   | 0.0899     |

---

## 📸 Output Samples

### 🔲 Confusion Matrix  
<img width="1600" height="861" alt="Output1" src="https://github.com/user-attachments/assets/ea54e9ac-7125-4f46-9908-f630f4e2c314" />

### 📉 ROC Curve  
<img width="567" height="455" alt="roc_curve" src="https://github.com/user-attachments/assets/4f4bee7e-4365-4158-924a-759b57187298" />

### 🧮 Feature Importance  
<img width="990" height="590" alt="feature_importance" src="https://github.com/user-attachments/assets/dee3ad60-690d-4a44-bd30-365391a6bf90" />

### 🧾 Model Output
<img width="1234" height="828" alt="terminal output" src="https://github.com/user-attachments/assets/cc503c84-cb5c-4c3c-b67d-b14948fb48ab" />

---

## 🔑 Key Takeaways

- Most fraudulent transactions are of type **TRANSFER** or **CASH_OUT**
- Imbalance in origin/destination balances is a strong fraud indicator
- Real-time flagging and transaction pattern analysis can significantly help prevent fraud

---

## 🚀 Future Improvements

- Implement **XGBoost** or **LightGBM** for higher accuracy
- Use **SHAP** for better model explainability
- Deploy with **FastAPI** or **Flask** for real-time inference
- Schedule retraining pipelines with updated data

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♀️ Author

**Allmin Fatima**  
🔗 [GitHub Profile](https://github.com/Allminfatima)  
