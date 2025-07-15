# Fraud Detection - Machine Learning Project

## 1. Importing Libraries and Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('Fraud.csv', nrows=100000)  # limit to 100k rows

## 2. Data Cleaning
# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# One-hot encode 'type' column
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Correlation Matrix
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix (Numeric Features Only)")
plt.tight_layout()
plt.savefig('images/heatmap.png', bbox_inches='tight')  # ✅ Save heatmap
plt.show()

## 3. Feature Engineering and Selection
# No additional feature engineering needed in this basic version

## 4. Preprocessing
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Balancing data with SMOTE
X_sample = X.sample(n=min(200000, len(X)), random_state=42)
y_sample = y.loc[X_sample.index]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_sample, y_sample)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## 5. Model Building
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## 6. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, model.predict_proba(X_test)[:,1])))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.savefig('images/roc_curve.png', bbox_inches='tight')  # ✅ Save ROC Curve plot
plt.show()

## 7. Interpretation of Important Features
importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

print("\nTop Important Features:")
print(feature_df.head())
# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(data=feature_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig('images/feature_importance.png', bbox_inches='tight')  # ✅ Save feature importance plot
plt.show()


## 8. Recommendations & Monitoring Impact
# (Add in your report / final documentation)
# - Flag accounts with frequent TRANSFER + CASH_OUT quickly
# - Track accounts sending >200k often
# - Monitor feature importance for retraining
