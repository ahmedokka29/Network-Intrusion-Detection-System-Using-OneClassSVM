from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import SelectKBest, f_classif
import ipaddress
import warnings
warnings.filterwarnings('ignore')

csv_path = "./sampled_NF-CSE-CIC-IDS2018-v2.csv"
features_path = "/NetFlow_v2_Features.csv"

file_path = csv_path
df = pd.read_csv(file_path)


print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['Label'].value_counts()}")
print(f"\nAttack categories:\n{df['Attack_Category'].value_counts()}")


print(f"\nMissing values:\n{df.isnull().sum().sum()}")


print("\nNumeric features summary:")
print(df.describe())


plt.figure(figsize=(14, 8))
sns.boxplot(x='Attack_Category', y='FLOW_DURATION_MILLISECONDS', data=df)
plt.yscale('log')
plt.title('Flow Duration by Attack Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(16, 14))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return np.nan


df['IPV4_SRC_ADDR_INT'] = df['IPV4_SRC_ADDR'].apply(ip_to_int)
df['IPV4_DST_ADDR_INT'] = df['IPV4_DST_ADDR'].apply(ip_to_int)


columns_to_drop = ['IPV4_SRC_ADDR',
                   'IPV4_DST_ADDR', 'Attack', 'Attack_Category']
X = df.drop(columns=columns_to_drop + ['Label'], errors='ignore')


y = df['Label']


X = X.fillna(0)


selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_features_idx = selector.get_support(indices=True)
selected_features = X.columns[selected_features_idx]
print("\nSelected features:", selected_features.tolist())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


X_train_normal = X_train[y_train == 0]


print("\nTraining OneClassSVM...")
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm.fit(X_train_normal)


decision_scores = ocsvm.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, -decision_scores)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = -thresholds[optimal_idx]


y_pred_test = np.where(decision_scores < optimal_threshold, 1, 0)


accuracy = accuracy_score(y_test, y_pred_test)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_test, average='binary')

print("\nModel Performance with Optimal Threshold:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Optimal Threshold')
plt.tight_layout()
plt.show()


print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_test,
      target_names=['Normal', 'Attack']))


if hasattr(ocsvm, 'coef_'):
    feature_importance = np.abs(ocsvm.coef_[0])
    feature_names = selected_features.tolist()

    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)),
             feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance (OneClassSVM)')
    plt.tight_layout()
    plt.show()


def custom_score(estimator, X):
    pred = estimator.predict(X)
    pred_binary = np.where(pred == -1, 1, 0)

    return f1_score(y_test_for_tuning, pred_binary)


X_tune, X_holdout, y_tune, y_holdout = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)
X_test_for_tuning, y_test_for_tuning = X_tune, y_tune


param_grid = {
    'nu': [0.01, 0.05, 0.1],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}


grid_search = GridSearchCV(
    OneClassSVM(),
    param_grid=param_grid,
    scoring=make_scorer(custom_score, greater_is_better=True),
    cv=3,
    verbose=1
)


X_tune_normal = X_tune[y_tune == 0]
grid_search.fit(X_tune_normal, y_tune[y_tune == 0])

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")


best_model = grid_search.best_estimator_
y_pred_holdout = best_model.predict(X_holdout)
y_pred_holdout_binary = np.where(y_pred_holdout == -1, 1, 0)

print("\nBest Model Performance on Holdout Set:")
print(classification_report(y_holdout, y_pred_holdout_binary,
      target_names=['Normal', 'Attack']))
