#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 


# In[3]:


organic = pd.read_csv('organics.csv')


# In[5]:


organic


# In[6]:


organic.info()


# In[7]:


organic.isnull().sum()


# In[8]:


organic.isna().sum() / len(organic) * 100


# In[11]:


organic_final2 = organic.dropna()

organic_final2


# In[17]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset (Replace 'your_dataframe' with the actual variable name)
organic_final = organic_final2  

# Ensure 'TargetBuy' is the target variable
y = organic_final['TargetBuy']

# Drop potential data leakage variables
drop_cols = ['ID', 'TargetBuy', 'TargetAmt']  # Ensure TargetAmt is not in features
X = organic_final.drop(columns=drop_cols)

# Identify categorical columns
categorical_cols = ['DemCluster', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass']

# Apply Label Encoding instead of One-Hot Encoding to avoid excessive feature expansion
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store encoders if needed later

# Check for correlations before modeling (Debugging Step)
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Train-Test Split (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numerical features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model with L2 regularization (prevents overfitting)
log_model = LogisticRegression(max_iter=1000, C=0.1)
log_model.fit(X_train, y_train)

# Predict on test data
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]  # Get probability scores for ROC-AUC

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print performance metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# In[15]:


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': log_model.coef_[0]
}).sort_values(by='Importance', ascending=False)
print(feature_importance)


# In[19]:


#Decison tree
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Train Optimized Decision Tree Classifier
dt_model = DecisionTreeClassifier(
    max_depth=4,            # Restrict depth to prevent overfitting
    min_samples_split=50,   # Minimum samples required to split a node
    min_samples_leaf=20,    # Minimum samples required in a leaf node
    criterion='gini',       # Split based on Gini impurity (can also use 'entropy')
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predict on test data
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]  # Get probability scores for ROC-AUC

# Evaluate Decision Tree performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)

# Print performance metrics
print(f"Decision Tree - Accuracy: {accuracy_dt:.4f}")
print(f"Decision Tree - Precision: {precision_dt:.4f}")
print(f"Decision Tree - Recall: {recall_dt:.4f}")
print(f"Decision Tree - AUC-ROC: {roc_auc_dt:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(
    dt_model, 
    feature_names=X.columns, 
    class_names=["Non-Organic", "Organic"], 
    filled=True, 
    rounded=True
)
plt.title("Optimized Decision Tree Visualization")
plt.show()


# In[20]:


#Random Forest 
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    max_depth=6,           # Limit tree depth to prevent overfitting
    min_samples_split=50,  # Minimum samples required to split a node
    min_samples_leaf=20,   # Minimum samples required in a leaf node
    random_state=42,
    n_jobs=-1,             # Use all CPU cores for faster computation
    class_weight="balanced"  # Adjusts for imbalanced classes (if needed)
)

rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# Evaluate Random Forest performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Print performance metrics
print(f"Random Forest - Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest - Precision: {precision_rf:.4f}")
print(f"Random Forest - Recall: {recall_rf:.4f}")
print(f"Random Forest - AUC-ROC: {roc_auc_rf:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance from Random Forest
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print(feature_importance_rf)

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_importance_rf["Feature"], feature_importance_rf["Importance"], color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()  # Invert y-axis to show most important features on top
plt.show()


# In[22]:


pip install xgboost


# In[23]:


# Import necessary libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=200,       # Number of boosting rounds (trees)
    learning_rate=0.1,      # Controls step size (lower values prevent overfitting)
    max_depth=4,            # Limits tree depth to prevent overfitting
    subsample=0.8,          # Fraction of samples used per tree (reduces variance)
    colsample_bytree=0.8,   # Fraction of features used per tree
    eval_metric='auc',      # Evaluates model performance with AUC
    use_label_encoder=False,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Get probability scores for ROC-AUC

# Evaluate Gradient Boosting Performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Print performance metrics
print(f"XGBoost - Accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost - Precision: {precision_xgb:.4f}")
print(f"XGBoost - Recall: {recall_xgb:.4f}")
print(f"XGBoost - AUC-ROC: {roc_auc_xgb:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))


# In[24]:


import matplotlib.pyplot as plt
import pandas as pd

# Get feature importance
feature_importance_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print(feature_importance_xgb)

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_importance_xgb["Feature"], feature_importance_xgb["Importance"], color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - XGBoost")
plt.gca().invert_yaxis()  # Show most important features at the top
plt.show()


# In[25]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'max_depth': [3, 4, 6],  # Maximum tree depth
    'subsample': [0.7, 0.8, 1.0],  # Sample ratio of training instances
    'colsample_bytree': [0.7, 0.8, 1.0]  # Fraction of features used per tree
}

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False, random_state=42)

# GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='roc_auc',  # Optimizing for AUC-ROC
    cv=3,  # 3-fold cross-validation
    verbose=1, 
    n_jobs=-1  # Use all CPU cores
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best hyperparameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train XGBoost with best parameters
best_xgb = grid_search.best_estimator_

# Predict on test data
y_pred_best_xgb = best_xgb.predict(X_test)
y_prob_best_xgb = best_xgb.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# Evaluate Fine-Tuned XGBoost Performance
accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)
precision_best_xgb = precision_score(y_test, y_pred_best_xgb)
recall_best_xgb = recall_score(y_test, y_pred_best_xgb)
roc_auc_best_xgb = roc_auc_score(y_test, y_prob_best_xgb)

# Print performance metrics
print(f"Fine-Tuned XGBoost - Accuracy: {accuracy_best_xgb:.4f}")
print(f"Fine-Tuned XGBoost - Precision: {precision_best_xgb:.4f}")
print(f"Fine-Tuned XGBoost - Recall: {recall_best_xgb:.4f}")
print(f"Fine-Tuned XGBoost - AUC-ROC: {roc_auc_best_xgb:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best_xgb))

