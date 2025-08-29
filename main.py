import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import pickle

# Load dataset
data = pd.read_csv('heart.csv')

# Print column names to verify
print(data.columns)

# Preprocessing (handle missing values, encode categories, etc.)
data = data.dropna()  # Remove rows with missing values

# Encode categorical columns
data = pd.get_dummies(data, columns=['sex', 'diabetes'])

# Feature engineering: create interaction features
data['age_cholesterol'] = data['age'] * data['cholesterol']
data['bp_cholesterol'] = data['bp'] * data['cholesterol']

# Features and target
X = data.drop('heart_disease', axis=1)
y = data['heart_disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Train model with Grid Search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

# Train XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0]
}

# Initialize Grid Search for XGBoost
xgb_grid_search = GridSearchCV(estimator=xgb.XGBClassifier(eval_metric='logloss'), param_grid=xgb_param_grid, cv=5, n_jobs=-1, verbose=2)

# Train XGBoost model with Grid Search
xgb_grid_search.fit(X_train, y_train)

# Best parameters for XGBoost
print("Best parameters for XGBoost found: ", xgb_grid_search.best_params_)

# Evaluate
y_pred_xgb_grid = xgb_grid_search.predict(X_test)
print(classification_report(y_test, y_pred_xgb_grid))

# Initialize ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('rf', grid_search.best_estimator_),
    ('xgb', xgb_grid_search.best_estimator_)
], voting='soft')

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('rf', grid_search.best_estimator_),
    ('xgb', xgb_grid_search.best_estimator_)
], voting='soft')

# Cross-validation
cv_scores = cross_val_score(ensemble_model, X_resampled, y_resampled, cv=5)
print("Cross-validation scores: ", cv_scores)
print("Average cross-validation score: ", cv_scores.mean())

# Train model with resampled data
ensemble_model.fit(X_resampled, y_resampled)

# Evaluate ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)
print(classification_report(y_test, y_pred_ensemble))

# Save the trained ensemble model
with open('ensemble_model.pkl', 'wb') as model_file:
    pickle.dump(ensemble_model, model_file)