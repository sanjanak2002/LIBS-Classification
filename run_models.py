# run_models.py

from preprocessing import load_data, split_data, scale_data
from lda import LDAClassifier
from knn import KNNClassifier
from boosting import (
    train_gradient_boosting,
    train_hist_gradient_boosting,
    train_xgboost,
    train_lightgbm
)
from sklearn.metrics import accuracy_score

# Load and preprocess data
file_path = 'your_data.csv'  # Replace with your actual file path
X, y = load_data(file_path)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Train and evaluate models
lda_classifier = LDAClassifier()
lda_classifier.fit(X_train_scaled, y_train)
lda_predictions = lda_classifier.predict(X_test_scaled)
lda_accuracy = accuracy_score(y_test, lda_predictions)
print(f"LDA Accuracy: {lda_accuracy * 100:.2f}%")

knn_classifier = KNNClassifier(k=3)
knn_classifier.fit(X_train_scaled, y_train)
knn_predictions = knn_classifier.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

gb_classifier = train_gradient_boosting(X_train_scaled, y_train)
gb_predictions = gb_classifier.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")

hist_gb_classifier = train_hist_gradient_boosting(X_train_scaled, y_train)
hist_gb_predictions = hist_gb_classifier.predict(X_test_scaled)
hist_gb_accuracy = accuracy_score(y_test, hist_gb_predictions)
print(f"Histogram-Based GB Accuracy: {hist_gb_accuracy * 100:.2f}%")

xgb_classifier = train_xgboost(X_train_scaled, y_train)
xgb_predictions = xgb_classifier.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

lgbm_classifier = train_lightgbm(X_train_scaled, y_train)
lgbm_predictions = lgbm_classifier.predict(X_test_scaled)
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)
print(f"LGBM Accuracy: {lgbm_accuracy * 100:.2f}%")
