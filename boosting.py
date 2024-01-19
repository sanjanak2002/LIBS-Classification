# boosting.py

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_gradient_boosting(X_train, y_train):
    gb_classifier = GradientBoostingClassifier()
    gb_classifier.fit(X_train, y_train)
    return gb_classifier

def train_hist_gradient_boosting(X_train, y_train):
    hist_gb_classifier = HistGradientBoostingClassifier()
    hist_gb_classifier.fit(X_train, y_train)
    return hist_gb_classifier

def train_xgboost(X_train, y_train):
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    return xgb_classifier

def train_lightgbm(X_train, y_train):
    lgbm_classifier = LGBMClassifier()
    lgbm_classifier.fit(X_train, y_train)
    return lgbm_classifier
