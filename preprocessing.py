# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def prepare_features(df, target_col='label'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Feature Selection (Select top 20 features based on ANOVA F-value)
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X, y)
    
    # Get feature names
    feature_names = X.columns[selector.get_support()].tolist()
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    return X_scaled, y, scaler, feature_names