import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, silhouette_score
import joblib  # For saving models

# Step 1: Load dataset
df = pd.read_csv('voice_data.csv')
X = df.drop('label', axis=1)
y = df['label']

# Step 2: Preprocessing
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 3: Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Feature selection
selector = SelectKBest(score_func=f_classif, k=20)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)

# Step 5: Clustering (K-Means)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train_sel)
silhouette_kmeans = silhouette_score(X_train_sel, kmeans_labels)
print("K-Means Silhouette Score:", silhouette_kmeans)

# Step 6: Classification - Random Forest
rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=3)
rf_grid.fit(X_train_sel, y_train)
rf_best = rf_grid.best_estimator_

# Step 7: Evaluate RF on test set
y_pred_rf = rf_best.predict(X_test_sel)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Step 8: SVM (optional, for comparison)
svm = SVC(random_state=42)
svm_params = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm, svm_params, cv=3)
svm_grid.fit(X_train_sel, y_train)
svm_best = svm_grid.best_estimator_

# Step 9: Save models and preprocessors
joblib.dump(rf_best, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
print("All models and preprocessors saved successfully!")