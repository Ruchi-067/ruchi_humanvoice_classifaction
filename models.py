# src/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

def train_models(X, y):
    # 1. Classification Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    clf_score = accuracy_score(y, y_pred)
    
    # 2. Clustering Model (Assume 2 clusters for Male/Female grouping)
    clusterer = KMeans(n_clusters=2, random_state=42)
    cluster_labels = clusterer.fit_predict(X)
    sil_score = silhouette_score(X, cluster_labels)
    
    return clf, clusterer, clf_score, sil_score