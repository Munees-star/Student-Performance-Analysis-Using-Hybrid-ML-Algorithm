import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    silhouette_score,
    silhouette_samples,
    roc_curve,
    roc_auc_score
)


# ------------------ LOAD DATA ------------------
data = pd.read_csv("StudentPerformanceFactors.csv")


# ------------------ ENCODE CATEGORICAL FEATURES ------------------
le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])


# ------------------ FEATURES & TARGET ------------------
X = data.drop("Exam_Score", axis=1)
y = data["Exam_Score"]


# ============================================================
# DBSCAN CLUSTERING (DENSITY DISCOVERY)
# ============================================================


cluster_features = [
    'Hours_Studied',
    'Previous_Scores',
    'Sleep_Hours',
    'Attendance',
    'Motivation_Level'
]


scaler = StandardScaler()
X_cluster = scaler.fit_transform(X[cluster_features])


# ------------------ DBSCAN AUTO-TUNING ------------------
eps_values = [0.5, 1.0, 1.5]
min_samples_values = [5, 10]


best_sil = -1
best_labels = None
best_eps = None
best_min = None


for eps in eps_values:
    for ms in min_samples_values:
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_cluster)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)


        if n_clusters > 1:
            sil = silhouette_score(X_cluster, labels)
            if sil > best_sil:
                best_sil = sil
                best_labels = labels
                best_eps = eps
                best_min = ms


print(f"Best DBSCAN → eps={best_eps}, min_samples={best_min}, silhouette={best_sil:.3f}")


ADD CLUSTER FEATURE 
X['DBSCAN_Cluster'] = pd.Categorical(best_labels).codes




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)


rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
print("Hybrid RF MSE:", rf_mse)


# ------------------ RF CV ------------------
cv_scores = cross_val_score(
    rf, X, y, cv=5, scoring="neg_mean_squared_error"
)
print(f"Hybrid RF CV MSE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ------------------ RF FEATURE IMPORTANCE ------------------
feat_imp = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })
    .sort_values("Importance", ascending=False)
    .head(6)
)
print("\nTop Feature Importances:")
print(feat_imp)


# ------------------ RF PLOT ------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Hybrid RF: Actual vs Predicted")
plt.savefig("hybrid_rf_results.png")
plt.close()




X_no_cluster = X.drop("DBSCAN_Cluster", axis=1)


X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(
    X_no_cluster, y, test_size=0.2, random_state=42
)


rf_nc = RandomForestRegressor(n_estimators=150, random_state=42)
rf_nc.fit(X_train_nc, y_train_nc)


rf_nc_pred = rf_nc.predict(X_test_nc)
rf_nc_mse = mean_squared_error(y_test_nc, rf_nc_pred)


print("RF without Cluster MSE:", rf_nc_mse)
print("MSE Improvement:", rf_nc_mse - rf_mse)




y_class = (y >= 65).astype(int)


X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)


scaler_lr = StandardScaler()
X_train_c = scaler_lr.fit_transform(X_train_c)
X_test_c = scaler_lr.transform(X_test_c)


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_c, y_train_c)


lr_pred = lr.predict(X_test_c)
lr_prob = lr.predict_proba(X_test_c)[:,1]


print("Hybrid LR Accuracy:", accuracy_score(y_test_c, lr_pred))


fpr, tpr, _ = roc_curve(y_test_c, lr_prob)
auc = roc_auc_score(y_test_c, lr_prob)


plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Hybrid Logistic Regression ROC")
plt.legend()
plt.savefig("hybrid_lr_roc.png")
plt.close()




pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)


sil_vals = silhouette_samples(X_cluster, best_labels)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)


plt.scatter(X_pca[:,0], X_pca[:,1], c=best_labels, cmap="tab10", s=40)
plt.title("DBSCAN Clusters (PCA)")


plt.subplot(1,2,2)
plt.hist(sil_vals, bins=20)
plt.title("Silhouette Score Distribution")


plt.tight_layout()
plt.savefig("silhouette_analysis.png")
plt.close()


print("\nSaved files:")
print("hybrid_rf_results.png")
print("hybrid_lr_roc.png")
print("silhouette_analysis.png")
