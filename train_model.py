import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

print("--- FASE 1: PEMBUATAN MODEL (TRAINING PHASE) ---")

# 1. Memuat dataset mentah
df = pd.read_csv('online_shoppers_intention.csv')

# 2. Pra-pemrosesan: Seleksi Fitur
kolom_fitur = ['Administrative', 'Informational', 'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues']
X = df[kolom_fitur]

# 3. Pra-pemrosesan: Scaling (Normalisasi)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Pelatihan Model Clustering (K-Means)
model_kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
model_kmeans.fit(X_scaled)

# 5. Menyimpan hasil penting ke file (Model & Preprocessing) 
joblib.dump(model_kmeans, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Proses Selesai! Berkas model.pkl dan scaler.pkl siap digunakan.")