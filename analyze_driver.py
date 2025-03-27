
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar datos
X = np.load("X_driving_model.npy")
y = np.load("y_driving_model.npy")
FEATURES = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS']
SAMPLES_PER_LAP = 100
N_FEATURES = len(FEATURES)

# Número de clusters
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Extraer perfiles promedio por cluster
cluster_profiles = np.zeros((n_clusters, N_FEATURES))

for i in range(n_clusters):
    cluster_data = X[labels == i]
    reshaped = cluster_data.reshape(cluster_data.shape[0], SAMPLES_PER_LAP, N_FEATURES)
    cluster_profiles[i] = reshaped.mean(axis=(0, 1))  # promedio total por feature

# Mostrar tabla
df_profile = pd.DataFrame(cluster_profiles, columns=FEATURES)
df_profile.index.name = "Cluster"
print("📊 Promedio por variable en cada estilo de conducción (cluster):\n")
print(df_profile.round(2))

# Descripción simple (puedes ajustar según los valores observados)
descriptions = []
for i, row in df_profile.iterrows():
    desc = []
    if row['Throttle'] > 0.7 and row['Brake'] < 0.2:
        desc.append("Agresivo")
    if row['Brake'] > 0.5:
        desc.append("Frenador")
    if row['Speed'] > 250:
        desc.append("Alta Velocidad")
    if row['Throttle'] < 0.5:
        desc.append("Conservador")
    if row['nGear'] > 6:
        desc.append("Usa marchas altas")
    descriptions.append(", ".join(desc) if desc else "Equilibrado")

df_profile['Descripción'] = descriptions

# Guardar resultado
df_profile.to_csv("driving_style_cluster_profiles.csv", index=True)
print("\n✅ Guardado: driving_style_cluster_profiles.csv")
