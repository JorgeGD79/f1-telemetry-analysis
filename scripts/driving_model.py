
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# Configuración
INPUT_FOLDERS = ["data/season_2023_R", "data/season_2024_R"]  # Puedes añadir más rutas
OUTPUT_X = "X_driving_model.npy"
OUTPUT_Y = "y_driving_model.npy"
FEATURES = ['Speed', 'Throttle', 'Brake', 'RPM']
SAMPLES_PER_LAP = 200

X = []
y = []
scaler = MinMaxScaler()

for folder in INPUT_FOLDERS:
    files = [f for f in os.listdir(folder) if f.endswith("ALL.csv")]
    for file in tqdm(files, desc=f"Procesando {folder}"):
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df = df[df['LapNumber'].notna()]
            df['LapNumber'] = df['LapNumber'].astype(int)

            grouped = df.groupby(['Driver', 'LapNumber'])

            for (driver, lap), group in grouped:
                group = group.sort_values('Distance')
                if len(group) < 20:
                    continue
                try:
                    interpolated = {}
                    distance_uniform = np.linspace(group['Distance'].min(), group['Distance'].max(), SAMPLES_PER_LAP)
                    for feature in FEATURES:
                        interpolated[feature] = np.interp(distance_uniform, group['Distance'], group[feature])
                    lap_array = np.stack([interpolated[f] for f in FEATURES], axis=1)
                    lap_scaled = scaler.fit_transform(lap_array)
                    X.append(lap_scaled.flatten())
                    y.append(driver)
                except Exception as e:
                    print(f"❌ Error en vuelta {lap} de {driver} ({file}): {e}")
                    continue
        except Exception as e:
            print(f"❌ Error cargando {file} en {folder}: {e}")
            continue

# Guardar resultados
X = np.array(X)
y = np.array(y)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print(f"✅ Guardado: {OUTPUT_X} con shape {X.shape}")
print(f"✅ Guardado: {OUTPUT_Y} con {len(y)} etiquetas")
