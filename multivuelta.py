import fastf1
import matplotlib.pyplot as plt
import pandas as pd

session = fastf1.get_session(2023, 'Monza', 'R')
session.load()

laps_ham = session.laps.pick_driver('HAM').pick_quicklaps()  # Filtra vueltas limpias

laps_ham['LapNumber'] = laps_ham['LapNumber'].astype(int)

plt.figure(figsize=(10, 5))
plt.plot(laps_ham['LapNumber'], laps_ham['LapTime'].dt.total_seconds(), marker='o')
plt.title('Evolución del Ritmo - Hamilton (Carrera Monza 2023)')
plt.xlabel('Vuelta')
plt.ylabel('Tiempo de Vuelta (segundos)')
plt.grid(True)
plt.show()

# Seleccionar tres vueltas a comparar
v1 = laps_ham.loc[laps_ham['LapNumber'] == 10].iloc[0]
v2 = laps_ham.loc[laps_ham['LapNumber'] == 15].iloc[0]
v3 = laps_ham.loc[laps_ham['LapNumber'] == 20].iloc[0]

# Obtener telemetrías
tel1 = v1.get_telemetry().add_distance()
tel2 = v2.get_telemetry().add_distance()
tel3 = v3.get_telemetry().add_distance()

# Graficar velocidades
plt.figure(figsize=(12, 6))
plt.plot(tel1['Distance'], tel1['Speed'], label='Vuelta 10')
plt.plot(tel2['Distance'], tel2['Speed'], label='Vuelta 15')
plt.plot(tel3['Distance'], tel3['Speed'], label='Vuelta 20')

plt.xlabel('Distancia (m)')
plt.ylabel('Velocidad (km/h)')
plt.title('Comparación de velocidad entre vueltas - HAM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

telemetrias = []

for lap in [v1, v2, v3]:
    tel = lap.get_telemetry().add_distance()
    tel['LapNumber'] = lap['LapNumber']
    tel['Driver'] = 'HAM'
    telemetrias.append(tel)

df_multi = pd.concat(telemetrias)
df_multi.to_csv('data/hamilton_monza_race_multilap.csv', index=False)
