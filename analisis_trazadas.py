import fastf1
import matplotlib.pyplot as plt

# Activar caché
fastf1.Cache.enable_cache('data/')

# Cargar sesión de clasificación (o carrera)
session = fastf1.get_session(2023, 'Monza', 'Q')
session.load()

# Elegir pilotos
lap_ham = session.laps.pick_driver('HAM').pick_fastest()
lap_ver = session.laps.pick_driver('VER').pick_fastest()

# Obtener telemetría con coordenadas
tel_ham = lap_ham.get_telemetry()
tel_ver = lap_ver.get_telemetry()


plt.figure(figsize=(10, 8))

plt.plot(tel_ham['X'], tel_ham['Y'], label='Hamilton')
plt.plot(tel_ver['X'], tel_ver['Y'], label='Verstappen')

plt.xlabel('X Position (track coordinates)')
plt.ylabel('Y Position (track coordinates)')
plt.title('Trazadas - Vuelta Rápida Monza Q 2023')


plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np

plt.figure(figsize=(10, 8))
plt.scatter(tel_ham['X'], tel_ham['Y'], c=tel_ham['Speed'], cmap='viridis', s=1, label='Hamilton')
plt.scatter(tel_ver['X'], tel_ver['Y'], c=tel_ver['Speed'], cmap='plasma', s=1, label='Verstappen')
plt.colorbar(label='Speed (km/h)')
plt.title('Trazadas coloreadas por velocidad')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

