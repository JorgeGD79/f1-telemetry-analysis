import fastf1
fastf1.Cache.enable_cache('data/')  # Asegúrate de que esta ruta exista

session = fastf1.get_session(2023, 'Monza', 'Q')  # Año, circuito, tipo sesión
session.load()

laps = session.laps

# Vuelta más rápida de cada piloto
ham = laps.pick_driver('HAM').pick_fastest()
ver = laps.pick_driver('VER').pick_fastest()

# Obtener telemetría con distancia
ham_tel = ham.get_telemetry().add_distance()
ver_tel = ver.get_telemetry().add_distance()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(ham_tel['Distance'], ham_tel['Speed'], label='Hamilton')
plt.plot(ver_tel['Distance'], ver_tel['Speed'], label='Verstappen')

plt.title('Velocidad - Vuelta Rápida Monza Q 2023')
plt.xlabel('Distancia (m)')
plt.ylabel('Velocidad (km/h)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
