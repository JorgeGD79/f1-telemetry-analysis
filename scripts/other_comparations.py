import fastf1
import matplotlib.pyplot as plt


fastf1.Cache.enable_cache('../data/')  # Asegúrate de que esta ruta exista

session = fastf1.get_session(2023, 'Monza', 'Q')  # Año, circuito, tipo sesión
session.load()

laps = session.laps

# Vuelta más rápida de cada piloto
ham = laps.pick_driver('HAM').pick_fastest()
ver = laps.pick_driver('VER').pick_fastest()

# Obtener telemetría con distancia
ham_tel = ham.get_telemetry().add_distance()
ver_tel = ver.get_telemetry().add_distance()


fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(ham_tel['Distance'], ham_tel['Throttle'], label='HAM')
axs[0].plot(ver_tel['Distance'], ver_tel['Throttle'], label='VER')
axs[0].set_ylabel('Throttle (%)')
axs[0].legend()

axs[1].plot(ham_tel['Distance'], ham_tel['Brake'], label='HAM')
axs[1].plot(ver_tel['Distance'], ver_tel['Brake'], label='VER')
axs[1].set_ylabel('Brake (0/1)')
axs[1].legend()

axs[2].plot(ham_tel['Distance'], ham_tel['DRS'], label='HAM')
axs[2].plot(ver_tel['Distance'], ver_tel['DRS'], label='VER')
axs[2].set_ylabel('DRS')
axs[2].set_xlabel('Distancia (m)')
axs[2].legend()

plt.suptitle('Comparación: Throttle, Brake, DRS - Monza Q 2023')
plt.grid(True)
plt.tight_layout()
plt.show()

