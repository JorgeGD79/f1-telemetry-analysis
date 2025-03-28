import fastf1
import matplotlib.pyplot as plt
import pandas as pd

session = fastf1.get_session(2023, 'Monza', 'R')
session.load()

laps_ham = session.laps.pick_driver('HAM')
laps_ver = session.laps.pick_driver('VER')


# Convertir LapTime a segundos
laps_ham['LapTimeSeconds'] = laps_ham['LapTime'].dt.total_seconds()
laps_ver['LapTimeSeconds'] = laps_ver['LapTime'].dt.total_seconds()

# Filtrar vueltas válidas
laps_ham_clean = laps_ham[laps_ham['LapTimeSeconds'].notna()]
laps_ver_clean = laps_ver[laps_ver['LapTimeSeconds'].notna()]

# Agrupar por stint y graficar
for stint, group in laps_ham_clean.groupby('Stint'):
    plt.plot(group['LapNumber'], group['LapTimeSeconds'], marker='o', label=f'Ham Stint {stint} - {group["Compound"].iloc[0]}')

plt.xlabel('Vuelta')
plt.ylabel('Tiempo de vuelta (seg)')
plt.title('Evolución del ritmo por Stint - Hamilton (Monza 2023)')


for stint, group in laps_ver_clean.groupby('Stint'):
    plt.plot(group['LapNumber'], group['LapTimeSeconds'], marker='x', label=f'Ver Stint {stint} - {group["Compound"].iloc[0]}')


plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for stint, group in laps_ham_clean.groupby('Stint'):
    compound = group["Compound"].iloc[0]
    x = group['LapNumber']
    y = group['LapTimeSeconds']
    coef = pd.Series(y).diff().mean()
    print(f"Stint {stint} ({compound}): Degradación media de ham ≈ {coef:.2f} s/vuelta")

for stint, group in laps_ver_clean.groupby('Stint'):
    compound = group["Compound"].iloc[0]
    x = group['LapNumber']
    y = group['LapTimeSeconds']
    coef = pd.Series(y).diff().mean()
    print(f"Stint {stint} ({compound}): Degradación media de verstappen ≈ {coef:.2f} s/vuelta")



stints_df = laps_ham_clean[['LapNumber', 'Stint', 'LapTimeSeconds', 'Compound', 'TyreLife']]
stints_df['Driver'] = 'HAM'
stints_df.to_csv('powerbi/hamilton_monza_stints.csv', index=False)

stints_df = laps_ver_clean[['LapNumber', 'Stint', 'LapTimeSeconds', 'Compound', 'TyreLife']]
stints_df['Driver'] = 'HAM'
stints_df.to_csv('powerbi/hamilton_monza_stints.csv', index=False)

