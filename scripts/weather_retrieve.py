
import os
import pandas as pd
import fastf1
from fastf1 import get_session
from fastf1.events import get_event_schedule

fastf1.Cache.enable_cache("cache")

# Configuración
SEASON = 2023
SESSION_TYPE = "R"  # 'R', 'Q', 'FP1', etc.
OUTPUT_FOLDER = f"data/season_{SEASON}_{SESSION_TYPE}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Obtener calendario de eventos
schedule = get_event_schedule(SEASON, include_testing=False)

# Recorrer todos los eventos del año
for _, row in schedule.iterrows():
    gp_name = row['EventName'].replace(" ", "_")
    try:
        print(f"🌦️ Descargando clima para {gp_name} - {SESSION_TYPE}")
        session = get_session(SEASON, gp_name, SESSION_TYPE)
        session.load()

        weather = session.weather_data
        summary = weather.mean(numeric_only=True).to_frame(name='Mean').T

        weather.to_csv(f"{OUTPUT_FOLDER}/{gp_name}_WEATHER.csv", index=False)
        summary.to_csv(f"{OUTPUT_FOLDER}/{gp_name}_WEATHER_SUMMARY.csv", index=False)
        print(f"✅ Guardado: {gp_name}_WEATHER.csv")

    except Exception as e:
        print(f"❌ Error en {gp_name}: {e}")
