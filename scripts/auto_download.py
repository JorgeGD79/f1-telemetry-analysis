
import fastf1
import pandas as pd
import os

# Enable cache
fastf1.Cache.enable_cache('../data/')

# Parameters
season = 2025
session_type = 'FP2'  # 'R' = Race, 'Q' = Quali, etc.
base_dir = f"../data/season_{season}_{session_type}"
os.makedirs(base_dir, exist_ok=True)

# Get season schedule
schedule = fastf1.get_event_schedule(season)
schedule = schedule[schedule['Session1'].notna()]

# Loop through each GP
for _, row in schedule.iterrows():
    gp_name = row['EventName'].replace(" ", "_")
    gp_round = row['RoundNumber']

    try:
        print(f"\n📥 Downloading {gp_name} - {session_type}")
        session = fastf1.get_session(season, gp_round, session_type)
        session.load()

        all_laps = []

        for drv in session.laps['Driver'].unique():
            laps = session.laps.pick_driver(drv).pick_quicklaps()

            if laps.empty:
                continue

            for _, lap in laps.iterrows():
                try:
                    tel = lap.get_telemetry().add_distance()
                    tel['LapNumber'] = lap['LapNumber']
                    tel['Driver'] = drv
                    tel['Team'] = lap['Team']
                    tel['Compound'] = lap['Compound']
                    tel['Stint'] = lap['Stint']
                    tel['Track'] = gp_name
                    tel['Session'] = session_type
                    all_laps.append(tel)
                except Exception as e:
                    print(f"   ⚠️ Error loading telemetry for {drv} Lap {lap['LapNumber']}: {e}")

        # Combine and save per GP
        if all_laps:
            df = pd.concat(all_laps)
            combined_path = f"{base_dir}/{gp_round:02d}_{gp_name}_ALL.csv"
            df.to_csv(combined_path, index=False)
            print(f"✔ Combined telemetry saved: {combined_path}")

    except Exception as e:
        print(f"❌ Error loading session {gp_name}: {e}")
