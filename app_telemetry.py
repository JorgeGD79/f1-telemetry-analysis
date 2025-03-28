import streamlit as st
import pandas as pd
import plotly.express as px
import os

import streamlit as st

from google.cloud import storage
from io import StringIO

import numpy as np
from io import BytesIO

import os

import json
from google.oauth2 import service_account
from google.cloud import storage

# Cargar el JSON desde los secrets
gcs_credentials = json.loads(st.secrets["GCS_CREDENTIALS_JSON"])

# Reemplazar \\n por \n en la clave privada
gcs_credentials["private_key"] = gcs_credentials["private_key"].replace("\\n", "\n")

# Crear las credenciales
credentials = service_account.Credentials.from_service_account_info(gcs_credentials)
client = storage.Client(credentials=credentials)

bucket_name = st.secrets["GCS_BUCKET"]
bucket = client.bucket(bucket_name)

st.set_page_config(layout="wide")
st.title("🏎️ F1 Telemetry Dashboard")

# Secciones del menú

# Menú lateral dividido
section_options = [
    "🏠 Welcome",
    "-------------------------------",
    "🏁 Fast Lap Comparison",
    "⚙️ Throttle & Brake",
    "🕹️ NGear",
    "📉 Consistency",
    "⏱️ Real Delta",
    "🏁 Race Mode",
    "🧭 Racing Line",
    "-------------------------------",
    "📏 Lap Time Comparison",
    "📍 Pilot Positioning",
    "🔥 Aggressiveness",
    "📉 Driver Consistency Analysis",
    "🌦️ Weather Summary",
    "------------------------------",
    "🧠 Driving Styles",
    "🏁 Performance Summary",
    "📈 Season Progress"
]

section = st.sidebar.selectbox("📂 Select Section", section_options)


def load_npy_from_gcs(bucket_name, blob_name):
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        return np.load(BytesIO(content), allow_pickle=True)
    except Exception as e:
        st.error(f"❌ Error loading {blob_name} from GCS: {e}")
        return None


@st.cache_data
def load_data_from_gcs(bucket_name, gcs_path):
    try:
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        content = blob.download_as_text()
        return pd.read_csv(StringIO(content))
    except Exception as e:
        st.error(f"❌ Error loading data from GCS: {e}")
        return pd.DataFrame()


def list_files_in_gcs(prefix):
    blobs = client.list_blobs(bucket, prefix=prefix)
    return [blob.name.replace(f"{prefix}", "") for blob in blobs if blob.name.endswith("ALL.csv")]


# Sidebar
st.sidebar.title("📊 Configuration")
season = st.sidebar.selectbox("Season", ["2023", "2024", "2025"])
session_type = st.sidebar.selectbox("Session", ["R", "Q", "S"])
gcs_prefix = f"data/season_{season}_{session_type}/"
all_files = list_files_in_gcs(gcs_prefix)
gp_file = st.sidebar.selectbox("Grand Prix", all_files, index=0)
csv_path = f"data/season_{season}_{session_type}/{gp_file}"

GCS_BUCKET = "f1-telemetry-data"  # Cambia esto por el nombre real del bucket
df = load_data_from_gcs(GCS_BUCKET, csv_path)

if not df.empty:
    drivers = df['Driver'].unique()
    driver1 = st.sidebar.selectbox("Driver 1", drivers, index=0)
    driver2 = st.sidebar.selectbox("Driver 2", drivers, index=1 if len(drivers) > 1 else 0)

    lap_options = df[df['Driver'] == driver1]['LapNumber'].unique()
    lap_options_2 = df[df['Driver'] == driver2]['LapNumber'].unique()

    lap1 = st.sidebar.selectbox("Lap Driver 1", lap_options, index=0)
    lap2 = st.sidebar.selectbox("Lap Driver 2", lap_options_2, index=0)

    if section == "🏠 Welcome":
        st.markdown("# 👋 Welcome to the F1 Telemetry Dashboard")

        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1280px-F1.svg.png", width=200)

        st.markdown("""
        ---
        🎯 **Explore, Compare and Analyze**  
        This dashboard lets you dive into **Formula 1 telemetry** like never before.

        #### 🔧 What you can do:
        - 🏁 **Compare** speed, throttle, braking and gears between drivers  
        - 🧠 **Classify** and visualize **driving styles** using clustering and PCA  
        - 🔍 **Analyze** lap-by-lap performance, delta times and consistency  
        - 🌦️ **See how weather** affects performance  
        - 📈 **Track progression** through the season, points and standings  
        - 🔥 **Evaluate aggressiveness** and driving risk  

        #### 🧭 Navigation:
        - Use the **sidebar** to select year, session, GP and drivers  
        - Choose a section from the dropdown: _Comparatives_ or _Insights_  

        ---
        👨‍💻 Built with **Streamlit** & powered by **FastF1**, **Pandas**, **Plotly** and **Scikit-Learn**
        """)

        st.info("Use the left sidebar to begin exploring the data.")

    if section == "🏁 Fast Lap Comparison":
        st.subheader("Speed Comparison")

        lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)]
        lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)]

        fig = px.line()
        fig.add_scatter(x=lap1_data['Distance'], y=lap1_data['Speed'], mode='lines', name=f'{driver1} - Lap {lap1}')
        fig.add_scatter(x=lap2_data['Distance'], y=lap2_data['Speed'], mode='lines', name=f'{driver2} - Lap {lap2}')
        fig.update_layout(xaxis_title="Distance (m)", yaxis_title="Speed (km/h)")
        st.plotly_chart(fig, use_container_width=True)

    if section == "🧭 Racing Line":
        st.subheader("Racing Line")

        lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)]

        fig_trace = px.scatter(
            lap1_data,
            x="X",
            y="Y",
            color="Speed",
            title=f"Racing Line - {driver1} - Lap {lap1}",
            color_continuous_scale="viridis",
            size_max=1
        )
        fig_trace.update_layout(
            xaxis_title="X",
            yaxis_title="Y",
            yaxis_scaleanchor="x",
            showlegend=False
        )
        st.plotly_chart(fig_trace, use_container_width=True)

    if section == "⚙️ Throttle & Brake":
        st.subheader("Throttle Break Compare")

        lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)]
        lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)]

        fig_throttle = px.line()
        fig_throttle.add_scatter(x=lap1_data['Distance'], y=lap1_data['Throttle'], mode='lines',
                                 name=f'{driver1} - Lap {lap1}')
        fig_throttle.add_scatter(x=lap2_data['Distance'], y=lap2_data['Throttle'], mode='lines',
                                 name=f'{driver2} - Lap {lap2}')
        fig_throttle.update_layout(xaxis_title="Distance (m)", yaxis_title="Throttle %")

        st.plotly_chart(fig_throttle, use_container_width=True)

        fig_brake = px.line()
        fig_brake.add_scatter(x=lap1_data['Distance'], y=lap1_data['Brake'], mode='lines',
                              name=f'{driver1} - Lap {lap1}')
        fig_brake.add_scatter(x=lap2_data['Distance'], y=lap2_data['Brake'], mode='lines',
                              name=f'{driver2} - Lap {lap2}')
        fig_brake.update_layout(xaxis_title="Distance (m)", yaxis_title="Brake %")

        st.plotly_chart(fig_brake, use_container_width=True)

    if section == "🕹️ NGear":
        st.subheader("Gear Compare")

        lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)]
        lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)]

        fig_gear = px.line()
        fig_gear.add_scatter(x=lap1_data['Distance'], y=lap1_data['nGear'], mode='lines',
                             name=f'{driver1} - Lap {lap1}')
        fig_gear.add_scatter(x=lap2_data['Distance'], y=lap2_data['nGear'], mode='lines',
                             name=f'{driver2} - Lap {lap2}')
        fig_gear.update_layout(xaxis_title="Distance (m)", yaxis_title="nGear")

        st.plotly_chart(fig_gear, use_container_width=True)

    if section == "🧠 Driving Styles":
        st.subheader("🧠 Driving Style Clustering")

        try:
            import numpy as np
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.cluster import KMeans
            import seaborn as sns
            import matplotlib.pyplot as plt

            # Cargar datos
            X = load_npy_from_gcs(GCS_BUCKET, "models/X_driving_model.npy")
            y = load_npy_from_gcs(GCS_BUCKET, "models/y_driving_model.npy")

            FEATURES = ['Speed', 'Throttle', 'Brake', 'RPM']

            # Clustering
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)

            df_clustered = pd.DataFrame(X)
            df_clustered['Driver'] = y
            df_clustered['Cluster'] = labels

            cluster_summary = df_clustered.groupby(['Driver', 'Cluster']).size().unstack(fill_value=0)
            cluster_percent = cluster_summary.div(cluster_summary.sum(axis=1), axis=0)

            st.markdown("### 🔍 Cluster Distribution per Driver")
            st.dataframe(cluster_percent.style.format("{:.2%}"))

            st.markdown("### 🎯 Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cluster_percent, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Driving Styles by Driver (Cluster %)")
            st.pyplot(fig)

            selected_driver = st.selectbox("👤 Select a Driver to analyze dominant style",
                                           sorted(df_clustered['Driver'].unique()))
            if selected_driver:
                driver_distribution = cluster_percent.loc[selected_driver]
                dominant_cluster = driver_distribution.idxmax()

            st.markdown("### 🧬 Cluster Profiles & Interpretation")

            # Detección automática de dimensiones
            n_samples = X.shape[0]
            vector_length = X.shape[1]
            for factor in range(1, 51):
                if vector_length % factor == 0:
                    possible_features = factor
                    possible_samples = vector_length // factor
                    if possible_features in [len(FEATURES), 4, 6]:
                        break
            else:
                raise ValueError("No se pudo deducir dimensiones")

            cluster_profiles = np.zeros((n_clusters, possible_features))
            for i in range(n_clusters):
                cluster_data = X[labels == i]
                reshaped = cluster_data.reshape(cluster_data.shape[0], possible_samples, possible_features)
                cluster_profiles[i] = reshaped.mean(axis=(0, 1))

            df_profiles = pd.DataFrame(cluster_profiles, columns=FEATURES[:possible_features])
            df_profiles.index.name = "Cluster"

            descriptions = []
            long_descriptions = []
            for i, row in df_profiles.iterrows():
                desc = []
                long_desc = []
                if 'Throttle' in row and row['Throttle'] > 0.75 and row.get('Brake', 0) < 0.2:
                    desc.append("Aggressive")
                    long_desc.append("High throttle usage and minimal braking")
                if 'Brake' in row and row['Brake'] > 0.5:
                    desc.append("Heavy Braking")
                    long_desc.append("Frequent or intense use of brakes")
                if 'Speed' in row and row['Speed'] > 250:
                    desc.append("High Speed")
                    long_desc.append("Consistently fast pace on straights and curves")
                if 'Throttle' in row and row['Throttle'] < 0.5:
                    desc.append("Conservative")
                    long_desc.append("Careful throttle management")
                if 'nGear' in row and row['nGear'] > 6.5:
                    desc.append("High Gear Usage")
                    long_desc.append("Stays in high gears longer")
                if 'RPM' in row and row['RPM'] > 11000:
                    desc.append("High Revving")
                    long_desc.append("Keeps engine at high RPMs")

                descriptions.append(", ".join(desc) if desc else "Balanced")
                long_descriptions.append(" / ".join(long_desc) if long_desc else "Adaptable across scenarios")

            df_profiles['Description'] = descriptions
            df_profiles['Details'] = long_descriptions

            st.dataframe(df_profiles.style.format("{:.2f}", subset=df_profiles.columns[:-2]))

            for idx, row in df_profiles.iterrows():
                st.markdown(f"**Cluster {idx}**: {row['Description']}")
                st.markdown(f"_Details_: {row['Details']}")

            if selected_driver:
                dominant_description = df_profiles.loc[dominant_cluster, 'Description']
                dominant_details = df_profiles.loc[dominant_cluster, 'Details']
                st.success(
                    f"**{selected_driver}'s dominant driving style is Cluster {dominant_cluster}: {dominant_description}**")
                st.caption(f"_Details: {dominant_details}_")

            st.markdown("### 📊 Radar Chart of Driving Styles")
            radar_fig = go.Figure()
            for i in range(n_clusters):
                radar_fig.add_trace(go.Scatterpolar(
                    r=df_profiles.loc[i, df_profiles.columns[:-2]].values,
                    theta=df_profiles.columns[:-2],
                    fill='toself',
                    name=f"Cluster {i}"
                ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading driving style clustering data: {e}")

    if section == "📍 Pilot Positioning":
        st.subheader("📍 Pilot Positioning by Driving Style")

        try:
            import numpy as np
            import pandas as pd
            from sklearn.decomposition import PCA
            import plotly.express as px

            # Cargar datos
            X = load_npy_from_gcs(GCS_BUCKET, "models/X_driving_model.npy")
            y = load_npy_from_gcs(GCS_BUCKET, "models/y_driving_model.npy")

            # Calcular promedio por piloto
            pilot_data = pd.DataFrame(X)
            pilot_data['Driver'] = y
            pilot_means = pilot_data.groupby('Driver').mean()

            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(pilot_means)

            df_pca = pd.DataFrame(X_pca, columns=['Style Axis 1', 'Style Axis 2'])
            df_pca['Driver'] = pilot_means.index

            fig_pca = px.scatter(
                df_pca,
                x='Style Axis 1',
                y='Style Axis 2',
                text='Driver',
                color='Driver',
                title="Driver Positioning by Driving Style (PCA Projection)"
            )
            fig_pca.update_traces(textposition='top center')
            fig_pca.update_layout(showlegend=False)

            st.plotly_chart(fig_pca, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating driver style map: {e}")

    if section == "⏱️ Real Delta":
        st.subheader("⏱️ Real Delta Time Between Drivers")

        try:
            import pandas as pd
            import numpy as np
            import plotly.express as px

            # Filtrar y ordenar los datos por distancia
            lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)].sort_values("Distance")
            lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)].sort_values("Distance")

            # Convertir a segundos desde la primera muestra
            lap1_data['Seconds'] = pd.to_timedelta(lap1_data['Time']).dt.total_seconds()
            lap2_data['Seconds'] = pd.to_timedelta(lap2_data['Time']).dt.total_seconds()

            # Interpolar a una base común de distancia
            common_distance = pd.Series(np.linspace(
                max(lap1_data['Distance'].min(), lap2_data['Distance'].min()),
                min(lap1_data['Distance'].max(), lap2_data['Distance'].max()),
                500
            ))

            lap1_interp = pd.DataFrame({
                "Distance": common_distance,
                "Time1": np.interp(common_distance, lap1_data["Distance"], lap1_data["Seconds"])
            })
            lap2_interp = pd.DataFrame({
                "Distance": common_distance,
                "Time2": np.interp(common_distance, lap2_data["Distance"], lap2_data["Seconds"])
            })

            merged = pd.merge(lap1_interp, lap2_interp, on="Distance")
            merged["DeltaTime"] = merged["Time2"] - merged["Time1"]

            fig_delta = px.line(merged, x="Distance", y="DeltaTime", title="Δ Delta Time Between Drivers (Actual Time)")
            fig_delta.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Time Difference (s)",
                showlegend=False
            )
            st.plotly_chart(fig_delta, use_container_width=True)

        except Exception as e:
            st.error(f"Error calculating real delta time: {e}")

    if section == "🏁 Race Mode":
        st.subheader("🏁 Race Mode: Animated Lap Playback (Fixed v2)")

        try:
            import pandas as pd
            import numpy as np
            import plotly.graph_objects as go

            # Obtener datos de las vueltas
            lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)].sort_values("Distance")
            lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)].sort_values("Distance")

            # Interpolar a base común
            num_frames = 100
            common_distance = np.linspace(
                max(lap1_data['Distance'].min(), lap2_data['Distance'].min()),
                min(lap1_data['Distance'].max(), lap2_data['Distance'].max()),
                num_frames
            )

            lap1_interp = pd.DataFrame({
                "X": np.interp(common_distance, lap1_data["Distance"], lap1_data["X"]),
                "Y": np.interp(common_distance, lap1_data["Distance"], lap1_data["Y"])
            })

            lap2_interp = pd.DataFrame({
                "X": np.interp(common_distance, lap2_data["Distance"], lap2_data["X"]),
                "Y": np.interp(common_distance, lap2_data["Distance"], lap2_data["Y"])
            })

            # Rango fijo de ejes
            x_range = [
                min(lap1_interp['X'].min(), lap2_interp['X'].min()) - 10,
                max(lap1_interp['X'].max(), lap2_interp['X'].max()) + 10
            ]
            y_range = [
                min(lap1_interp['Y'].min(), lap2_interp['Y'].min()) - 10,
                max(lap1_interp['Y'].max(), lap2_interp['Y'].max()) + 10
            ]

            # Crear figura
            fig = go.Figure()

            # Trayectorias
            trace1 = go.Scatter(x=lap1_interp["X"], y=lap1_interp["Y"],
                                mode='lines', name=f'{driver1} Path',
                                line=dict(color='blue'))
            trace2 = go.Scatter(x=lap2_interp["X"], y=lap2_interp["Y"],
                                mode='lines', name=f'{driver2} Path',
                                line=dict(color='red'))

            # Marcadores iniciales
            marker1 = go.Scatter(x=[lap1_interp["X"][0]], y=[lap1_interp["Y"][0]],
                                 mode='markers+text', name=driver1,
                                 marker=dict(color='blue', size=12),
                                 text=[driver1], textposition="top center")
            marker2 = go.Scatter(x=[lap2_interp["X"][0]], y=[lap2_interp["Y"][0]],
                                 mode='markers+text', name=driver2,
                                 marker=dict(color='red', size=12),
                                 text=[driver2], textposition="top center")

            fig.add_traces([trace1, trace2, marker1, marker2])

            # Frames con los 4 trazos actualizados en cada paso
            frames = []
            for i in range(num_frames):
                frames.append(go.Frame(
                    data=[
                        trace1,
                        trace2,
                        go.Scatter(x=[lap1_interp["X"][i]], y=[lap1_interp["Y"][i]],
                                   mode='markers+text', marker=dict(color='blue', size=12),
                                   text=[driver1], textposition="top center"),
                        go.Scatter(x=[lap2_interp["X"][i]], y=[lap2_interp["Y"][i]],
                                   mode='markers+text', marker=dict(color='red', size=12),
                                   text=[driver2], textposition="top center")
                    ],
                    name=str(i)
                ))

            # Configuración final
            fig.frames = frames
            fig.update_layout(
                title="Lap Animation: Position on Track",
                xaxis_title="X", yaxis_title="Y",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range, scaleanchor="x"),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="▶ Play", method="animate", args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True
                        }]),
                        dict(label="⏸ Pause", method="animate", args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }])
                    ]
                )]
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in animated race mode: {e}")

    if section == "📏 Lap Time Comparison":
        st.subheader("📏 Lap Time Comparison (Calculated)")

        try:
            import pandas as pd
            import numpy as np
            import plotly.express as px

            # Asegurar que Time está en formato datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_timedelta(df['Time'])

            # Calcular tiempo de vuelta por Driver y LapNumber
            lap_times = df.groupby(['Driver', 'LapNumber'])['Time'].agg(['min', 'max']).reset_index()
            lap_times['LapTimeSeconds'] = (lap_times['max'] - lap_times['min']).dt.total_seconds()

            # Interfaz
            drivers_available = sorted(lap_times['Driver'].unique())
            selected_drivers = st.multiselect("Select Drivers to Compare", drivers_available,
                                              default=[driver1, driver2])

            filtered = lap_times[lap_times['Driver'].isin(selected_drivers)]

            fig = px.line(filtered, x="LapNumber", y="LapTimeSeconds", color="Driver",
                          markers=True, title="Lap Time per Driver (calculated)")
            fig.update_layout(xaxis_title="Lap Number", yaxis_title="Lap Time (s)")
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar resumen
            st.markdown("### 🧾 Lap Time Summary")
            stats = filtered.groupby("Driver")["LapTimeSeconds"].agg(["mean", "min", "max", "std"]).round(2)
            st.dataframe(stats.rename(columns={
                "mean": "Avg",
                "min": "Best",
                "max": "Worst",
                "std": "Std Dev"
            }))

        except Exception as e:
            st.error(f"Error calculating lap times: {e}")

    if section == "📉 Consistency":
        st.subheader("Gear Compare")

        lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)]
        lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)]

        fig_rpm = px.line()
        fig_rpm.add_scatter(x=lap1_data['Distance'], y=lap1_data['RPM'], mode='lines',
                            name=f'{driver1} - Lap {lap1}')
        fig_rpm.add_scatter(x=lap2_data['Distance'], y=lap2_data['RPM'], mode='lines',
                            name=f'{driver2} - Lap {lap2}')
        fig_rpm.update_layout(xaxis_title="Distance (m)", yaxis_title="RPM")

        st.plotly_chart(fig_rpm, use_container_width=True)

    if section == "🌦️ Weather Summary":
        st.subheader("🌦️ Weather Summary for Selected Grand Prix")

        try:
            import pandas as pd
            import os
            import plotly.express as px

            # Extraer nombre base del GP quitando prefijo numerado
            gp_name = "_".join(gp_file.replace("_ALL.csv", "").split("_")[1:])

            # Ruta a archivos de clima
            weather_file = f"data/season_{season}_{session_type}/{gp_name}_WEATHER.csv"
            summary_file = f"data/season_{season}_{session_type}/{gp_name}_WEATHER_SUMMARY.csv"

            summary = load_data_from_gcs(GCS_BUCKET, summary_file)

            if not summary.empty:
                st.markdown("### 📋 Weather Summary")
                st.dataframe(summary.style.format("{:.2f}"))

                radar_vars = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
                if all(v in summary.columns for v in radar_vars):
                    radar_data = summary[radar_vars].iloc[0]
                    # Normalización simple (0 a 1 dentro del GP)
                    radar_data_scaled = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=radar_data_scaled.values,
                        theta=radar_data_scaled.index,
                        fill='toself',
                        name='Weather (scaled)'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        title="Scaled Weather Radar Overview"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            weather_df = load_data_from_gcs(GCS_BUCKET, weather_file)
            if not weather_df.empty:
                st.markdown("### 📈 Weather Evolution (per minute)")
                selected_metric = st.selectbox("Select Weather Variable", weather_df.columns.drop('Time'), index=0)
                if 'Time' in weather_df.columns and selected_metric:
                    fig_line = px.line(weather_df, x='Time', y=selected_metric, title=f"{selected_metric} over Time")
                    st.plotly_chart(fig_line, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading weather data: {e}")

    if section == "📉 Driver Consistency Analysis":
        st.subheader("📉 Driver Consistency Analysis")

        try:
            import pandas as pd
            import numpy as np
            import plotly.express as px
            import plotly.graph_objects as go

            # Calcular tiempos por vuelta como antes
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_timedelta(df['Time'])

            lap_times = df.groupby(['Driver', 'LapNumber'])['Time'].agg(['min', 'max']).reset_index()
            lap_times['LapTimeSeconds'] = (lap_times['max'] - lap_times['min']).dt.total_seconds()

            # Boxplot de tiempos por vuelta
            st.markdown("### ⏱️ Lap Time Boxplot")
            fig_box = px.box(lap_times, x='Driver', y='LapTimeSeconds', points="all",
                             title="Lap Time Spread per Driver")
            fig_box.update_layout(xaxis_title="Driver", yaxis_title="Lap Time (s)")
            st.plotly_chart(fig_box, use_container_width=True)

            # Variables técnicas por vuelta
            metrics = ['Speed', 'Throttle', 'Brake', 'RPM']
            metric_std = {m: [] for m in metrics}
            driver_labels = []

            for driver in df['Driver'].unique():
                driver_df = df[df['Driver'] == driver]
                lap_group = driver_df.groupby('LapNumber')
                driver_metrics = {m: [] for m in metrics}

                for _, lap in lap_group:
                    for m in metrics:
                        if lap[m].notna().sum() > 10:
                            driver_metrics[m].append(lap[m].std())

                if all(len(driver_metrics[m]) > 1 for m in metrics):
                    for m in metrics:
                        metric_std[m].append(np.mean(driver_metrics[m]))
                    driver_labels.append(driver)

            # Crear radar chart con métricas de consistencia (desviación más baja = más consistente)
            st.markdown("### 🎯 Consistency Radar per Driver (Lower = More Consistent)")

            radar_fig = go.Figure()
            for idx, driver in enumerate(driver_labels):
                values = [metric_std[m][idx] for m in metrics]
                # Normalizar por métrica
                normalized = [(v - min(metric_std[m])) / (max(metric_std[m]) - min(metric_std[m]) + 1e-6) for v, m in
                              zip(values, metrics)]
                radar_fig.add_trace(go.Scatterpolar(
                    r=normalized,
                    theta=metrics,
                    fill='toself',
                    name=driver
                ))

            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating consistency analysis: {e}")

    if section == "🏁 Performance Summary":
        st.subheader("🏁 Performance Summary")

        try:
            import pandas as pd
            import numpy as np

            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_timedelta(df['Time'])

            lap_times = df.groupby(['Driver', 'LapNumber'])['Time'].agg(['min', 'max']).reset_index()
            lap_times['LapTimeSeconds'] = (lap_times['max'] - lap_times['min']).dt.total_seconds()

            summary = lap_times.groupby("Driver")["LapTimeSeconds"].agg(["mean", "min", "max", "std"]).round(2)
            summary = summary.rename(columns={"mean": "Avg", "min": "Best", "max": "Worst", "std": "Std Dev"})

            st.markdown("### 📋 Lap Time Summary Table")
            st.dataframe(summary)

            most_consistent = summary["Std Dev"].idxmin()
            least_consistent = summary["Std Dev"].idxmax()
            fastest_avg = summary["Avg"].idxmin()
            fastest_lap = summary["Best"].idxmin()

            st.markdown("### 🏆 Key Insights")
            st.success(f"🧠 Most consistent driver: **{most_consistent}** (lowest std dev)")
            st.warning(f"⚠️ Least consistent driver: **{least_consistent}** (highest std dev)")
            st.info(f"🚀 Fastest driver on average: **{fastest_avg}**")
            st.info(f"⏱️ Best single lap: **{fastest_lap}**")

        except Exception as e:
            st.error(f"Error in performance summary: {e}")

    if section == "🔥 Aggressiveness":
        st.subheader("🔥 Driving Risk & Aggressiveness")

        try:
            import pandas as pd
            import numpy as np
            import plotly.express as px

            st.markdown("### 🧪 Aggression Metrics per Driver")

            metrics = []

            for driver in df['Driver'].unique():
                df_driver = df[df['Driver'] == driver]

                # Métricas de agresividad
                total_points = len(df_driver)
                if total_points == 0:
                    continue

                throttle_aggr = (df_driver['Throttle'] > 0.9).sum() / total_points
                brake_hard = (df_driver['Brake'] > 0.75).sum() / total_points
                rpm_aggr = (df_driver['RPM'] > 11000).sum() / total_points
                high_gear = (df_driver['nGear'] >= 7).sum() / total_points

                metrics.append({
                    "Driver": driver,
                    "% Full Throttle": throttle_aggr * 100,
                    "% Hard Braking": brake_hard * 100,
                    "% High RPM": rpm_aggr * 100,
                    "% High Gear": high_gear * 100,
                    "Aggression Score": (throttle_aggr + brake_hard + rpm_aggr + high_gear) / 4 * 100
                })

            df_aggr = pd.DataFrame(metrics)
            df_aggr = df_aggr.sort_values("Aggression Score", ascending=False)

            numeric_cols = ["% Full Throttle", "% Hard Braking", "% High RPM", "% High Gear", "Aggression Score"]
            st.dataframe(df_aggr.style.format({col: "{:.1f}" for col in numeric_cols}))

            st.markdown("### 📊 Aggression Score by Driver")
            fig = px.bar(df_aggr, x="Driver", y="Aggression Score", color="Aggression Score",
                         color_continuous_scale="Reds", title="Overall Aggression Score")
            fig.update_layout(xaxis_title="Driver", yaxis_title="Aggression Score (0-100)")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating aggression metrics: {e}")

    if section == "📈 Season Progress":
        st.subheader("📈 Season Progress: Positions and Points")

        try:
            import pandas as pd
            import plotly.express as px
            import os

            season_options = ["2023", "2024", "2025"]
            selected_season = st.selectbox("Select Season", season_options, key="season_points_select")

            csv_path = f"data/overall/season_{selected_season}_R_positions_points.csv"
            df_positions = load_data_from_gcs(GCS_BUCKET, csv_path)
            if df_positions.empty:
                st.warning(f"No data found for season {selected_season}")
            else:
                df_positions = pd.read_csv(csv_path)

                if 'GrandPrix' not in df_positions.columns or 'Driver' not in df_positions.columns:
                    st.warning("Missing required columns.")
                else:
                    df_positions['GrandPrix'] = pd.Categorical(df_positions['GrandPrix'],
                                                               categories=sorted(df_positions['GrandPrix'].unique(),
                                                                                 key=lambda x: x.lower()),
                                                               ordered=True)

                    st.markdown("### 🔢 Position Trend")
                    fig_pos = px.line(df_positions, x="GrandPrix", y="Position", color="Driver",
                                      markers=True, title="Driver Position per Grand Prix")
                    fig_pos.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Grand Prix",
                                          yaxis_title="Position")
                    st.plotly_chart(fig_pos, use_container_width=True)

                    st.markdown("### 🏆 Championship Points Evolution")
                    df_positions['Round'] = df_positions.groupby('Driver').cumcount()
                    df_positions['CumulativePoints'] = df_positions.groupby('Driver')['Points'].cumsum()

                    fig_points = px.line(df_positions, x="GrandPrix", y="CumulativePoints", color="Driver",
                                         markers=True, title="Cumulative Points by Grand Prix")
                    fig_points.update_layout(xaxis_title="Grand Prix", yaxis_title="Points")
                    st.plotly_chart(fig_points, use_container_width=True)

                    st.markdown("### 📋 Final Championship Standings")
                    final_table = df_positions.groupby("Driver")["Points"].sum().reset_index()
                    final_table = final_table.sort_values("Points", ascending=False).reset_index(drop=True)
                    final_table.index += 1
                    st.dataframe(final_table)

                    # Clasificación de constructores (si existe la columna 'Team')
                    if 'Team' in df_positions.columns:
                        st.markdown("### 🏢 Constructors Championship")
                        constructor_points = df_positions.groupby("Team")["Points"].sum().reset_index()
                        constructor_points = constructor_points.sort_values("Points", ascending=False).reset_index(
                            drop=True)
                        constructor_points.index += 1
                        st.dataframe(constructor_points)

        except Exception as e:
            st.error(f"Error loading season progress: {e}")
