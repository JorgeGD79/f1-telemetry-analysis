import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")
st.title("🏎️ F1 Telemetry Dashboard")


@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.warning(f"No data found at {path}")
        return pd.DataFrame()


# Sidebar
st.sidebar.title("📊 Configuration")
season = st.sidebar.selectbox("Season", ["2023", "2024"])
session_type = st.sidebar.selectbox("Session", ["R", "Q"])
gp_file = st.sidebar.selectbox("Grand Prix", sorted(
    [f for f in os.listdir(f"data/season_{season}_{session_type}") if f.endswith("ALL.csv")]), index=0)
csv_path = f"data/season_{season}_{session_type}/{gp_file}"

with st.spinner("Loading data..."):
    df = load_data(csv_path)

if not df.empty:
    drivers = df['Driver'].unique()
    driver1 = st.sidebar.selectbox("Driver 1", drivers, index=0)
    driver2 = st.sidebar.selectbox("Driver 2", drivers, index=1 if len(drivers) > 1 else 0)

    lap_options = df[df['Driver'] == driver1]['LapNumber'].unique()
    lap_options_2 = df[df['Driver'] == driver2]['LapNumber'].unique()

    lap1 = st.sidebar.selectbox("Lap Driver 1", lap_options, index=0)
    lap2 = st.sidebar.selectbox("Lap Driver 2", lap_options_2, index=0)



    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["🏁 Fast Lap Comparison", "🧭 Racing Line", "Throttle", "NGear", "🧠 Driving Styles", "📍 Pilot Positioning",
         "⏱️ Real Delta", "Race Mode", "📏 Lap Time Comparison"])

    with tab1:
        st.subheader("Speed Comparison")

        lap1_data = df[(df['Driver'] == driver1) & (df['LapNumber'] == lap1)]
        lap2_data = df[(df['Driver'] == driver2) & (df['LapNumber'] == lap2)]

        fig = px.line()
        fig.add_scatter(x=lap1_data['Distance'], y=lap1_data['Speed'], mode='lines', name=f'{driver1} - Lap {lap1}')
        fig.add_scatter(x=lap2_data['Distance'], y=lap2_data['Speed'], mode='lines', name=f'{driver2} - Lap {lap2}')
        fig.update_layout(xaxis_title="Distance (m)", yaxis_title="Speed (km/h)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
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

    with tab3:
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

    with tab4:
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

    with tab5:
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
            X = np.load("X_driving_model.npy")
            y = np.load("y_driving_model.npy")
            FEATURES = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS']
            SAMPLES_PER_LAP = 100
            N_FEATURES = len(FEATURES)

            # Parámetros
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)

            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)

            df_clustered = pd.DataFrame(X)
            df_clustered['Driver'] = y
            df_clustered['Cluster'] = labels

            # Tabla por piloto
            cluster_summary = df_clustered.groupby(['Driver', 'Cluster']).size().unstack(fill_value=0)
            cluster_percent = cluster_summary.div(cluster_summary.sum(axis=1), axis=0)

            st.markdown("### 🔍 Cluster Distribution per Driver")
            st.dataframe(cluster_percent.style.format("{:.2%}"))

            # Heatmap
            st.markdown("### 🎯 Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cluster_percent, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Driving Styles by Driver (Cluster %)")
            st.pyplot(fig)

            # Selector de piloto
            selected_driver = st.selectbox("👤 Select a Driver to analyze dominant style",
                                           sorted(df_clustered['Driver'].unique()))
            if selected_driver:
                driver_distribution = cluster_percent.loc[selected_driver]
                dominant_cluster = driver_distribution.idxmax()

            # Perfiles de estilo
            st.markdown("### 🧬 Cluster Profiles & Interpretation")

            cluster_profiles = np.zeros((n_clusters, N_FEATURES))
            for i in range(n_clusters):
                cluster_data = X[labels == i]
                reshaped = cluster_data.reshape(cluster_data.shape[0], SAMPLES_PER_LAP, N_FEATURES)
                cluster_profiles[i] = reshaped.mean(axis=(0, 1))

            df_profiles = pd.DataFrame(cluster_profiles, columns=FEATURES)
            df_profiles.index.name = "Cluster"

            # Descripciones interpretativas
            descriptions = []
            long_descriptions = []
            for i, row in df_profiles.iterrows():
                desc = []
                long_desc = []
                if row['Throttle'] > 0.75 and row['Brake'] < 0.2:
                    desc.append("Aggressive")
                    long_desc.append("High throttle usage and minimal braking")
                if row['Brake'] > 0.5:
                    desc.append("Heavy Braking")
                    long_desc.append("Frequent or intense use of brakes")
                if row['Speed'] > 250:
                    desc.append("High Speed")
                    long_desc.append("Consistently fast pace on straights and curves")
                if row['Throttle'] < 0.5:
                    desc.append("Conservative")
                    long_desc.append("Careful throttle management")
                if row['nGear'] > 6.5:
                    desc.append("High Gear Usage")
                    long_desc.append("Stays in high gears longer")
                if row['RPM'] > 11000:
                    desc.append("High Revving")
                    long_desc.append("Keeps engine at high RPMs")

                descriptions.append(", ".join(desc) if desc else "Balanced")
                long_descriptions.append(" / ".join(long_desc) if long_desc else "Adaptable across scenarios")

            df_profiles['Description'] = descriptions
            df_profiles['Details'] = long_descriptions

            st.dataframe(df_profiles.style.format("{:.2f}", subset=FEATURES))

            for idx, row in df_profiles.iterrows():
                st.markdown(f"**Cluster {idx}**: {row['Description']}")
                st.markdown(f"_Details_: {row['Details']}")

            if selected_driver:
                dominant_description = df_profiles.loc[dominant_cluster, 'Description']
                dominant_details = df_profiles.loc[dominant_cluster, 'Details']
                st.success(
                    f"**{selected_driver}'s dominant driving style is Cluster {dominant_cluster}: {dominant_description}**")
                st.caption(f"_Details: {dominant_details}_")

            # Radar plot
            st.markdown("### 📊 Radar Chart of Driving Styles")
            radar_fig = go.Figure()
            for i in range(n_clusters):
                radar_fig.add_trace(go.Scatterpolar(
                    r=df_profiles.loc[i, FEATURES].values,
                    theta=FEATURES,
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

    with tab6:
        st.subheader("📍 Pilot Positioning by Driving Style")

        try:
            import numpy as np
            import pandas as pd
            from sklearn.decomposition import PCA
            import plotly.express as px

            # Cargar datos
            X = np.load("X_driving_model.npy")
            y = np.load("y_driving_model.npy")

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

    with tab7:
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

    with tab8:
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

    with tab9:
        st.subheader("📏 Lap Time Comparison")

        try:
            import plotly.express as px

            if 'LapTimeSeconds' not in df.columns:
                st.warning("LapTimeSeconds column not found in the data.")
            else:
                df_laptimes = df[['Driver', 'LapNumber', 'LapTimeSeconds']].drop_duplicates()

                drivers_to_compare = st.multiselect("Select Drivers to Compare", sorted(df_laptimes['Driver'].unique()),
                                                    default=[driver1, driver2])

                df_filtered = df_laptimes[df_laptimes['Driver'].isin(drivers_to_compare)]

                fig = px.line(df_filtered, x="LapNumber", y="LapTimeSeconds", color="Driver",
                              markers=True, title="Lap Time per Driver")
                fig.update_layout(xaxis_title="Lap Number", yaxis_title="Lap Time (s)")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating lap time comparison: {e}")
