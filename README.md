
# 🏎️ F1 Telemetry Analysis

This personal project enables the analysis and comparison of Formula 1 telemetry data using the [FastF1](https://theoehrly.github.io/Fast-F1/) library, Python visualizations, and Power BI dashboards. It includes fastest lap comparisons, multi-lap analysis, stint breakdowns, and data exports for advanced visualization.

---

## 📁 Project Structure

```
f1-telemetry-analysis/
├── data/           # Exported data (CSV for Power BI, FastF1 cache)
├── notebooks/      # Exploratory analysis in Jupyter
├── scripts/        # Reusable scripts and automation
├── visuals/        # Plots generated with matplotlib
├── powerbi/        # .pbix files and dashboard configs
└── README.md       # This file
```

---

## ⚙️ Technologies

- **FastF1** – Access to telemetry, laps, and session data.
- **Python** – Data analysis and processing.
- **Pandas / Matplotlib / Seaborn** – Visualization and data manipulation.
- **Power BI** – Interactive dashboards and reporting.

---

## 🚦 Project Features

### ✅ Fastest Lap Analysis
- Compare two drivers on their fastest lap.
- Speed vs distance plots.
- Variables like `Throttle`, `Brake`, `Gear`, `DRS`, and `RPM`.

### ✅ Multi-Lap Analysis
- Study multiple consecutive laps of a single driver.
- Compare driving style evolution across laps.
- Export data for Power BI.

### ✅ Stint Analysis
- Automatically group laps by stints (between pit stops).
- Visualize pace and lap time evolution per stint.
- Calculate average degradation per compound.

---

## 📊 Exporting to Power BI

Telemetry, fastest laps, and stint data can be exported as `.csv` files and loaded into Power BI for dynamic analysis.

```python
df.to_csv('data/file.csv', index=False)
```

Suggested Power BI visuals:
- Lap time trend line.
- Comparison by driver, compound, or stint.
- Filters for session, driver, track, compound, etc.

---

## 🧪 Scripts and Automation

- `scripts/export_multilap.py`: Extracts multiple laps for analysis.
- `scripts/analyze_stints.py`: Generates stint metrics.
- `scripts/compare_fastest_laps.py`: Compares two drivers on fastest lap.

---

## 🔧 Quick Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install fastf1 pandas matplotlib seaborn jupyter openpyxl
```

---

## 🗂️ To Do / Future Improvements

- Add racing line (X/Y track map) analysis.
- Detect undercut/overcut strategies.
- Automate full season data download.
- Build interactive web dashboards with Dash or Streamlit.

---

## 📬 Contact

Personal project developed by Jorge Garcia.  
Ideas, suggestions, or improvements are welcome!
