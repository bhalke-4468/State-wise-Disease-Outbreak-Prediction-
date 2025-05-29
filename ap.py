# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from statsmodels.tsa.stattools import adfuller
# from datetime import datetime, timedelta
# import warnings
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import LSTM, Dense # type: ignore
# from sklearn.metrics import mean_squared_error, r2_score

# warnings.filterwarnings("ignore")

# # Custom CSS for styling with theme support
# def apply_theme(theme):
#     if theme == "Dark":
#         css = """
#             <style>
#             .main { background-color: #2c3e50; color: #ecf0f1; font-family: 'Arial', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(255,255,255,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(255,255,255,0.1); }
#             .stMetric { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(255,255,255,0.1); margin: 10px 0; }
#             .card { background-color: #34495e; color: #ecf0f1; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(255,255,255,0.1); }
#             h1, h2, h3, h4 { color: #ecf0f1; }
#             .sidebar .sidebar-content { background-color: #2c3e50; color: #ecf0f1; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: #34495e; }
#             </style>
#         """
#     else:  # Light theme
#         css = """
#             <style>
#             .main { background-color: #f5f7fa; font-family: 'Arial', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: white; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#             .stMetric { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 10px 0; }
#             .card { background-color: white; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
#             h1, h2, h3, h4 { color: #2c3e50; }
#             .sidebar .sidebar-content { background-color: #34495e; color: white; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: white; }
#             </style>
#         """
#     st.markdown(css, unsafe_allow_html=True)

# # Initialize session state
# if 'df_covid' not in st.session_state:
#     st.session_state.df_covid = None
# if 'df_malaria' not in st.session_state:
#     st.session_state.df_malaria = None
# if 'user_profile' not in st.session_state:
#     st.session_state.user_profile = {'name': '', 'email': '', 'analyses_run': 0}
# if 'settings' not in st.session_state:
#     st.session_state.settings = {'theme': 'Light', 'look_back_period': 60}

# # Apply theme
# apply_theme(st.session_state.settings['theme'])

# # Sidebar navigation
# st.sidebar.title("🦠 Disease Prediction Dashboard")
# st.sidebar.markdown("Analyze and predict COVID-19 and malaria cases in India.")
# page = st.sidebar.radio("Go to:", ["Profile", "Home", "COVID-19 Analysis", "Malaria Analysis", "Combined Insights", "Settings"])

# # Header
# st.markdown("""
#     <div style="text-align: center; padding: 20px; background-color: #34495e; color: white; border-radius: 10px;">
#         <h1>🦠 Disease Prediction Dashboard</h1>
#         <p style="font-size: 18px;">Analyze and predict COVID-19 and malaria cases across India using LSTM and ARIMA models.</p>
#     </div>
# """, unsafe_allow_html=True)

# # LSTM Forecasting Function
# @st.cache_data
# def lstm_forecast(series, steps, look_back=None):
#     try:
#         look_back = look_back if look_back is not None else st.session_state.settings['look_back_period']
#         if series is None or len(series) == 0:
#             raise ValueError("Input series is None or empty")
#         if series.isna().all():
#             raise ValueError("Input series contains only NaN values")
#         if len(series) < look_back:
#             raise ValueError(f"Input series has {len(series)} data points, but look_back={look_back} is required")

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         series_values = series.values.reshape(-1, 1)
#         scaled_series = scaler.fit_transform(series_values)

#         def create_sequences(data, look_back):
#             X, y = [], []
#             for i in range(len(data) - look_back):
#                 X.append(data[i:(i + look_back), 0])
#                 y.append(data[i + look_back, 0])
#             return np.array(X), np.array(y)

#         X, y = create_sequences(scaled_series, look_back)
#         if len(X) == 0 or len(y) == 0:
#             st.warning(f"No sequences created. Series length={len(series)}, look_back={look_back}. Returning flat forecast.")
#             return pd.Series(
#                 [series.iloc[-1] if not pd.isna(series.iloc[-1]) else 0] * steps,
#                 index=pd.date_range(start=series.index[-1] + timedelta(days=1), periods=steps, freq='D')
#             ), float('nan'), float('nan')

#         X = X.reshape((X.shape[0], X.shape[1], 1))
#         train_size = int(len(X) * 0.8)
#         if train_size == 0:
#             raise ValueError("Training set is empty after splitting. Increase data size or reduce look_back.")

#         X_train, X_val = X[:train_size], X[train_size:]
#         y_train, y_val = y[:train_size], y[train_size:]

#         model = Sequential()
#         model.add(LSTM(50, activation='relu', input_shape=(look_back, 1), return_sequences=True))
#         model.add(LSTM(50, activation='relu'))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_val, y_val))

#         last_sequence = scaled_series[-look_back:].reshape((1, look_back, 1))
#         forecast = []
#         current_sequence = last_sequence.copy()
#         for _ in range(steps):
#             pred = model.predict(current_sequence, verbose=0)
#             forecast.append(pred[0, 0])
#             current_sequence = np.roll(current_sequence, -1, axis=1)
#             current_sequence[0, -1, 0] = pred[0, 0]

#         forecast = np.array(forecast).reshape(-1, 1)
#         forecast = scaler.inverse_transform(forecast).flatten()
#         forecast_series = pd.Series(
#             forecast,
#             index=pd.date_range(start=series.index[-1] + timedelta(days=1), periods=steps, freq='D')
#         )

#         mse, r2 = float('nan'), float('nan')
#         if len(X_val) > 0:
#             val_pred = model.predict(X_val, verbose=0)
#             val_pred = scaler.inverse_transform(val_pred).flatten()
#             val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
#             mse = mean_squared_error(val_true, val_pred)
#             r2 = r2_score(val_true, val_pred)

#         return forecast_series.clip(lower=0), mse, r2
#     except Exception as e:
#         st.error(f"Error in LSTM forecasting: {str(e)}")
#         default_start_date = pd.to_datetime('2025-05-19') if series is None or series.empty else series.index[-1]
#         return pd.Series(
#             [0] * steps,
#             index=pd.date_range(start=default_start_date + timedelta(days=1), periods=steps, freq='D')
#         ), float('nan'), float('nan')

# # COVID-19 Data Parsing
# @st.cache_data
# def parse_covid_data(raw_data):
#     try:
#         lines = raw_data.strip().split('\n')
#         headers = lines[0].split()
#         data = [line.split(maxsplit=len(headers)-1) for line in lines[1:]]
#         df = pd.DataFrame(data, columns=headers)
#         df['Date_reported'] = pd.to_datetime(df['Date_reported'], format='%d-%m-%Y')
#         for col in ['New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
#         df['New_cases'] = df['New_cases'].clip(lower=0)
#         df['New_deaths'] = df['New_deaths'].clip(lower=0)
#         return df
#     except Exception as e:
#         st.error(f"Error parsing COVID-19 data: {str(e)}")
#         return None

# # COVID-19 Data Simulation
# @st.cache_data
# def generate_simulated_covid_data(initial_data):
#     try:
#         df = initial_data.copy()
#         last_date = df['Date_reported'].max()
#         last_cumulative_cases = df['Cumulative_cases'].iloc[-1]
#         last_cumulative_deaths = df['Cumulative_deaths'].iloc[-1]
#         end_date = pd.to_datetime('2025-05-19')

#         def generate_wave(start_day, peak_day, end_day, start_cases, peak_cases):
#             days = []
#             for day in range(start_day, peak_day + 1):
#                 progress = (day - start_day) / (peak_day - start_day)
#                 cases = int(start_cases + (peak_cases - start_cases) * progress * progress)
#                 days.append(cases)
#             for day in range(peak_day + 1, end_day + 1):
#                 progress = (day - peak_day) / (end_day - peak_day)
#                 cases = int(peak_cases * (1 - progress * progress))
#                 days.append(cases)
#             return days

#         wave1 = generate_wave(0, 150, 300, 100, 100000)
#         wave2 = generate_wave(0, 40, 100, 30000, 400000)
#         wave3 = generate_wave(0, 30, 90, 10000, 350000)
#         wave4 = generate_wave(0, 20, 60, 5000, 50000)
#         wave5 = generate_wave(0, 15, 45, 2000, 25000)
#         wave6 = generate_wave(0, 15, 40, 1000, 15000)
#         wave7 = generate_wave(0, 10, 30, 500, 8000)

#         def get_new_cases(date):
#             year = date.year
#             month = date.month
#             total_days = (date - pd.to_datetime('2020-03-01')).days
#             if year == 2020 or (year == 2021 and month < 3):
#                 day_in_wave = total_days % len(wave1)
#                 return wave1[day_in_wave]
#             elif year == 2021 and 3 <= month <= 6:
#                 day_in_wave = (total_days - (pd.to_datetime('2021-03-01') - pd.to_datetime('2020-03-01')).days) % len(wave2)
#                 return wave2[day_in_wave]
#             elif (year == 2021 and month >= 12) or (year == 2022 and month <= 2):
#                 day_in_wave = (total_days - (pd.to_datetime('2021-12-01') - pd.to_datetime('2020-03-01')).days) % len(wave3)
#                 return wave3[day_in_wave]
#             elif year == 2022 and 6 <= month <= 8:
#                 day_in_wave = (total_days - (pd.to_datetime('2022-06-01') - pd.to_datetime('2020-03-01')).days) % len(wave4)
#                 return wave4[day_in_wave]
#             elif year == 2023 and 1 <= month <= 3:
#                 day_in_wave = (total_days - (pd.to_datetime('2023-01-01') - pd.to_datetime('2020-03-01')).days) % len(wave5)
#                 return wave5[day_in_wave]
#             elif year == 2023 and 9 <= month <= 11:
#                 day_in_wave = (total_days - (pd.to_datetime('2023-09-01') - pd.to_datetime('2020-03-01')).days) % len(wave6)
#                 return wave6[day_in_wave]
#             elif year == 2024 and 3 <= month <= 5:
#                 day_in_wave = (total_days - (pd.to_datetime('2024-03-01') - pd.to_datetime('2020-03-01')).days) % len(wave7)
#                 return wave7[day_in_wave]
#             elif year == 2024 and 10 <= month <= 12:
#                 day_in_wave = (total_days - (pd.to_datetime('2024-10-01') - pd.to_datetime('2020-03-01')).days) % len(wave7)
#                 return int(wave7[day_in_wave] * 0.7)
#             elif year == 2025 and 1 <= month <= 4:
#                 day_in_wave = (total_days - (pd.to_datetime('2025-01-01') - pd.to_datetime('2020-03-01')).days) % len(wave7)
#                 return int(wave7[day_in_wave] * 0.5)
#             else:
#                 return np.random.randint(500, 1500)

#         current_date = last_date + timedelta(days=1)
#         new_rows = []
#         while current_date <= end_date:
#             new_cases = get_new_cases(current_date)
#             last_cumulative_cases += new_cases
#             death_rate = 0.02 if current_date <= pd.to_datetime('2021-03-01') else \
#                          0.012 if current_date <= pd.to_datetime('2021-12-01') else \
#                          0.005 if current_date <= pd.to_datetime('2022-07-01') else \
#                          0.002 if current_date <= pd.to_datetime('2023-01-01') else \
#                          0.001 if current_date <= pd.to_datetime('2024-01-01') else 0.0005
#             new_deaths = int(new_cases * death_rate)
#             last_cumulative_deaths += new_deaths
#             new_rows.append({
#                 'Date_reported': current_date,
#                 'Country': 'India',
#                 'WHO_region': 'SEAR',
#                 'New_cases': new_cases,
#                 'Cumulative_cases': last_cumulative_cases,
#                 'New_deaths': new_deaths,
#                 'Cumulative_deaths': last_cumulative_deaths
#             })
#             current_date += timedelta(days=1)
#         full_data = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
#         full_data = full_data.sort_values('Date_reported')
#         return full_data
#     except Exception as e:
#         st.error(f"Error generating simulated COVID-19 data: {str(e)}")
#         return None

# # Malaria Data Preprocessing (Updated to Use File Path)
# @st.cache_data
# def preprocess_malaria_data(file_path):
#     try:
#         # Read the CSV file from the provided file path
#         df_malaria = pd.read_csv(file_path)
#         if df_malaria.empty:
#             return None
#         required_columns = ['STATE_UT', 'BSE_2020', 'Malaria_2020']
#         if not all(col in df_malaria.columns for col in required_columns):
#             df_malaria.columns = df_malaria.iloc[0]
#             df_malaria = df_malaria[1:].reset_index(drop=True)
#             df_malaria = df_malaria.iloc[1:-1]
#         df_malaria.columns = [
#             "Sr", "STATE_UT", "BSE_2020", "Malaria_2020", "Pf_2020", "Deaths_2020",
#             "BSE_2021", "Malaria_2021", "Pf_2021", "Deaths_2021",
#             "BSE_2022", "Malaria_2022", "Pf_2022", "Deaths_2022",
#             "BSE_2023", "Malaria_2023", "Pf_2023", "Deaths_2023",
#             "BSE_2024", "Malaria_2024", "Pf_2024", "Deaths_2024"
#         ]
#         num_cols = df_malaria.columns[2:]
#         df_malaria[num_cols] = df_malaria[num_cols].apply(pd.to_numeric, errors="coerce")
#         df_malaria.drop('Sr', axis=1, inplace=True)
#         return df_malaria
#     except Exception as e:
#         st.error(f"Error preprocessing malaria data: {str(e)}")
#         return None

# # Malaria Visualizations
# @st.cache_data
# def generate_malaria_visualizations(df_malaria):
#     try:
#         visualizations = {}
#         numeric_cols = df_malaria.select_dtypes(include=[np.number]).columns
#         df_numeric = df_malaria[numeric_cols]
#         corr_matrix = df_numeric.corr()
#         fig1, ax1 = plt.subplots(figsize=(12, 8))
#         sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Spectral", linewidths=0.5, ax=ax1)
#         ax1.set_title("Correlation Matrix for Malaria Cases", fontsize=14, pad=15)
#         visualizations['corr_matrix'] = fig1

#         fig2, ax2 = plt.subplots(figsize=(12, 6))
#         palette = sns.color_palette("Set2", 6)
#         for i, year in enumerate(range(2020, 2025)):
#             col = f"Malaria_{year}"
#             if col in df_malaria.columns:
#                 sns.kdeplot(df_malaria[col], label=f"{year} (Actual)", fill=True, alpha=0.5, color=palette[i], ax=ax2)
#         if "Predicted_Malaria_2025" in df_malaria.columns:
#             sns.kdeplot(df_malaria["Predicted_Malaria_2025"], label="2025 (Forecast)", fill=True, alpha=0.5, 
#                         color=palette[5], linestyle='--', ax=ax2)
#         ax2.set_xlabel("Number of Malaria Cases", fontsize=12)
#         ax2.set_ylabel("Density", fontsize=12)
#         ax2.legend(title="Year")
#         ax2.set_title("Malaria Cases Distribution (2020-2025)", fontsize=14, pad=15)
#         visualizations['kde_plot'] = fig2

#         malaria_cols = [col for col in df_malaria.columns if "Malaria_" in col]
#         df_malaria['Avg_Malaria'] = df_malaria[malaria_cols].mean(axis=1)
#         top_states = df_malaria.nlargest(10, 'Avg_Malaria')[['STATE_UT', 'Avg_Malaria']]
#         fig3, ax3 = plt.subplots(figsize=(10, 6))
#         sns.barplot(data=top_states, y='STATE_UT', x='Avg_Malaria', palette='magma', ax=ax3)
#         ax3.set_title("Top 10 States by Average Malaria Cases (2020-2024)", fontsize=14, pad=15)
#         ax3.set_xlabel("Average Malaria Cases", fontsize=12)
#         ax3.set_ylabel("State/UT", fontsize=12)
#         visualizations['top_states'] = fig3
#         return visualizations
#     except Exception as e:
#         st.error(f"Error generating malaria visualizations: {str(e)}")
#         return {}

# # Malaria Feature Engineering
# @st.cache_data
# def perform_malaria_feature_engineering(df_malaria):
#     try:
#         df_malaria = df_malaria.copy()
#         malaria_cols = [f"Malaria_{y}" for y in range(2020, 2025)]
#         df_malaria["Avg_Malaria_Cases"] = df_malaria[malaria_cols].mean(axis=1)

#         def categorize_risk(row):
#             avg = row["Avg_Malaria_Cases"]
#             return "High-Risk" if avg > 15000 else "Medium-Risk" if avg > 5000 else "Low-Risk"

#         df_malaria["Risk_Category"] = df_malaria.apply(categorize_risk, axis=1)
#         df_malaria["Avg_Malaria_Cases"] = df_malaria["Avg_Malaria_Cases"].round(0).astype(int)

#         visualizations = {}
#         fig4, ax4 = plt.subplots(figsize=(8, 5))
#         sns.countplot(data=df_malaria, x='Risk_Category', palette='Set2', ax=ax4)
#         ax4.set_title("Risk Category Distribution", fontsize=14, pad=15)
#         ax4.set_xlabel("Risk Category", fontsize=12)
#         ax4.set_ylabel("Count", fontsize=12)
#         visualizations['risk_category'] = fig4

#         fig5, ax5 = plt.subplots(figsize=(15, 8))
#         palette = {"High-Risk": "#e74c3c", "Medium-Risk": "#f39c12", "Low-Risk": "#2ecc71"}
#         sns.barplot(data=df_malaria, x="STATE_UT", y="Avg_Malaria_Cases", hue="Risk_Category",
#                     palette=palette, hue_order=["High-Risk", "Medium-Risk", "Low-Risk"], ax=ax5)
#         ax5.set_title("State-wise Malaria Risk Categorization", fontsize=14, pad=15)
#         ax5.set_xlabel("State/UT", fontsize=12)
#         ax5.set_ylabel("Avg. Malaria Cases", fontsize=12)
#         ax5.tick_params(axis='x', rotation=90)
#         visualizations['state_risk'] = fig5
#         return df_malaria, visualizations
#     except Exception as e:
#         st.error(f"Error in malaria feature engineering: {str(e)}")
#         return df_malaria, {}

# # Malaria ARIMA Predictions
# @st.cache_data
# def run_malaria_arima(df_malaria):
#     try:
#         from statsmodels.tsa.arima.model import ARIMA
#         df_malaria = df_malaria.copy()
#         malaria_cols = [f"Malaria_{y}" for y in range(2020, 2025)]
#         predictions = []
#         mse_total = 0
#         actuals = []
#         forecasts = []

#         for index, row in df_malaria.iterrows():
#             series = [row[col] for col in malaria_cols]
#             series = [0 if pd.isna(x) else x for x in series]
#             train = series[:4]
#             test = series[4]
#             try:
#                 model = ARIMA(train, order=(1, 1, 0))
#                 model_fit = model.fit()
#                 forecast = model_fit.forecast(steps=2)
#                 pred_2024 = forecast[0]
#                 pred_2025 = forecast[1]
#                 predictions.append(pred_2025 if pred_2025 > 0 else 0)
#                 mse_total += (test - pred_2024) ** 2
#                 actuals.append(test)
#                 forecasts.append(pred_2024)
#             except Exception:
#                 predictions.append(series[-1] if series[-1] > 0 else 0)

#         df_malaria["Predicted_Malaria_2025"] = predictions
#         mse = mse_total / len(df_malaria) if len(df_malaria) > 0 else float('nan')
#         mean_actual = np.mean(actuals) if actuals else float('nan')
#         ss_tot = sum((a - mean_actual) ** 2 for a in actuals) if actuals else float('nan')
#         ss_res = sum((a - f) ** 2 for a, f in zip(actuals, forecasts)) if actuals else float('nan')
#         r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

#         metrics = {
#             "mse": mse,
#             "r2": r2,
#             "accuracy": r2 * 100 if not np.isnan(r2) else float('nan')
#         }
#         return df_malaria, metrics
#     except Exception as e:
#         st.error(f"Error in ARIMA forecasting: {str(e)}")
#         return df_malaria, {"mse": float('nan'), "r2": float('nan'), "accuracy": float('nan')}

# # Risk Classification
# def classify_risk(cases, disease="malaria"):
#     if disease == "malaria":
#         if cases > 50000:
#             return ("🔴 High Risk", "#e74c3c")
#         elif cases > 25000:
#             return ("🟠 Medium Risk", "#f39c12")
#         elif cases > 10000:
#             return ("🟡 Moderate Risk", "#f1c40f")
#         else:
#             return ("🟢 Low Risk", "#2ecc71")
#     else:  # COVID-19
#         if cases > 10000:
#             return ("🔴 High Risk", "#e74c3c")
#         elif cases > 5000:
#             return ("🟠 Medium Risk", "#f39c12")
#         elif cases > 1000:
#             return ("🟡 Moderate Risk", "#f1c40f")
#         else:
#             return ("🟢 Low Risk", "#2ecc71")

# # Format Number
# def format_number(num):
#     if pd.isna(num):
#         return "N/A"
#     if num >= 1_000_000:
#         return f"{num / 1_000_000:.1f}M"
#     elif num >= 1_000:
#         return f"{num / 1_000:.1f}K"
#     return int(num)

# # Page: Home (Updated to Remove Manual File Upload and Use File Path)
# if page == "Home":
#     st.markdown("""
#         <div class="card">
#             <h2>Welcome to the Disease Prediction Dashboard</h2>
#             <p style="color: #7f8c8d;">
#                 Load datasets for COVID-19 and malaria to analyze and predict cases across India using LSTM and ARIMA models.
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

#     st.subheader("Load Datasets")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**COVID-19 Data**")
#         st.info("COVID-19 data is preloaded from January 2020 to May 2025.")
#         if st.button("🚀 Process COVID-19 Data"):
#             with st.spinner("Processing COVID-19 data..."):
#                 try:
#                     raw_data = """Date_reported Country WHO_region New_cases Cumulative_cases New_deaths Cumulative_deaths
# 04-01-2020 India SEAR 0 0 0 0
# 05-01-2020 India SEAR 1 1 0 0
# 06-01-2020 India SEAR 2 3 1 1

# """
#                     df_covid = parse_covid_data(raw_data)
#                     if df_covid is None:
#                         st.error("Failed to parse COVID-19 data.")
#                         st.stop()
#                     full_covid_data = generate_simulated_covid_data(df_covid)
#                     if full_covid_data is None or full_covid_data['New_cases'].isna().any() or (full_covid_data['New_cases'] < 0).any():
#                         st.error("Simulated COVID-19 data contains invalid values (NaN or negative).")
#                         st.stop()
#                     st.session_state.df_covid = full_covid_data
#                     st.session_state.user_profile['analyses_run'] += 1
#                     st.success("✅ COVID-19 data processed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing COVID-19 data: {str(e)}")
#                     st.stop()
#     with col2:
#         st.markdown("**Malaria Data**")
#         malaria_file_path = st.text_input("Enter the file path to the malaria dataset (CSV format)", value="", key="malaria_file_path")
#         if malaria_file_path and st.button("🚀 Process Malaria Dataset"):
#             with st.spinner("Processing malaria dataset..."):
#                 try:
#                     df_malaria = preprocess_malaria_data(malaria_file_path)
#                     if df_malaria is None:
#                         st.error("Failed to process malaria dataset. Please check the file path and ensure the file is a valid CSV.")
#                         st.stop()
#                     st.session_state.df_malaria = df_malaria
#                     st.session_state.user_profile['analyses_run'] += 1
#                     st.success("✅ Malaria dataset processed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing malaria dataset: {str(e)}")
#                     st.stop()

# # Page: COVID-19 Analysis
# if page == "COVID-19 Analysis":
#     if st.session_state.df_covid is None:
#         st.warning("COVID-19 data is not loaded. Please go to the Home page and process the COVID-19 data.")
#         st.stop()
    
#     st.markdown("<div class='card'><h2>COVID-19 Case and Death Forecast</h2></div>", unsafe_allow_html=True)
#     df_covid = st.session_state.df_covid
#     current_datetime = datetime(2025, 5, 25, 10, 54)  # Updated system-provided date and time
#     st.write(f"**Current Date and Time:** {current_datetime.strftime('%A, %B %d, %Y %I:%M %p IST')}")
#     st.markdown("""
#         Showing historical data from January 2020 to May 19, 2025, with a forecast for the user-specified date range.
#         **Note**: Forecast uses LSTM with a configurable look-back period. Actual numbers may vary due to changes in testing rates, public health measures, new variants, or vaccination status.
#     """)

#     # Prepare historical data
#     historical_df = df_covid.set_index('Date_reported')[['New_cases', 'New_deaths', 'Cumulative_cases', 'Cumulative_deaths']]
#     last_historical_date = historical_df.index.max()

#     # Validate historical data
#     if historical_df.empty or historical_df['New_cases'].isna().all() or len(historical_df) < st.session_state.settings['look_back_period']:
#         st.error(f"Insufficient or invalid COVID-19 data for forecasting. Please ensure the dataset has at least {st.session_state.settings['look_back_period']} days of valid data.")
#         st.stop()

#     # User input for forecast date range
#     st.subheader("Select Forecast Date Range")
#     col1, col2 = st.columns(2)
#     with col1:
#         forecast_start_date = st.date_input(
#             "Forecast Start Date",
#             value=pd.to_datetime('2025-05-20'),
#             min_value=(last_historical_date + timedelta(days=1)).date(),
#             max_value=pd.to_datetime('2030-12-31').date()
#         )
#     with col2:
#         forecast_end_date = st.date_input(
#             "Forecast End Date",
#             value=pd.to_datetime('2025-06-18'),
#             min_value=(pd.to_datetime(forecast_start_date) + timedelta(days=1)).date(),
#             max_value=pd.to_datetime('2030-12-31').date()
#         )

#     forecast_start_date = pd.to_datetime(forecast_start_date)
#     forecast_end_date = pd.to_datetime(forecast_end_date)

#     if forecast_end_date <= forecast_start_date:
#         st.error("End date must be after start date.")
#         st.stop()
#     if forecast_start_date <= last_historical_date:
#         st.error(f"Start date must be after the last historical date ({last_historical_date.strftime('%Y-%m-%d')}).")
#         st.stop()

#     forecast_steps = (forecast_end_date - last_historical_date).days
#     if forecast_steps <= 0:
#         st.error("Forecast period must be in the future.")
#         st.stop()

#     # Forecast with LSTM
#     with st.spinner("Generating COVID-19 forecast..."):
#         historical_days = len(historical_df['New_cases'])
#         days_to_use = min(365, historical_days)
#         if days_to_use < st.session_state.settings['look_back_period']:
#             st.error(f"Insufficient data for LSTM forecasting. Only {days_to_use} days available, need at least {st.session_state.settings['look_back_period']}.")
#             st.stop()

#         forecast_dates = pd.date_range(start=last_historical_date + timedelta(days=1), periods=forecast_steps, freq='D')
#         forecast_df = pd.DataFrame(index=forecast_dates)

#         forecast_cases, mse_cases, r2_cases = lstm_forecast(historical_df['New_cases'][-days_to_use:], forecast_steps)
#         forecast_deaths, mse_deaths, r2_deaths = lstm_forecast(historical_df['New_deaths'][-days_to_use:], forecast_steps)
#         st.session_state.user_profile['analyses_run'] += 1

#         forecast_df['New_cases'] = forecast_cases.values
#         forecast_df['New_deaths'] = forecast_deaths.values
#         last_cumulative_cases = historical_df['Cumulative_cases'].iloc[-1]
#         last_cumulative_deaths = historical_df['Cumulative_deaths'].iloc[-1]
#         forecast_df['Cumulative_cases'] = last_cumulative_cases + forecast_df['New_cases'].cumsum()
#         forecast_df['Cumulative_deaths'] = last_cumulative_deaths + forecast_df['New_deaths'].cumsum()
#         forecast_df['Type'] = 'Forecast'
#         historical_df['Type'] = 'Historical'
#         combined_df = pd.concat([historical_df, forecast_df])

#         col1, col2 = st.columns(2)
#         col1.metric("Cases MSE (Validation)", f"{mse_cases:.2f}" if not np.isnan(mse_cases) else "N/A", help="Mean Squared Error on validation set")
#         col2.metric("Cases R² Score (Validation)", f"{r2_cases:.4f}" if not np.isnan(r2_cases) else "N/A", help="R² Score indicating model fit")
#         col3, col4 = st.columns(2)
#         col3.metric("Deaths MSE (Validation)", f"{mse_deaths:.2f}" if not np.isnan(mse_deaths) else "N/A", help="Mean Squared Error on validation set")
#         col4.metric("Deaths R² Score (Validation)", f"{r2_deaths:.4f}" if not np.isnan(r2_deaths) else "N/A", help="R² Score indicating model fit")

#     # Summary Statistics
#     last_14_days = historical_df[-14:]['New_cases']
#     last_7_days = historical_df[-7:]['New_cases']
#     avg_7_day = last_7_days.mean()
#     avg_14_day = last_14_days.mean()
#     trend = ((avg_7_day - avg_14_day) / avg_14_day * 100) if avg_14_day != 0 else 0
#     forecast_avg = forecast_df['New_cases'].mean()
#     forecast_total = forecast_df['New_cases'].sum()

#     col1, col2, col3 = st.columns(3)
#     col1.metric("7-Day Average (Current)", format_number(avg_7_day), f"{trend:.1f}% {'↑' if trend >= 0 else '↓'}",
#                 delta_color="normal" if trend >= 0 else "inverse")
#     col2.metric(f"Average Forecast ({forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')})",
#                 format_number(forecast_avg))
#     col3.metric(f"Total Forecast Cases ({forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')})",
#                 format_number(forecast_total))

#     # View Selection
#     view = st.radio("Select View", ["All Data", "Recent (90 Days)", "Forecast Focus"], horizontal=True)
#     if view == "Recent (90 Days)":
#         display_df = combined_df[-120:]
#     elif view == "Forecast Focus":
#         display_df = combined_df[-60:]
#     else:
#         display_df = combined_df

#     # Plot Cases
#     if display_df.empty or display_df['New_cases'].isna().all():
#         st.error("No valid data available to plot COVID-19 cases.")
#         st.stop()
#     display_df.index = pd.to_datetime(display_df.index)
#     fig_cases = go.Figure()
#     fig_cases.add_trace(go.Scatter(
#         x=display_df.index, y=display_df['New_cases'], mode='lines', name='Daily New Cases', line=dict(color='#3498db')))
#     fig_cases.add_trace(go.Scatter(
#         x=display_df[display_df['Type'] == 'Forecast'].index, y=display_df[display_df['Type'] == 'Forecast']['New_cases'],
#         mode='lines', name='Forecast Cases', line=dict(color='#e74c3c', dash='dash')))
#     fig_cases.update_layout(
#         title="COVID-19 New Cases: Historical and Forecasted (LSTM)", xaxis_title="Date", yaxis_title="New Cases",
#         height=400, template="plotly_white", title_x=0.5)
#     st.plotly_chart(fig_cases, use_container_width=True)

#     # Plot Deaths
#     if display_df.empty or display_df['New_deaths'].isna().all():
#         st.error("No valid data available to plot COVID-19 deaths.")
#         st.stop()
#     fig_deaths = go.Figure()
#     fig_deaths.add_trace(go.Scatter(
#         x=display_df.index, y=display_df['New_deaths'], mode='lines', name='Daily New Deaths', line=dict(color='#2ecc71')))
#     fig_deaths.add_trace(go.Scatter(
#         x=display_df[display_df['Type'] == 'Forecast'].index, y=display_df[display_df['Type'] == 'Forecast']['New_deaths'],
#         mode='lines', name='Forecast Deaths', line=dict(color='#f39c12', dash='dash')))
#     fig_deaths.update_layout(
#         title="COVID-19 New Deaths: Historical and Forecasted (LSTM)", xaxis_title="Date", yaxis_title="New Deaths",
#         height=400, template="plotly_white", title_x=0.5)
#     st.plotly_chart(fig_deaths, use_container_width=True)

#     # Forecast Table
#     st.subheader(f"Forecast from {forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')}")
#     forecast_display = forecast_df[['New_cases', 'Cumulative_cases']].copy()
#     forecast_display['New_cases'] = forecast_display['New_cases'].round(0).astype(int)
#     forecast_display['Cumulative_cases'] = forecast_display['Cumulative_cases'].round(0).astype(int)
#     forecast_display = forecast_display.reset_index().rename(columns={'index': 'Date'})
#     forecast_display['Date'] = forecast_display['Date'].dt.strftime('%d-%m-%Y')
#     st.dataframe(forecast_display.head(10).style.set_table_styles([
#         {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#         {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#     ]), use_container_width=True)
#     if len(forecast_display) > 10:
#         st.write(f"Showing 10 of {len(forecast_display)} forecast days")

# # Page: Malaria Analysis
# if page == "Malaria Analysis":
#     if st.session_state.df_malaria is None:
#         st.warning("Malaria data is not loaded. Please go to the Home page and load the malaria data.")
#         st.stop()

#     st.markdown("<div class='card'><h2>Malaria Case Prediction</h2></div>", unsafe_allow_html=True)
#     df_malaria = st.session_state.df_malaria

#     st.header("📈 Data Visualizations")
#     with st.spinner("Generating visualizations..."):
#         visualizations = {}
        
#         # Updated Correlation Matrix for BSE_2020 to Deaths_2024 with Enhanced Styling
#         relevant_columns = [
#             'BSE_2020', 'Malaria_2020', 'Pf_2020', 'Deaths_2020',
#             'BSE_2021', 'Malaria_2021', 'Pf_2021', 'Deaths_2021',
#             'BSE_2022', 'Malaria_2022', 'Pf_2022', 'Deaths_2022',
#             'BSE_2023', 'Malaria_2023', 'Pf_2023', 'Deaths_2023',
#             'BSE_2024', 'Malaria_2024', 'Pf_2024', 'Deaths_2024'
#         ]

#         # Ensure all columns exist in the dataset
#         available_columns = [col for col in relevant_columns if col in df_malaria.columns]
#         if not available_columns:
#             st.error("None of the requested columns (BSE_2020 to Deaths_2024) are available in the dataset.")
#             st.stop()

#         # Filter the dataset for the relevant columns
#         df_subset = df_malaria[available_columns]

#         # Convert to numeric, handle any non-numeric values
#         df_subset = df_subset.apply(pd.to_numeric, errors='coerce')

#         # Compute the correlation matrix
#         corr_matrix = df_subset.corr()

#         # Visualize the correlation matrix using a styled heatmap
#         fig1, ax1 = plt.subplots(figsize=(14, 10), facecolor='#f5f5f5')
#         sns.heatmap(
#             corr_matrix,
#             annot=True,
#             fmt=".2f",
#             cmap="coolwarm",
#             linewidths=1,
#             linecolor='white',
#             cbar_kws={
#                 'label': 'Correlation Coefficient',
#                 'shrink': 0.8,
#                 'ticks': [-1, -0.5, 0, 0.5, 1]
#             },
#             vmin=-1, vmax=1,
#             center=0,
#             square=True,
#             ax=ax1
#         )

#         for i in range(len(corr_matrix)):
#             for j in range(len(corr_matrix)):
#                 value = corr_matrix.iloc[i, j]
#                 if (value > 0.7 or value < -0.7) and i != j:
#                     ax1.text(
#                         j + 0.5, i + 0.5, f"{value:.2f}",
#                         ha='center', va='center', color='red',
#                         fontweight='bold', fontsize=10
#                     )

#         ax1.set_title(
#             "Correlation Matrix: Malaria Data (2020-2024)",
#             fontsize=16, pad=20, color='#2c3e50', fontweight='bold'
#         )
#         ax1.set_xlabel("Features", fontsize=12, color='#2c3e50')
#         ax1.set_ylabel("Features", fontsize=12, color='#2c3e50')
#         plt.xticks(rotation=45, ha='right', fontsize=10, color='#2c3e50')
#         plt.yticks(rotation=0, fontsize=10, color='#2c3e50')
#         plt.tight_layout()
#         visualizations['corr_matrix'] = fig1

#         fig2, ax2 = plt.subplots(figsize=(12, 6))
#         palette = sns.color_palette("Set2", 6)
#         for i, year in enumerate(range(2020, 2025)):
#             col = f"Malaria_{year}"
#             if col in df_malaria.columns:
#                 sns.kdeplot(df_malaria[col], label=f"{year} (Actual)", fill=True, alpha=0.5, color=palette[i], ax=ax2)
#         if "Predicted_Malaria_2025" in df_malaria.columns:
#             sns.kdeplot(df_malaria["Predicted_Malaria_2025"], label="2025 (Forecast)", fill=True, alpha=0.5, 
#                         color=palette[5], linestyle='--', ax=ax2)
#         ax2.set_xlabel("Number of Malaria Cases", fontsize=12)
#         ax2.set_ylabel("Density", fontsize=12)
#         ax2.legend(title="Year")
#         ax2.set_title("Malaria Cases Distribution (2020-2025)", fontsize=14, pad=15)
#         visualizations['kde_plot'] = fig2

#         malaria_cols = [col for col in df_malaria.columns if "Malaria_" in col]
#         df_malaria['Avg_Malaria'] = df_malaria[malaria_cols].mean(axis=1)
#         top_states = df_malaria.nlargest(10, 'Avg_Malaria')[['STATE_UT', 'Avg_Malaria']]
#         fig3, ax3 = plt.subplots(figsize=(10, 6))
#         sns.barplot(data=top_states, y='STATE_UT', x='Avg_Malaria', palette='magma', ax=ax3)
#         ax3.set_title("Top 10 States by Average Malaria Cases (2020-2024)", fontsize=14, pad=15)
#         ax3.set_xlabel("Average Malaria Cases", fontsize=12)
#         ax3.set_ylabel("State/UT", fontsize=12)
#         visualizations['top_states'] = fig3

#     with st.container():
#         if 'corr_matrix' in visualizations:
#             st.subheader("Correlation Matrix (BSE_2020 to Deaths_2024)")
#             st.pyplot(visualizations['corr_matrix'])
#             plt.close(visualizations['corr_matrix'])
#         if 'kde_plot' in visualizations:
#             st.subheader("Malaria Cases Distribution (2020-2025)")
#             st.pyplot(visualizations['kde_plot'])
#             plt.close(visualizations['kde_plot'])
#         if 'top_states' in visualizations:
#             st.subheader("Top 10 States by Average Malaria Cases (2020-2024)")
#             st.pyplot(visualizations['top_states'])
#             plt.close(visualizations['top_states'])

#     st.header("🔧 Feature Engineering")
#     with st.spinner("Creating features..."):
#         df_malaria, feat_viz = perform_malaria_feature_engineering(df_malaria)
#         st.session_state.df_malaria = df_malaria
#         st.session_state.user_profile['analyses_run'] += 1
#     st.success("Feature engineering completed.")
#     with st.expander("📊 Updated Dataset with Features", expanded=False):
#         st.dataframe(df_malaria.style.set_table_styles([
#             {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#             {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#         ]))
#     st.subheader("Feature Analysis Visualizations")
#     if 'risk_category' in feat_viz:
#         st.pyplot(feat_viz['risk_category'])
#         plt.close(feat_viz['risk_category'])
#     if 'state_risk' in feat_viz:
#         st.pyplot(feat_viz['state_risk'])
#         plt.close(feat_viz['state_risk'])

#     st.header("🤖 ARIMA Time Series Prediction Model")
#     with st.spinner("Running ARIMA model..."):
#         df_malaria, metrics = run_malaria_arima(df_malaria)
#         st.session_state.df_malaria = df_malaria
#         st.session_state.user_profile['analyses_run'] += 1
#     col1, col2, col3 = st.columns(3)
#     col1.metric("MSE", f"{metrics['mse']:.2f}" if not np.isnan(metrics['mse']) else "N/A", help="Mean Squared Error of the model")
#     col2.metric("R² Score", f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A", help="R² Score indicating model fit")
#     col3.metric("Accuracy", f"{metrics['accuracy']:.2f}%" if not np.isnan(metrics['accuracy']) else "N/A", help="Accuracy based on R²")

#     st.header("📊 State-wise Malaria Trend Analysis")
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         selected_state = st.selectbox(
#             "Select State:",
#             df_malaria['STATE_UT'].unique(),
#             index=0,
#             key='malaria_state_selector')
#         st.subheader(f"Data for {selected_state}")
#         state_data = df_malaria[df_malaria['STATE_UT'] == selected_state].iloc[0]
#         display_data = {
#             "Year": [2020, 2021, 2022, 2023, 2024, 2025],
#             "Cases": [
#                 state_data['Malaria_2020'],
#                 state_data['Malaria_2021'],
#                 state_data['Malaria_2022'],
#                 state_data['Malaria_2023'],
#                 state_data['Malaria_2024'],
#                 state_data['Predicted_Malaria_2025']
#             ],
#             "Type": ["Actual", "Actual", "Actual", "Actual", "Actual", "Forecast"]
#         }
#         display_data["Cases"] = [0 if pd.isna(x) else x for x in display_data["Cases"]]
#         st.dataframe(pd.DataFrame(display_data).style.set_table_styles([
#             {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#             {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#         ]))
#     with col2:
#         st.subheader(f"Trend for {selected_state}")
#         fig = plt.figure(figsize=(10, 5))
#         ax = fig.add_subplot(111)
#         years = [2020, 2021, 2022, 2023, 2024, 2025]
#         cases = [
#             state_data['Malaria_2020'],
#             state_data['Malaria_2021'],
#             state_data['Malaria_2022'],
#             state_data['Malaria_2023'],
#             state_data['Malaria_2024'],
#             state_data['Predicted_Malaria_2025']
#         ]
#         cases = [0 if pd.isna(x) else x for x in cases]
#         ax.plot(years[:5], cases[:5], marker='o', linestyle='-', color='#3498db', label='Actual', linewidth=2)
#         ax.plot(years[4:], cases[4:], marker='o', linestyle='--', color='#e74c3c', label='Forecast', linewidth=2)
#         ax.set_title(f"Malaria Cases Trend (2020–2025)", fontsize=14, pad=15)
#         ax.set_xlabel("Year", fontsize=12)
#         ax.set_ylabel("Number of Cases", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend()
#         ax.axvspan(2024.5, 2025.5, color='#f1c40f', alpha=0.1)
#         ax.text(2024.8, max(cases)*0.9, 'Forecast', color='#e74c3c', fontsize=10)
#         st.pyplot(fig)
#         plt.close(fig)

#     st.header("🔮 Top Predictions for 2025")
#     tab1, tab2 = st.tabs(["📊 Top 10 States", "📋 All States"])
#     with tab1:
#         top10 = df_malaria[['STATE_UT', 'Predicted_Malaria_2025']].sort_values(
#             by='Predicted_Malaria_2025', ascending=False).head(10)
#         fig_top, ax_top = plt.subplots(figsize=(10, 5))
#         sns.barplot(data=top10, x='Predicted_Malaria_2025', y='STATE_UT', palette='coolwarm', ax=ax_top)
#         ax_top.set_title("Top 10 Predicted Malaria Cases for 2025", fontsize=14, pad=15)
#         ax_top.set_xlabel("Predicted Cases", fontsize=12)
#         ax_top.set_ylabel("State/UT", fontsize=12)
#         st.pyplot(fig_top)
#         plt.close(fig_top)
#     with tab2:
#         df_display = df_malaria[['STATE_UT', 'Malaria_2020', 'Malaria_2021', 'Malaria_2022',
#                                  'Malaria_2023', 'Malaria_2024', 'Predicted_Malaria_2025']].copy()
#         df_display[['Malaria_2020', 'Malaria_2021', 'Malaria_2022',
#                     'Malaria_2023', 'Malaria_2024', 'Predicted_Malaria_2025']] = df_display[[
#             'Malaria_2020', 'Malaria_2021', 'Malaria_2022',
#             'Malaria_2023', 'Malaria_2024', 'Predicted_Malaria_2025'
#         ]].apply(lambda x: x.fillna(0).astype(int))
#         st.dataframe(df_display.sort_values(by='Predicted_Malaria_2025', ascending=False).style.set_table_styles([
#             {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#             {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#         ]))

#     st.header("📊 Annual Malaria Risk by State")
#     selected_state = st.selectbox(
#         "Select State/UT:",
#         options=sorted(df_malaria["STATE_UT"].unique()),
#         key='malaria_risk_state_selector')
#     available_years = ['2020', '2021', '2022', '2023', '2024', '2025']
#     selected_year = st.selectbox(
#         "Select Year:",
#         options=available_years,
#         key='malaria_risk_year_selector')
#     state_data = df_malaria[df_malaria['STATE_UT'] == selected_state].iloc[0]
#     years = ['2020', '2021', '2022', '2023', '2024', '2025']
#     cases = [state_data[f'Malaria_{year}'] for year in years[:-1]] + [state_data['Predicted_Malaria_2025']]
#     cases = [0 if pd.isna(x) else x for x in cases]
#     risk_data = [classify_risk(c, "malaria") for c in cases]
#     risk_df = pd.DataFrame({
#         'Year': years,
#         'Cases': cases,
#         'Risk Category': [r[0] for r in risk_data],
#         'Color': [r[1] for r in risk_data]
#     })
#     st.subheader(f"Risk Assessment for {selected_state} in {selected_year}")
#     selected_year_data = risk_df[risk_df['Year'] == selected_year].iloc[0]
#     st.write(f"Year: {selected_year_data['Year']}")
#     st.write(f"Number of Cases: {format_number(selected_year_data['Cases'])}")
#     st.markdown(f"Risk Category: <span style='color:{selected_year_data['Color']}'>{selected_year_data['Risk Category']}</span>", unsafe_allow_html=True)

#     st.header("🗺️ Predicted 2025 Malaria Risk by State")
#     df_malaria['Risk_2025'] = df_malaria['Predicted_Malaria_2025'].apply(lambda x: classify_risk(x, "malaria")[0])
#     color_map = {
#         "🔴 High Risk": "#e74c3c",
#         "🟠 Medium Risk": "#f39c12",
#         "🟡 Moderate Risk": "#f1c40f",
#         "🟢 Low Risk": "#2ecc71"
#     }
#     try:
#         fig = px.choropleth(
#             df_malaria,
#             geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
#             featureidkey='properties.ST_NM',
#             locations='STATE_UT',
#             color='Risk_2025',
#             color_discrete_map=color_map,
#             category_orders={"Risk_2025": ["🔴 High Risk", "🟠 Medium Risk", "🟡 Moderate Risk", "🟢 Low Risk"]},
#             title='Predicted Malaria Risk for 2025',
#             hover_data={'Predicted_Malaria_2025': ':,.0f', 'STATE_UT': True},
#             labels={'Risk_2025': 'Risk Category'}
#         )
#         fig.update_geos(
#             fitbounds="locations",
#             visible=False,
#             projection_type="mercator"
#         )
#         fig.update_layout(
#             height=600,
#             margin={"r":0,"t":40,"l":0,"b":0},
#             legend_title_text='Risk Category',
#             title_font_size=20,
#             title_x=0.5
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     except Exception as e:
#         st.error(f"Error rendering choropleth map: {str(e)}")
#     st.caption("""
#         **Malaria Risk Classification:**  
#         🔴 High Risk (>50,000 cases) | 🟠 Medium Risk (25,000-50,000)  
#         🟡 Moderate Risk (10,000-25,000) | 🟢 Low Risk (<10,000)
#     """)

# # Page: Combined Insights
# if page == "Combined Insights":
#     st.markdown("<div class='card'><h2>Combined Insights: COVID-19 and Malaria</h2></div>", unsafe_allow_html=True)
#     if st.session_state.df_covid is None and st.session_state.df_malaria is None:
#         st.warning("Please upload and process datasets for COVID-19 and/or malaria on the Home page to view combined insights.")
#         st.stop()

#     if st.session_state.df_covid is not None and st.session_state.df_malaria is not None:
#         df_covid = st.session_state.df_covid
#         df_malaria = st.session_state.df_malaria

#         df_covid['Year'] = df_covid['Date_reported'].dt.year
#         covid_yearly = df_covid.groupby('Year')['New_cases'].sum().reset_index()
#         covid_yearly['Disease'] = 'COVID-19'

#         malaria_cols = [f'Malaria_{y}' for y in range(2020, 2025)] + ['Predicted_Malaria_2025']
#         years = list(range(2020, 2026))
#         malaria_yearly = pd.DataFrame({
#             'Year': years,
#             'New_cases': [df_malaria[col].sum() if col in df_malaria.columns else 0 for col in malaria_cols]
#         })
#         malaria_yearly['Disease'] = 'Malaria'
#         malaria_yearly['New_cases'] = malaria_yearly['New_cases'].round(0).astype(int)

#         combined_yearly = pd.concat([covid_yearly, malaria_yearly])

#         st.subheader("Annual Case Trends: COVID-19 vs. Malaria")
#         try:
#             fig_combined = px.line(
#                 combined_yearly,
#                 x='Year',
#                 y='New_cases',
#                 color='Disease',
#                 title='Annual Case Trends (2020-2025)',
#                 labels={'New_cases': 'Total Cases', 'Year': 'Year'},
#                 color_discrete_map={'COVID-19': '#3498db', 'Malaria': '#2ecc71'}
#             )
#             malaria_2025 = combined_yearly[(combined_yearly['Disease'] == 'Malaria') & (combined_yearly['Year'] >= 2024)]
#             fig_combined.add_trace(go.Scatter(
#                 x=malaria_2025['Year'],
#                 y=malaria_2025['New_cases'],
#                 mode='lines',
#                 name='Malaria (Forecast)',
#                 line=dict(color='#2ecc71', dash='dash'),
#                 showlegend=False
#             ))
#             fig_combined.update_layout(
#                 height=400,
#                 template="plotly_white",
#                 title_x=0.5,
#                 yaxis=dict(title='Total Cases (Log Scale)', type='log')
#             )
#             st.plotly_chart(fig_combined, use_container_width=True)
#         except Exception as e:
#             st.error(f"Error plotting combined trends: {str(e)}")

#         st.subheader("Risk Comparison for 2024")
#         covid_2024_cases = df_covid[df_covid['Year'] == 2024]['New_cases'].sum()
#         malaria_2024_cases = df_malaria['Malaria_2024'].sum().round(0).astype(int)
#         covid_risk = classify_risk(covid_2024_cases, "covid")[0]
#         malaria_risk = classify_risk(malaria_2024_cases, "malaria")[0]

#         col1, col2 = st.columns(2)
#         col1.metric("COVID-19 Risk (2024)", covid_risk, f"{format_number(covid_2024_cases)} cases")
#         col2.metric("Malaria Risk (2024)", malaria_risk, f"{format_number(malaria_2024_cases)} cases")

#         st.subheader("Model Accuracy Comparison")
#         historical_df = df_covid.set_index('Date_reported')['New_cases']
#         forecast_steps = 30
#         _, mse_cases, r2_cases = lstm_forecast(historical_df[-365:], forecast_steps)
#         covid_accuracy = r2_cases * 100 if not np.isnan(r2_cases) else float('nan')
#         _, malaria_metrics = run_malaria_arima(df_malaria)
#         malaria_accuracy = malaria_metrics['accuracy']
#         st.session_state.user_profile['analyses_run'] += 1

#         col1, col2 = st.columns(2)
#         col1.metric("COVID-19 LSTM Accuracy", f"{covid_accuracy:.2f}%" if not np.isnan(covid_accuracy) else "N/A",
#                     help="Accuracy based on R² score for LSTM model on validation set")
#         col2.metric("Malaria ARIMA Accuracy", f"{malaria_accuracy:.2f}%" if not np.isnan(malaria_accuracy) else "N/A",
#                     help="Accuracy based on R² score for ARIMA model")
#     else:
#         st.info("Please upload both datasets to view combined insights.")


# # Page: Profile
# if page == "Profile":
#     st.markdown("<div class='card'><h2>👤 User Profile</h2></div>", unsafe_allow_html=True)
#     st.subheader("User Information")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write(f"Name: {st.session_state.user_profile['name'] or 'Not set'}")
#         st.write(f"Email: {st.session_state.user_profile['email'] or 'Not set'}")
#     with col2:
#         st.write(f"Analyses Run: {st.session_state.user_profile['analyses_run']}")
#     with st.form("profile_form"):
#         st.subheader("Edit Profile")
#         name = st.text_input("Name", value=st.session_state.user_profile['name'])
#         email = st.text_input("Email", value=st.session_state.user_profile['email'])
#         submitted = st.form_submit_button("💾 Save Profile")
#         if submitted:
#             if email and '@' not in email:
#                 st.error("Please enter a valid email address.")
#             else:
#                 st.session_state.user_profile['name'] = name
#                 st.session_state.user_profile['email'] = email
#                 st.success("Profile updated successfully!")
#                 st.rerun()
#     st.subheader("Analysis History")
#     if st.session_state.user_profile['analyses_run'] > 0:
#         st.write(f"You have run {st.session_state.user_profile['analyses_run']} analyses.")
#         st.info("Detailed analysis history is not yet implemented. Future updates may include logs of past forecasts.")
#     else:
#         st.write("No analyses have been run yet.")
    

# # Page: Settings
# if page == "Settings":
#     st.markdown("<div class='card'><h2>⚙️ Settings</h2></div>", unsafe_allow_html=True)
#     st.subheader("Dashboard Configuration")
    
#     with st.form("settings_form"):
#         st.subheader("Appearance")
#         theme = st.selectbox("Theme", ["Light", "Dark"], index=["Light", "Dark"].index(st.session_state.settings['theme']))
        
#         st.subheader("Forecast Parameters")
#         look_back_period = st.slider(
#             "LSTM Look-Back Period (days)",
#             min_value=30,
#             max_value=180,
#             value=st.session_state.settings['look_back_period'],
#             step=10,
#             help="Number of past days to use for LSTM forecasting. Higher values may capture longer trends but require more data."
#         )
        
#         submitted = st.form_submit_button("💾 Save Settings")
#         if submitted:
#             st.session_state.settings['theme'] = theme
#             st.session_state.settings['look_back_period'] = look_back_period
#             st.success("Settings saved successfully!")
#             apply_theme(theme)
#             st.rerun()

#     st.subheader("Current Settings")
#     st.write(f"**Theme**: {st.session_state.settings['theme']}")
#     st.write(f"**LSTM Look-Back Period**: {st.session_state.settings['look_back_period']} days")

# # Sidebar Reset Button
# if st.sidebar.button("🔄 Reset App"):
#     st.session_state.df_covid = None
#     st.session_state.df_malaria = None
#     st.session_state.user_profile = {'name': '', 'email': '', 'analyses_run': 0}
#     st.session_state.settings = {'theme': 'Light', 'look_back_period': 60}
#     st.cache_data.clear()
#     st.rerun()
















# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import warnings
# from prophet import Prophet

# warnings.filterwarnings("ignore")

# # Custom CSS for styling with theme support
# def apply_theme(theme):
#     if theme == "Dark":
#         css = """
#             <style>
#             .main { background-color: #2c3e50; color: #ecf0f1; font-family: 'Arial', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(255,255,255,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(255,255,255,0.1); }
#             .stMetric { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(255,255,255,0.1); margin: 10px 0; }
#             .card { background-color: #34495e; color: #ecf0f1; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(255,255,255,0.1); }
#             h1, h2, h3, h4 { color: #ecf0f1; }
#             .sidebar .sidebar-content { background-color: #2c3e50; color: #ecf0f1; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: #34495e; }
#             </style>
#         """
#     else:  # Light theme
#         css = """
#             <style>
#             .main { background-color: #f5f7fa; font-family: 'Arial', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: white; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#             .stMetric { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 10px 0; }
#             .card { background-color: white; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
#             h1, h2, h3, h4 { color: #2c3e50; }
#             .sidebar .sidebar-content { background-color: #34495e; color: white; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: white; }
#             </style>
#         """
#     st.markdown(css, unsafe_allow_html=True)

# # Initialize session state
# if 'df_malaria' not in st.session_state:
#     st.session_state.df_malaria = None
# if 'df_weather' not in st.session_state:
#     st.session_state.df_weather = None
# if 'user_profile' not in st.session_state:
#     st.session_state.user_profile = {'name': '', 'email': '', 'analyses_run': 0}
# if 'settings' not in st.session_state:
#     st.session_state.settings = {'theme': 'Light', 'look_back_period': 60}

# # Apply theme
# apply_theme(st.session_state.settings['theme'])

# # Sidebar navigation
# st.sidebar.title("🦠 Disease Prediction Dashboard")
# st.sidebar.markdown("Analyze and predict malaria cases in India.")
# page = st.sidebar.radio("Go to:", ["Malaria Analysis"])

# # Header
# st.markdown("""
#     <div style="text-align: center; padding: 20px; background-color: #34495e; color: white; border-radius: 10px;">
#         <h1>🦠 Malaria Prediction Dashboard</h1>
#         <p style="font-size: 18px;">Analyze and predict malaria cases across India using Prophet models.</p>
#     </div>
# """, unsafe_allow_html=True)

# # Function to clean weather data
# @st.cache_data
# def clean_weather_data(df):
#     """Clean and preprocess weather dataset"""
#     df_clean = df.copy()
#     df_clean['DATE'] = pd.to_datetime(df_clean['DATE'])
#     df_clean = df_clean.sort_values('DATE').reset_index(drop=True)
#     expected_cols = ['temp', 'humidity', 'precip', 'sealevelpressure']
#     missing_cols = [col for col in expected_cols if col not in df_clean.columns]
#     if missing_cols:
#         st.error(f"Error: Missing expected columns in weather data: {missing_cols}. Available columns: {df_clean.columns.tolist()}")
#         return None
#     numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
#     df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
#     return df_clean

# # Function to create daily malaria data
# @st.cache_data
# def create_daily_malaria_data(weather_df, end_date='2025-05-24'):
#     """Create daily malaria estimates using weather correlation, extending to end_date"""
#     yearly_malaria = {
#         2020: 50000,
#         2021: 52000,
#         2022: 48000,
#         2023: 51000,
#         2024: 49000,
#         2025: 49500
#     }
#     daily_data = []
#     max_weather_date = weather_df['DATE'].max()
#     end_date = pd.to_datetime(end_date)
#     for _, weather_row in weather_df.iterrows():
#         date = weather_row['DATE']
#         year = date.year
#         if year in yearly_malaria:
#             base_rate = yearly_malaria[year] / 365
#             temp_factor = 1 + (weather_row['temp'] - 25) * 0.02
#             humidity_factor = 1 + (weather_row['humidity'] - 70) * 0.01
#             precip_factor = 1 + weather_row['precip'] * 0.1
#             month = date.month
#             seasonal_factor = 1.5 if month in [6, 7, 8, 9] else 1.0
#             daily_estimate = base_rate * temp_factor * humidity_factor * precip_factor * seasonal_factor
#             daily_estimate = max(0, daily_estimate) + 1
#             daily_data.append({
#                 'DATE': date,
#                 'Year': year,
#                 'Month': date.month,
#                 'Day': date.day,
#                 'Estimated_Daily_Cases': daily_estimate,
#                 'Temperature': weather_row['temp'],
#                 'Humidity': weather_row['humidity'],
#                 'Precipitation': weather_row['precip'],
#                 'Pressure': weather_row['sealevelpressure']
#             })
#     if max_weather_date < end_date:
#         last_weather = weather_df[weather_df['DATE'] == max_weather_date].iloc[0]
#         current_date = max_weather_date + timedelta(days=1)
#         while current_date <= end_date:
#             year = current_date.year
#             if year in yearly_malaria:
#                 base_rate = yearly_malaria[year] / 365
#                 temp_factor = 1 + (last_weather['temp'] - 25) * 0.02
#                 humidity_factor = 1 + (last_weather['humidity'] - 70) * 0.01
#                 precip_factor = 1 + last_weather['precip'] * 0.1
#                 month = current_date.month
#                 seasonal_factor = 1.5 if month in [6, 7, 8, 9] else 1.0
#                 daily_estimate = base_rate * temp_factor * humidity_factor * precip_factor * seasonal_factor
#                 daily_estimate = max(0, daily_estimate) + 1
#                 daily_data.append({
#                     'DATE': current_date,
#                     'Year': year,
#                     'Month': month,
#                     'Day': current_date.day,
#                     'Estimated_Daily_Cases': daily_estimate,
#                     'Temperature': last_weather['temp'],
#                     'Humidity': last_weather['humidity'],
#                     'Precipitation': last_weather['precip'],
#                     'Pressure': last_weather['sealevelpressure']
#                 })
#             current_date += timedelta(days=1)
#     df = pd.DataFrame(daily_data)
#     return df

# # Function to evaluate model
# def evaluate_model(actual, predicted):
#     """Calculate performance metrics including MAPE"""
#     mae = np.mean(np.abs(actual - predicted))
#     mse = np.mean((actual - predicted) ** 2)
#     rmse = np.sqrt(mse)
#     epsilon = 1e-10
#     mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
#     accuracy = max(0, min(100, 100 - mape))
#     return mae, mse, rmse, mape, accuracy

# # Preload datasets
# def preload_datasets():
#     dates = pd.date_range(start='2020-01-01', end='2025-05-24', freq='D')
#     weather_df = pd.DataFrame({
#         'DATE': dates,
#         'temp': np.random.normal(25, 5, len(dates)),
#         'humidity': np.random.normal(70, 10, len(dates)),
#         'precip': np.random.exponential(2, len(dates)),
#         'sealevelpressure': np.random.normal(1013, 5, len(dates))
#     })
#     st.session_state.df_weather = weather_df
#     df_malaria = create_daily_malaria_data(weather_df, end_date='2025-05-24')
#     st.session_state.df_malaria = df_malaria

# preload_datasets()

# # Page: Malaria Analysis
# if page == "Malaria Analysis":
#     if st.session_state.df_malaria is None or st.session_state.df_weather is None:
#         st.warning("Malaria or weather data failed to load. Please reset the app and try again.")
#         st.stop()

#     st.markdown("<div class='card'><h2>🦟 Malaria Disease Outbreak Prediction</h2></div>", unsafe_allow_html=True)
#     st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Time Series Forecasting with Weather Integration</p>', unsafe_allow_html=True)
    
#     weather_clean = clean_weather_data(st.session_state.df_weather)
#     if weather_clean is None:
#         st.error("Failed to process weather data.")
#         st.stop()
    
#     daily_malaria = st.session_state.df_malaria
    
#     st.success("✅ Weather and malaria datasets loaded successfully!")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Weather Records", len(weather_clean))
#     with col2:
#         st.metric("Daily Estimates", len(daily_malaria))
    
#     tabs = st.tabs(["📊 Data Overview", "🔍 Exploratory Analysis", "📈 Time Series Forecasting", "🌤️ Weather Correlation", "📉 Model Evaluation"])
    
#     with tabs[0]:
#         st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Dataset Overview</h2>', unsafe_allow_html=True)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Weather Dataset (First 10 rows)")
#             st.dataframe(weather_clean.head(10))
            
#             st.subheader("Weather Trends Over Time")
#             fig = make_subplots(
#                 rows=2, cols=2,
#                 subplot_titles=('Temperature', 'Humidity', "Precipitation", 'Pressure')
#             )
#             fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['temp'], 
#                                    name='Temperature'), row=1, col=1)
#             fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['humidity'], 
#                                    name='Humidity'), row=1, col=2)
#             fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['precip'], 
#                                    name='Precipitation'), row=2, col=1)
#             fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['sealevelpressure'], 
#                                    name='Pressure'), row=2, col=2)
#             fig.update_layout(height=600, showlegend=False)
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             st.subheader("Daily Malaria Estimates (First 10 rows)")
#             st.dataframe(daily_malaria.head(10))
            
#             st.subheader("Daily Malaria Trends")
#             fig = px.line(daily_malaria, x='DATE', y='Estimated_Daily_Cases', 
#                          title="Estimated Daily Malaria Cases (2020-2025)")
#             st.plotly_chart(fig, use_container_width=True)
    
#     with tabs[1]:
#         st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
#         st.subheader("Correlation Matrix")
#         corr_data = daily_malaria[['Estimated_Daily_Cases', 'Temperature', 'Humidity', 
#                                  'Precipitation', 'Pressure']].corr()
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig, ax = plt.subplots(figsize=(12, 8))
#         sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#         ax.set_title("Correlation Matrix")
#         st.pyplot(fig)
#         plt.close(fig)
        
#         st.subheader("Distribution of Daily Malaria Cases")
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig, ax = plt.subplots(figsize=(12, 6))
#         sns.kdeplot(daily_malaria['Estimated_Daily_Cases'], fill=True, alpha=0.5, ax=ax)
#         ax.set_xlabel("Estimated Daily Cases")
#         ax.set_ylabel("Density")
#         ax.set_title("Distribution of Daily Malaria Cases (2020-2025)")
#         st.pyplot(fig)
#         plt.close(fig)
    
#     with tabs[2]:
#         st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Time Series Forecasting</h2>', unsafe_allow_html=True)
        
#         prophet_df = daily_malaria[['DATE', 'Estimated_Daily_Cases', 'Temperature', 'Humidity', 'Precipitation']].rename(
#             columns={'DATE': 'ds', 'Estimated_Daily_Cases': 'y'}
#         )
        
#         model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
#         model.add_regressor('Temperature')
#         model.add_regressor('Humidity')
#         model.add_regressor('Precipitation')
#         model.fit(prophet_df)
        
#         with st.spinner("Generating forecast for 2025..."):
#             future_dates = pd.date_range(start='2025-05-25', end='2025-12-31', freq='D')
#             future_df = pd.DataFrame({'ds': future_dates})
#             historical_end = daily_malaria['DATE'].max()
#             last_30_days_start = historical_end - timedelta(days=29)
#             last_30_days = daily_malaria[(daily_malaria['DATE'] >= last_30_days_start) & 
#                                        (daily_malaria['DATE'] <= historical_end)][['Temperature', 'Humidity', 'Precipitation']].mean()
#             future_weather = pd.DataFrame({
#                 'ds': future_dates,
#                 'Temperature': last_30_days['Temperature'],
#                 'Humidity': last_30_days['Humidity'],
#                 'Precipitation': last_30_days['Precipitation']
#             })
#             future_df = future_df.merge(future_weather, on='ds', how='left')
#             forecast = model.predict(future_df)
#             forecast_2025 = forecast[['ds', 'yhat']].copy()
#             forecast_2025['yhat'] = forecast_2025['yhat'].clip(lower=0).round()
#             forecast_2025.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Daily_Cases'}, inplace=True)
            
#             st.subheader("Malaria Cases Forecast for 2025 (May 25 to Dec 31)")
#             fig = go.Figure()
#             historical_start = historical_end - timedelta(days=89)
#             historical = prophet_df[(prophet_df['ds'] >= historical_start) & (prophet_df['ds'] <= historical_end)]
#             fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'],
#                                    mode='lines', name=f'Historical (Last 90 days ending {historical_end.date()})',
#                                    line=dict(color='blue')))
#             fig.add_trace(go.Scatter(x=forecast_2025['Date'], y=forecast_2025['Predicted_Daily_Cases'],
#                                    mode='lines', name='Forecast for 2025',
#                                    line=dict(color='red', dash='dash')))
#             fig.update_layout(title="Daily Malaria Cases Forecast for 2025 (May 25 to Dec 31)",
#                             xaxis_title="Date", yaxis_title="Daily Cases")
#             st.plotly_chart(fig, use_container_width=True)
            
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Avg Daily Cases (May 25-Dec 31)", f"{forecast_2025['Predicted_Daily_Cases'].mean():.1f}")
#             with col2:
#                 st.metric("Total Cases (May 25-Dec 31)", f"{forecast_2025['Predicted_Daily_Cases'].sum():.0f}")
#             with col3:
#                 st.metric("Max Daily Cases", f"{forecast_2025['Predicted_Daily_Cases'].max():.1f}")
#             with col4:
#                 st.metric("Min Daily Cases", f"{forecast_2025['Predicted_Daily_Cases'].min():.1f}")
            
#             csv = forecast_2025.to_csv(index=False)
#             st.download_button(
#                 label="Download 2025 Forecast as CSV",
#                 data=csv,
#                 file_name="malaria_forecast_2025_may25_dec31.csv",
#                 mime="text/csv",
#             )
    
#     with tabs[3]:
#         st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Weather Correlation Analysis</h2>', unsafe_allow_html=True)
        
#         corr_data = daily_malaria[['Estimated_Daily_Cases', 'Temperature', 'Humidity', 
#                                  'Precipitation', 'Pressure']].corr()
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr_data, annot=True, fmt=".3f", cmap="RdYlBu_r", ax=ax)
#         ax.set_title("Weather Variables Correlation with Malaria Cases")
#         st.pyplot(fig)
#         plt.close(fig)
        
#         st.subheader("Weather vs Malaria Cases Relationships")
#         col1, col2 = st.columns(2)
#         with col1:
#             fig = px.scatter(daily_malaria, x='Temperature', y='Estimated_Daily_Cases',
#                            color='Humidity', title="Temperature vs Malaria Cases")
#             st.plotly_chart(fig, use_container_width=True)
#         with col2:
#             fig = px.scatter(daily_malaria, x='Humidity', y='Estimated_Daily_Cases',
#                            color='Precipitation', title="Humidity vs Malaria Cases")
#             st.plotly_chart(fig, use_container_width=True)
        
#         st.subheader("Seasonal Patterns")
#         monthly_avg = daily_malaria.groupby('Month')['Estimated_Daily_Cases'].mean()
#         fig = px.bar(x=monthly_avg.index, y=monthly_avg.values,
#                     title="Average Daily Malaria Cases by Month")
#         fig.update_xaxes(title="Month")
#         fig.update_yaxes(title="Average Daily Cases")
#         st.plotly_chart(fig, use_container_width=True)
    
#     with tabs[4]:
#         st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Model Evaluation</h2>', unsafe_allow_html=True)
        
#         prophet_df = daily_malaria[['DATE', 'Estimated_Daily_Cases', 'Temperature', 'Humidity', 'Precipitation']].rename(
#             columns={'DATE': 'ds', 'Estimated_Daily_Cases': 'y'}
#         )
#         train_data = prophet_df[prophet_df['ds'] < '2025-01-01']
#         test_data = prophet_df[prophet_df['ds'] >= '2025-01-01']
        
#         with st.spinner("Evaluating model performance..."):
#             model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
#             model.add_regressor('Temperature')
#             model.add_regressor('Humidity')
#             model.add_regressor('Precipitation')
#             model.fit(train_data)
#             future_dates = pd.date_range(start=train_data['ds'].max() + timedelta(days=1), 
#                                        end=test_data['ds'].max(), freq='D')
#             future_df = pd.DataFrame({'ds': future_dates})
#             test_weather = test_data[['ds', 'Temperature', 'Humidity', 'Precipitation']]
#             future_df = future_df.merge(test_weather, on='ds', how='left')
#             forecast = model.predict(future_df)
#             predictions = forecast['yhat'].clip(lower=0).round()
#             mae, mse, rmse, mape, accuracy = evaluate_model(test_data['y'], predictions)
            
#             st.subheader("Performance Metrics")
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
#             with col2:
#                 st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
#             with col3:
#                 st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
#             with col4:
#                 st.metric("Accuracy (100 - MAPE)", f"{accuracy:.2f}%")
            
#             st.markdown(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
            
#             st.subheader("Actual vs Predicted (2025)")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['y'],
#                                    mode='lines', name='Actual',
#                                    line=dict(color='blue')))
#             fig.add_trace(go.Scatter(x=future_df['ds'], y=predictions,
#                                    mode='lines', name='Predicted',
#                                    line=dict(color='red', dash='dash')))
#             fig.update_layout(title="Actual vs Predicted Daily Malaria Cases (2025)",
#                             xaxis_title="Date", yaxis_title="Daily Cases")
#             st.plotly_chart(fig, use_container_width=True)

# # Sidebar Reset Button
# if st.sidebar.button("🔄 Reset App"):
#     st.session_state.df_malaria = None
#     st.session_state.df_weather = None
#     st.session_state.user_profile = {'name': '', 'email': '', 'analyses_run': 0}
#     st.session_state.settings = {'theme': 'Light', 'look_back_period': 60}
#     st.cache_data.clear()
#     st.rerun()










# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import warnings
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from prophet import Prophet

# warnings.filterwarnings("ignore")

# # Custom CSS for styling with theme support
# def apply_theme(theme):
#     if theme == "Dark":
#         css = """
#             <style>
#             .main { background-color: #2c3e50; color: #ecf0f1; font-family: 'Arial', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(255,255,255,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(255,255,255,0.1); }
#             .stMetric { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(255,255,255,0.1); margin: 10px 0; }
#             .card { background-color: #34495e; color: #ecf0f1; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(255,255,255,0.1); }
#             h1, h2, h3, h4 { color: #ecf0f1; }
#             .sidebar .sidebar-content { background-color: #2c3e50; color: #ecf0f1; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: #34495e; }
#             </style>
#         """
#     else:  # Light theme
#         css = """
#             <style>
#             .main { background-color: #f5f7fa; font-family: 'Arial', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: white; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#             .stMetric { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 10px 0; }
#             .card { background-color: white; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
#             h1, h2, h3, h4 { color: #2c3e50; }
#             .sidebar .sidebar-content { background-color: #34495e; color: white; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: white; }
#             </style>
#         """
#     st.markdown(css, unsafe_allow_html=True)

# # Initialize session state
# if 'df_malaria' not in st.session_state:
#     st.session_state.df_malaria = None
# if 'df_weather' not in st.session_state:
#     st.session_state.df_weather = None
# if 'theme' not in st.session_state:
#     st.session_state.theme = 'Light'

# # Apply theme
# apply_theme(st.session_state.theme)

# # Sidebar navigation
# st.sidebar.title("🦠 Disease Prediction Dashboard")
# st.sidebar.markdown("Analyze and predict malaria cases in India.")

# # Header
# st.markdown("""
#     <div style="text-align: center; padding: 20px; background-color: #34495e; color: white; border-radius: 10px;">
#         <h1>🦠 Disease Prediction Dashboard</h1>
#         <p style="font-size: 18px;">Analyze and predict malaria cases across India using Prophet models.</p>
#     </div>
# """, unsafe_allow_html=True)

# # Preload Datasets
# def preload_datasets():
#     # Preload Weather Data (Simulated)
#     dates = pd.date_range(start='2020-01-01', end='2025-05-24', freq='D')
#     weather_df = pd.DataFrame({
#         'DATE': dates,
#         'temp': np.random.normal(25, 5, len(dates)),  # Simulated temperature
#         'humidity': np.random.normal(70, 10, len(dates)),  # Simulated humidity
#         'precip': np.random.exponential(2, len(dates)),  # Simulated precipitation
#         'sealevelpressure': np.random.normal(1013, 5, len(dates))  # Simulated pressure
#     })
#     st.session_state.df_weather = weather_df

#     # Preload Malaria Data (Simulated Daily Estimates)
#     df_malaria = create_daily_malaria_data(weather_df, end_date='2025-05-24')
#     st.session_state.df_malaria = df_malaria

# # Function to clean weather data
# @st.cache_data
# def clean_weather_data(df):
#     """Clean and preprocess weather dataset"""
#     df_clean = df.copy()
    
#     # Convert DATE column to datetime
#     df_clean['DATE'] = pd.to_datetime(df_clean['DATE'])
#     df_clean = df_clean.sort_values('DATE').reset_index(drop=True)
    
#     # Check for expected columns
#     expected_cols = ['temp', 'humidity', 'precip', 'sealevelpressure']
#     missing_cols = [col for col in expected_cols if col not in df_clean.columns]
#     if missing_cols:
#         st.error(f"Error: Missing expected columns in weather data: {missing_cols}. Available columns: {df_clean.columns.tolist()}")
#         return None
    
#     # Fill missing values with column means
#     numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
#     df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
#     return df_clean

# @st.cache_data
# def create_daily_malaria_data(weather_df, end_date='2025-05-24'):
#     """Create daily malaria estimates using weather correlation, extending to end_date"""
#     # Simulate yearly malaria totals
#     yearly_malaria = {
#         2020: 50000,
#         2021: 52000,
#         2022: 48000,
#         2023: 51000,
#         2024: 49000,
#         2025: 49500
#     }
    
#     # Create daily estimates based on weather patterns
#     daily_data = []
#     max_weather_date = weather_df['DATE'].max()
#     end_date = pd.to_datetime(end_date)
    
#     # Process available weather data
#     for _, weather_row in weather_df.iterrows():
#         date = weather_row['DATE']
#         year = date.year
        
#         if year in yearly_malaria:
#             # Base daily rate (yearly total / 365)
#             base_rate = yearly_malaria[year] / 365
            
#             # Weather impact factors
#             temp_factor = 1 + (weather_row['temp'] - 25) * 0.02
#             humidity_factor = 1 + (weather_row['humidity'] - 70) * 0.01
#             precip_factor = 1 + weather_row['precip'] * 0.1
            
#             # Seasonal factor (higher in monsoon months)
#             month = date.month
#             seasonal_factor = 1.5 if month in [6, 7, 8, 9] else 1.0
            
#             # Calculate daily estimate
#             daily_estimate = base_rate * temp_factor * humidity_factor * precip_factor * seasonal_factor
#             daily_estimate = max(0, daily_estimate) + 1
            
#             daily_data.append({
#                 'DATE': date,
#                 'Year': year,
#                 'Month': date.month,
#                 'Day': date.day,
#                 'Estimated_Daily_Cases': daily_estimate,
#                 'Temperature': weather_row['temp'],
#                 'Humidity': weather_row['humidity'],
#                 'Precipitation': weather_row['precip'],
#                 'Pressure': weather_row['sealevelpressure']
#             })
    
#     # If weather data doesn't extend to end_date, simulate additional data
#     if max_weather_date < end_date:
#         last_weather = weather_df[weather_df['DATE'] == max_weather_date].iloc[0]
#         current_date = max_weather_date + timedelta(days=1)
        
#         while current_date <= end_date:
#             year = current_date.year
#             if year in yearly_malaria:
#                 base_rate = yearly_malaria[year] / 365
#                 temp_factor = 1 + (last_weather['temp'] - 25) * 0.02
#                 humidity_factor = 1 + (last_weather['humidity'] - 70) * 0.01
#                 precip_factor = 1 + last_weather['precip'] * 0.1
#                 month = current_date.month
#                 seasonal_factor = 1.5 if month in [6, 7, 8, 9] else 1.0
#                 daily_estimate = base_rate * temp_factor * humidity_factor * precip_factor * seasonal_factor
#                 daily_estimate = max(0, daily_estimate) + 1
                
#                 daily_data.append({
#                     'DATE': current_date,
#                     'Year': year,
#                     'Month': month,
#                     'Day': current_date.day,
#                     'Estimated_Daily_Cases': daily_estimate,
#                     'Temperature': last_weather['temp'],
#                     'Humidity': last_weather['humidity'],
#                     'Precipitation': last_weather['precip'],
#                     'Pressure': last_weather['sealevelpressure']
#                 })
#             current_date += timedelta(days=1)
    
#     df = pd.DataFrame(daily_data)
#     return df

# def evaluate_model(actual, predicted):
#     """Calculate performance metrics including MAPE"""
#     mae = mean_absolute_error(actual, predicted)
#     mse = mean_squared_error(actual, predicted)
#     rmse = np.sqrt(mse)
#     epsilon = 1e-10
#     mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
#     accuracy = max(0, min(100, 100 - mape))
#     return mae, mse, rmse, mape, accuracy

# # Preload datasets at startup
# preload_datasets()

# # Malaria Analysis Page
# st.markdown("<div class='card'><h2>🦟 Malaria Disease Outbreak Prediction</h2></div>", unsafe_allow_html=True)
# st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Time Series Forecasting with Weather Integration</p>', unsafe_allow_html=True)

# weather_clean = clean_weather_data(st.session_state.df_weather)
# if weather_clean is None:
#     st.error("Failed to process weather data.")
#     st.stop()

# daily_malaria = st.session_state.df_malaria

# st.success("✅ Weather and malaria datasets loaded successfully!")

# # Display dataset info
# col1, col2 = st.columns(2)
# with col1:
#     st.metric("Weather Records", len(weather_clean))
# with col2:
#     st.metric("Daily Estimates", len(daily_malaria))

# # Tabs for different analyses
# tabs = st.tabs(["📊 Data Overview", "🔍 Exploratory Analysis", "📈 Time Series Forecasting", "🌤️ Weather Correlation", "📉 Model Evaluation"])

# with tabs[0]:
#     st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Dataset Overview</h2>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Weather Dataset (First 10 rows)")
#         st.dataframe(weather_clean.head(10))
        
#         # Weather trends
#         st.subheader("Weather Trends Over Time")
#         fig = make_subplots(
#             rows=2, cols=2,
#             subplot_titles=('Temperature', 'Humidity', "Precipitation", 'Pressure')
#         )
        
#         fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['temp'], 
#                                name='Temperature'), row=1, col=1)
#         fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['humidity'], 
#                                name='Humidity'), row=1, col=2)
#         fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['precip'], 
#                                name='Precipitation'), row=2, col=1)
#         fig.add_trace(go.Scatter(x=weather_clean['DATE'], y=weather_clean['sealevelpressure'], 
#                                name='Pressure'), row=2, col=2)
        
#         fig.update_layout(height=600, showlegend=False)
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         st.subheader("Daily Malaria Estimates (First 10 rows)")
#         st.dataframe(daily_malaria.head(10))
        
#         st.subheader("Daily Malaria Trends")
#         fig = px.line(daily_malaria, x='DATE', y='Estimated_Daily_Cases', 
#                      title="Estimated Daily Malaria Cases (2020-2025)")
#         st.plotly_chart(fig, use_container_width=True)

# with tabs[1]:
#     st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
#     # Correlation Matrix
#     st.subheader("Correlation Matrix")
#     corr_data = daily_malaria[['Estimated_Daily_Cases', 'Temperature', 'Humidity', 
#                              'Precipitation', 'Pressure']].corr()
    
#     fig, ax = plt.subplots(figsize=(12, 8))
#     sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     ax.set_title("Correlation Matrix")
#     st.pyplot(fig)
    
#     # KDE Plot
#     st.subheader("Distribution of Daily Malaria Cases")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.kdeplot(daily_malaria['Estimated_Daily_Cases'], fill=True, alpha=0.5, ax=ax)
#     ax.set_xlabel("Estimated Daily Cases")
#     ax.set_ylabel("Density")
#     ax.set_title("Distribution of Daily Malaria Cases (2020-2025)")
#     st.pyplot(fig)

# with tabs[2]:
#     st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Time Series Forecasting</h2>', unsafe_allow_html=True)
    
#     # Prepare data for Prophet
#     prophet_df = daily_malaria[['DATE', 'Estimated_Daily_Cases', 'Temperature', 'Humidity', 'Precipitation']].rename(
#         columns={'DATE': 'ds', 'Estimated_Daily_Cases': 'y'}
#     )
    
#     # Train Prophet model on historical data (2020 to May 24, 2025)
#     model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
#     model.add_regressor('Temperature')
#     model.add_regressor('Humidity')
#     model.add_regressor('Precipitation')
#     model.fit(prophet_df)
    
#     # Forecast for the rest of 2025 (May 25 to December 31, 2025)
#     with st.spinner("Generating forecast for 2025..."):
#         # Create future dataframe for the rest of 2025
#         future_dates = pd.date_range(start='2025-05-25', end='2025-12-31', freq='D')
#         future_df = pd.DataFrame({'ds': future_dates})
        
#         # Use the last 30 days of available data for weather estimation
#         historical_end = daily_malaria['DATE'].max()
#         last_30_days_start = historical_end - timedelta(days=29)
#         last_30_days = daily_malaria[(daily_malaria['DATE'] >= last_30_days_start) & 
#                                    (daily_malaria['DATE'] <= historical_end)][['Temperature', 'Humidity', 'Precipitation']].mean()
        
#         future_weather = pd.DataFrame({
#             'ds': future_dates,
#             'Temperature': last_30_days['Temperature'],
#             'Humidity': last_30_days['Humidity'],
#             'Precipitation': last_30_days['Precipitation']
#         })
#         future_df = future_df.merge(future_weather, on='ds', how='left')
        
#         # Make predictions for 2025
#         forecast = model.predict(future_df)
#         forecast_2025 = forecast[['ds', 'yhat']].copy()
#         forecast_2025['yhat'] = forecast_2025['yhat'].clip(lower=0).round()
#         forecast_2025.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Daily_Cases'}, inplace=True)
        
#         # Plot the forecast
#         st.subheader("Malaria Cases Forecast for 2025 (May 25 to Dec 31)")
#         fig = go.Figure()
        
#         # Historical data (last 90 days of available data)
#         historical_start = historical_end - timedelta(days=89)
#         historical = prophet_df[(prophet_df['ds'] >= historical_start) & (prophet_df['ds'] <= historical_end)]
#         fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'],
#                                mode='lines', name=f'Historical (Last 90 days ending {historical_end.date()})',
#                                line=dict(color='blue')))
        
#         # Forecast for the rest of 2025
#         fig.add_trace(go.Scatter(x=forecast_2025['Date'], y=forecast_2025['Predicted_Daily_Cases'],
#                                mode='lines', name='Forecast for 2025',
#                                line=dict(color='red', dash='dash')))
        
#         fig.update_layout(title="Daily Malaria Cases Forecast for 2025 (May 25 to Dec 31)",
#                         xaxis_title="Date", yaxis_title="Daily Cases")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Summary statistics for 2025 forecast period
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Avg Daily Cases (May 25-Dec 31)", f"{forecast_2025['Predicted_Daily_Cases'].mean():.1f}")
#         with col2:
#             st.metric("Total Cases (May 25-Dec 31)", f"{forecast_2025['Predicted_Daily_Cases'].sum():.0f}")
#         with col3:
#             st.metric("Max Daily Cases", f"{forecast_2025['Predicted_Daily_Cases'].max():.1f}")
#         with col4:
#             st.metric("Min Daily Cases", f"{forecast_2025['Predicted_Daily_Cases'].min():.1f}")
        
#         # Provide downloadable CSV
#         csv = forecast_2025.to_csv(index=False)
#         st.download_button(
#             label="Download 2025 Forecast as CSV",
#             data=csv,
#             file_name="malaria_forecast_2025_may25_dec31.csv",
#             mime="text/csv",
#         )

# with tabs[3]:
#     st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Weather Correlation Analysis</h2>', unsafe_allow_html=True)
    
#     # Weather correlation with malaria estimates
#     corr_data = daily_malaria[['Estimated_Daily_Cases', 'Temperature', 'Humidity', 
#                              'Precipitation', 'Pressure']].corr()
    
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr_data, annot=True, fmt=".3f", cmap="RdYlBu_r", ax=ax)
#     ax.set_title("Weather Variables Correlation with Malaria Cases")
#     st.pyplot(fig)
    
#     # Scatter plots
#     st.subheader("Weather vs Malaria Cases Relationships")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         fig = px.scatter(daily_malaria, x='Temperature', y='Estimated_Daily_Cases',
#                        color='Humidity', title="Temperature vs Malaria Cases")
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         fig = px.scatter(daily_malaria, x='Humidity', y='Estimated_Daily_Cases',
#                        color='Precipitation', title="Humidity vs Malaria Cases")
#         st.plotly_chart(fig, use_container_width=True)
    
#     # Seasonal analysis
#     st.subheader("Seasonal Patterns")
#     monthly_avg = daily_malaria.groupby('Month')['Estimated_Daily_Cases'].mean()
    
#     fig = px.bar(x=monthly_avg.index, y=monthly_avg.values,
#                 title="Average Daily Malaria Cases by Month")
#     fig.update_xaxes(title="Month")
#     fig.update_yaxes(title="Average Daily Cases")
#     st.plotly_chart(fig, use_container_width=True)

# with tabs[4]:
#     st.markdown('<h2 style="font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;">Model Evaluation</h2>', unsafe_allow_html=True)
    
#     # Split data into training (2020-2024) and test (2025)
#     prophet_df = daily_malaria[['DATE', 'Estimated_Daily_Cases', 'Temperature', 'Humidity', 'Precipitation']].rename(
#         columns={'DATE': 'ds', 'Estimated_Daily_Cases': 'y'}
#     )
#     train_data = prophet_df[prophet_df['ds'] < '2025-01-01']
#     test_data = prophet_df[prophet_df['ds'] >= '2025-01-01']
    
#     # Train Prophet model on training data
#     with st.spinner("Evaluating model performance..."):
#         model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
#         model.add_regressor('Temperature')
#         model.add_regressor('Humidity')
#         model.add_regressor('Precipitation')
#         model.fit(train_data)
        
#         # Create future dataframe for test period
#         future_dates = pd.date_range(start=train_data['ds'].max() + timedelta(days=1), 
#                                    end=test_data['ds'].max(), freq='D')
#         future_df = pd.DataFrame({'ds': future_dates})
        
#         # Use actual weather data for the test period
#         test_weather = test_data[['ds', 'Temperature', 'Humidity', 'Precipitation']]
#         future_df = future_df.merge(test_weather, on='ds', how='left')
        
#         # Make predictions
#         forecast = model.predict(future_df)
#         predictions = forecast['yhat'].clip(lower=0).round()
        
#         # Calculate performance metrics
#         mae, mse, rmse, mape, accuracy = evaluate_model(test_data['y'], predictions)
        
#         # Display metrics
#         st.subheader("Performance Metrics")
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
#         with col2:
#             st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
#         with col3:
#             st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
#         with col4:
#             st.metric("Accuracy (100 - MAPE)", f"{accuracy:.2f}%")
        
#         # Display MAPE separately
#         st.markdown(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
        
#         # Plot actual vs predicted
#         st.subheader("Actual vs Predicted (2025)")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['y'],
#                                mode='lines', name='Actual',
#                                line=dict(color='blue')))
#         fig.add_trace(go.Scatter(x=future_df['ds'], y=predictions,
#                                mode='lines', name='Predicted',
#                                line=dict(color='red', dash='dash')))
#         fig.update_layout(title="Actual vs Predicted Daily Malaria Cases (2025)",
#                         xaxis_title="Date", yaxis_title="Daily Cases")
#         st.plotly_chart(fig, use_container_width=True)

# # Sidebar Reset Button
# if st.sidebar.button("🔄 Reset App"):
#     st.session_state.df_malaria = None
#     st.session_state.df_weather = None
#     st.session_state.theme = 'Light'
#     st.cache_data.clear()
#     st.rerun()
# train_model.py
# train_model.py








# app.py
# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np

# # Page config
# st.set_page_config(page_title="COVID-19 Outbreak Prediction", layout="wide")

# # Title
# st.title("COVID-19 Outbreak Prediction with Weather Data (India)")

# # Load pretrained model and scaler
# try:
#     model = joblib.load('rf_model.pkl')
#     scaler = joblib.load('scaler.pkl')
#     st.success("Pretrained model and scaler loaded successfully!")
# except FileNotFoundError:
#     st.error("Pretrained model or scaler not found. Please run train_model.py first.")
#     st.stop()

# # Sidebar for navigation
# st.sidebar.header("Options")
# option = st.sidebar.selectbox("Choose Action", ["Predict Outbreak", "Retrain Model"])

# if option == "Predict Outbreak":
#     # Input form for new data
#     st.header("Enter Data for Prediction")
#     with st.form("prediction_form"):
#         col1, col2 = st.columns(2)
#         with col1:
#             temp = st.number_input("Temperature (°C)", -30.0, 50.0, 25.0)
#             humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
#             precip = st.number_input("Precipitation (mm)", 0.0, 100.0, 0.0)
#         with col2:
#             case_rolling_avg = st.number_input("7-day Case Rolling Avg", 0.0, 100000.0, 100.0)
#             submit = st.form_submit_button("Predict")

#         if submit:
#             # Prepare input data
#             temp_humidity = temp * humidity
#             input_data = np.array([[temp, humidity, precip, temp_humidity, case_rolling_avg]])
#             input_scaled = scaler.transform(input_data)
            
#             # Predict
#             prediction = model.predict(input_scaled)[0]
#             prob = model.predict_proba(input_scaled)[0][1]
            
#             # Display result
#             st.subheader("Prediction Result")
#             if prediction == 1:
#                 st.error(f"Outbreak Predicted! (Probability: {prob:.2%})")
#             else:
#                 st.success(f"No Outbreak Predicted. (Probability of Outbreak: {prob:.2%})")

#             # Store prediction for charting
#             if 'predictions' not in st.session_state:
#                 st.session_state.predictions = []
#                 st.session_state.probs = []
#             st.session_state.predictions.append(prediction)
#             st.session_state.probs.append(prob)

#     # Display chart of predictions
#     if st.session_state.get('predictions'):
#         st.subheader("Prediction Trend")
#         chart_data = {
#             "labels": [f"Pred {i+1}" for i in range(len(st.session_state.predictions))],
#             "datasets": [{
#                 "label": "Outbreak Probability",
#                 "data": st.session_state.probs,
#                 "backgroundColor": "rgba(255, 99, 132, 0.5)",
#                 "borderColor": "rgba(255, 99, 132, 1)",
#                 "borderWidth": 1
#             }]
#         }







# # app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# from datetime import datetime

# # Page config
# st.set_page_config(page_title="COVID-19 New Cases Prediction (India, Jan 1, 2023)", layout="wide")

# # Title
# st.title("Predict New COVID-19 Cases in India for January 1, 2023")

# # Load pretrained model and scaler
# try:
#     model = joblib.load('rf_model1.pkl')
#     scaler = joblib.load('scaler1.pkl')
#     st.success("Pretrained model and scaler loaded successfully!")
# except FileNotFoundError:
#     st.error("Pretrained model or scaler not found. Please run train_model.py first.")
#     st.stop()

# # Load weather and COVID-19 datasets for feature estimation (replace with your paths)
# try:
#     weather_df = pd.read_csv('nn1.csv')  # DATE, temp, humidity, precip, temp_humidity
#     covid_df = pd.read_csv('mm1.csv')  # Date_reported, Country, New_cases
#     covid_df = covid_df[covid_df['Country'] == 'India']
#     weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d-%b-%y')
#     covid_df['Date_reported'] = pd.to_datetime(covid_df['Date_reported'], format='%d-%b-%y')
# except FileNotFoundError:
#     st.error("Dataset files not found. Please ensure 'weather_data.csv' and 'covid_data.csv' are available.")
#     st.stop()

# # Estimate weather features for January
# january_weather = weather_df[weather_df['DATE'].dt.month == 1]
# if not january_weather.empty:
#     temp = january_weather['temp'].mean()
#     humidity = january_weather['humidity'].mean()
#     precip = january_weather['precip'].mean()
#     temp_humidity = january_weather['temp_humidity'].mean()
# else:
#     # Fallback values based on sample data
#     temp, humidity, precip, temp_humidity = 15.0, 60.0, 0.0, 900.0
#     st.warning("No January data found. Using default values: temp=15°C, humidity=60%, precip=0mm.")

# # Estimate case rolling average
# latest_cases = covid_df[covid_df['Date_reported'] >= covid_df['Date_reported'].max() - pd.Timedelta(days=7)]
# if not latest_cases.empty:
#     case_rolling_avg = latest_cases['New_cases'].mean()
# else:
#     case_rolling_avg = 250.0  # Fallback based on Dec 2022 web data (~200-300 cases/day)
#     st.warning("No recent case data found. Using default 7-day avg: 250 cases.")

# # Display estimated features
# st.subheader("Estimated Features for January 1, 2023")
# st.write(f"- Temperature: {temp:.1f}°C")
# st.write(f"- Humidity: {humidity:.1f}%")
# st.write(f"- Precipitation: {precip:.1f} mm")
# st.write(f"- Temp-Humidity: {temp_humidity:.1f}")
# st.write(f"- 7-day Case Rolling Avg: {case_rolling_avg:.1f} cases")

# # Input form for date
# st.header("Confirm Prediction Date")
# with st.form("prediction_form"):
#     date_input = st.date_input(
#         "Select date",
#         value=datetime(2023, 1, 1),
#         min_value=datetime(2023, 1, 1),
#         max_value=datetime(2023, 12, 31),
#         help="Prediction is set for Jan 1, 2023"
#     )
#     submit = st.form_submit_button("Predict New Cases")

#     if submit:
#         if date_input != datetime(2023, 1, 1).date():
#             st.warning("Prediction is configured for Jan 1, 2023. Ignoring other dates.")
        
#         # Prepare input data
#         input_data = np.array([[temp, humidity, precip, temp_humidity, case_rolling_avg]])
#         input_scaled = scaler.transform(input_data)
        
#         # Predict
#         prediction = model.predict(input_scaled)[0]
        
#         # Display result
#         st.subheader("Prediction Result")
#         st.success(f"Predicted New Cases on January 1, 2023: {round(prediction)}")

#         # Store for chart
#         st.session_state.predictions = [prediction]
#         st.session_state.dates = ["01-Jan-23"]

# # Display chart
# if 'predictions' in st.session_state and st.session_state.predictions:
#     st.subheader("Prediction")
#     chart_data = {
#         "labels": st.session_state.dates,
#         "datasets": [{
#             "label": "Predicted New Cases",
#             "data": st.session_state.predictions,
#             "backgroundColor": "rgba(54, 162, 235, 0.5)",
#             "borderColor": "rgba(54, 162, 235, 1)",
#             "borderWidth": 1
#         }]
#     }
   








# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# from datetime import datetime

# # Page config
# st.set_page_config(page_title="COVID-19 New Cases Prediction (India)", layout="wide")

# # Title
# st.title("Predict New COVID-19 Cases in India")

# # Load pretrained model and scaler
# try:
#     model = joblib.load('rf_model1.pkl')
#     scaler = joblib.load('scaler1.pkl')
#     st.success("Pretrained model and scaler loaded successfully!")
# except FileNotFoundError:
#     st.error("Pretrained model or scaler not found. Please run train_model.py first.")
#     st.stop()

# # Load weather and COVID-19 datasets
# try:
#     weather_df = pd.read_csv('nn1.csv')  # DATE, temp, humidity, precip, temp_humidity
#     covid_df = pd.read_csv('mm1.csv')    # Date_reported, Country, New_cases
#     covid_df = covid_df[covid_df['Country'] == 'India']
#     weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d-%b-%y')
#     covid_df['Date_reported'] = pd.to_datetime(covid_df['Date_reported'], format='%d-%b-%y')
# except FileNotFoundError:
#     st.error("Dataset files not found. Please ensure 'nn1.csv' and 'mm1.csv' are available.")
#     st.stop()

# # Input form for date
# st.header("Select Prediction Date")
# with st.form("prediction_form"):
#     date_input = st.date_input(
#         "Select date",
#         value=datetime(2023, 1, 1),
#         min_value=datetime(2020, 1, 1),
#         max_value=datetime(2025, 12, 31),
#     )
#     date_input = pd.to_datetime(date_input)  # Convert to pandas Timestamp
#     submit = st.form_submit_button("Predict New Cases")

#     if submit:
#         # Try to get exact weather for the selected date
#         weather_on_date = weather_df[weather_df['DATE'] == date_input]

#         if weather_on_date.empty:
#             st.warning("Exact date's weather not found. Using monthly averages.")
#             month = date_input.month
#             monthly_weather = weather_df[weather_df['DATE'].dt.month == month]
#             if monthly_weather.empty:
#                 st.error("No weather data for the month. Cannot proceed.")
#                 st.stop()
#             temp = monthly_weather['temp'].mean()
#             humidity = monthly_weather['humidity'].mean()
#             precip = monthly_weather['precip'].mean()
#             temp_humidity = monthly_weather['temp_humidity'].mean()
#         else:
#             temp = weather_on_date['temp'].values[0]
#             humidity = weather_on_date['humidity'].values[0]
#             precip = weather_on_date['precip'].values[0]
#             temp_humidity = weather_on_date['temp_humidity'].values[0]

#         # Calculate rolling average cases for 7 days *before* selected date
#         prior_window = covid_df[(covid_df['Date_reported'] < date_input) &
#                                 (covid_df['Date_reported'] >= date_input - pd.Timedelta(days=7))]
#         if prior_window.empty:
#             st.warning("No recent case data found for rolling average. Using default 7-day avg: 250 cases.")
#             case_rolling_avg = 250.0
#         else:
#             case_rolling_avg = prior_window['New_cases'].mean()

#         # Show the features
#         st.subheader(f"Features used for prediction on {date_input.strftime('%Y-%m-%d')}:")
#         st.write(f"- Temperature: {temp:.1f}°C")
#         st.write(f"- Humidity: {humidity:.1f}%")
#         st.write(f"- Precipitation: {precip:.1f} mm")
#         st.write(f"- Temp-Humidity: {temp_humidity:.1f}")
#         st.write(f"- 7-day Case Rolling Avg: {case_rolling_avg:.1f} cases")

#         # Prepare input and predict
#         input_data = np.array([[temp, humidity, precip, temp_humidity, case_rolling_avg]])
#         input_scaled = scaler.transform(input_data)
#         prediction = model.predict(input_scaled)[0]

#         st.subheader("Prediction Result")
#         st.success(f"Predicted New Cases on {date_input.strftime('%Y-%m-%d')}: {round(prediction)}")

#         # Save for charting
#         st.session_state.predictions = st.session_state.get('predictions', []) + [prediction]
#         st.session_state.dates = st.session_state.get('dates', []) + [date_input.strftime('%d-%b-%y')]

# # Display chart
# if 'predictions' in st.session_state and st.session_state.predictions:
#     st.subheader("Prediction Chart")
#     chart_df = pd.DataFrame({
#         'Date': st.session_state.dates,
#         'Predicted Cases': st.session_state.predictions
#     })
#     st.line_chart(chart_df.set_index('Date'))





# import streamlit as st
# import pandas as pd
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# # Load covid data
# covid_df = pd.read_csv('mm1.csv')
# covid_df = covid_df[covid_df['Country'] == 'India']
# covid_df['Date_reported'] = pd.to_datetime(covid_df['Date_reported'], format='%d-%b-%y')
# covid_df = covid_df.sort_values('Date_reported')
# covid_df.set_index('Date_reported', inplace=True)
# ts = covid_df['New_cases']

# # Fit model
# model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,7))
# model_fit = model.fit(disp=False)

# st.title("COVID-19 New Cases Time Series Forecast")

# selected_date = st.date_input("Select date to predict cases", value=pd.to_datetime('2023-01-01'))

# last_date = ts.index.max()
# days_ahead = (pd.to_datetime(selected_date) - last_date).days

# if days_ahead <= 0:
#     st.warning(f"Date selected ({selected_date}) is before or equal to last known date ({last_date.date()}). Please select a future date.")
# else:
#     forecast = model_fit.get_forecast(steps=days_ahead)
#     pred = forecast.predicted_mean[-1]
#     st.success(f"Predicted new COVID-19 cases on {selected_date}: {int(pred)}")


# # COVID-19 dataset
# data1 = {
#     'Date_reported': pd.date_range(start='2022-09-01', end='2022-12-30', freq='D'),
#     'New_cases': [7946, 6168, 7211, 6817, 5910, 4417, 5379, 6395, 6093, 5554, 5076, 5221, 4369, 5108,
#                   6422, 6298, 5747, 5664, 4858, 4043, 4510, 5443, 5383, 4912, 4777, 4129, 3230, 3615,
#                   4272, 3947, 3805, 3375, 3011, 1968, 2468, 2529, 1997, 2797, 2756, 2424, 1957, 2139,
#                   2786, 2678, 2430, 2401, 2060, 1542, 1946, 2141, 2119, 2112, 1994, 1334, 862, 830,
#                   1112, 2208, 1574, 1604, 1326, 1046, 1190, 1321, 1216, 1082, 1132, 937, 625, 811,
#                   1016, 842, 833, 734, 547, -749, 501, 635, 656, 556, 492, 406, 294, 360, 408, 347,
#                   389, 343, 291, 215, 279, 291, 275, 253, 226, 226, 165, 166, 241, 249, 210, 173,
#                   159, 114, 152, 200, 162, 167, 176, 135, 112, 131, 185, 163, 201, 227, 196, 157,
#                   188, 268, 243]
# }












# Xgboost
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from statsmodels.stats.diagnostic import acorr_ljungbox
# import warnings
# warnings.filterwarnings("ignore")

# # Streamlit app title
# st.title("COVID-19 Case Prediction & Evaluation in India (Sep-Dec 2022) using XGBoost")

# # COVID-19 dataset
# data = pd.DataFrame({
#     'Date_reported1': pd.date_range(start='2022-09-01', end='2022-12-30', freq='D'),
#     'New_cases1': [
#         7946, 6168, 7211, 6817, 5887, 4417, 5379, 6395, 6093, 5554, 5076, 5221, 4369, 5108,
#         6422, 6298, 5747, 5664, 4858, 4043, 4510, 5443, 5383, 4912, 5263, 4129, 3230, 3615,
#         4272, 3947, 3805, 3375, 3011, 1968, 2468, 2529, 1997, 2797, 2756, 2424, 1957, 2139,
#         2786, 2678, 2430, 2401, 2060, 1542, 1946, 2141, 2119, 2112, 1994, 1334, 862, 830,
#         1112, 2208, 1574, 1604, 1326, 1046, 1190, 1321, 1216, 1082, 1132, 937, 625, 811,
#         1016, 842, 833, 734, 547, 0, 501, 635, 656, 556, 492, 406, 294, 360, 408, 347,
#         389, 343, 291, 215, 279, 291, 275, 253, 226, 226, 165, 166, 241, 249, 210, 173,
#         159, 114, 152, 200, 162, 167, 176, 135, 112, 131, 185, 163, 201, 227, 196, 157,
#         188, 268, 243
#     ]
# })
# data.set_index('Date_reported1', inplace=True)

# # Simulate weather data
# np.random.seed(42)
# data['Temperature'] = np.random.uniform(20, 30, len(data))
# data['Humidity'] = np.random.uniform(60, 90, len(data))

# # Display datasets
# st.subheader("COVID-19 and Weather Dataset")
# st.dataframe(data[['New_cases1', 'Temperature', 'Humidity']].head())

# # Create lagged features for time series
# def create_lagged_features(df, target_col, lags=7):
#     for lag in range(1, lags + 1):
#         df[f'lag_{lag}'] = df[target_col].shift(lag)
#     return df.dropna()

# # Prepare data with lagged features
# data = create_lagged_features(data, 'New_cases1', lags=7)
# features = ['Temperature', 'Humidity'] + [f'lag_{i}' for i in range(1, 8)]
# target = 'New_cases1'

# # Train-test split starting from 2022-11-01
# train_end_date = '2022-11-01'
# train = data.loc[:train_end_date]
# test = data.loc['2022-11-01':]
# st.write(f"Training data: {len(train)} days (until {train.index[-1].date()})")
# st.write(f"Test data: {len(test)} days (from {test.index[0].date()})")

# # User input for forecast horizon
# st.subheader("Forecast Settings")
# forecast_horizon = st.slider("Select forecast horizon (days)", min_value=1, max_value=40, value=10)

# # Train XGBoost model
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model.fit(train[features], train[target])

# # Predict on test set
# test_pred = model.predict(test[features])
# test_pred = pd.Series(test_pred, index=test.index)

# # Calculate metrics
# mae = mean_absolute_error(test['New_cases1'], test_pred)
# rmse = np.sqrt(mean_squared_error(test['New_cases1'], test_pred))
# mape = np.mean(np.abs((test['New_cases1'] - test_pred) / test['New_cases1'].replace(0, np.nan))) * 100 if not test['New_cases1'].eq(0).all() else np.nan

# # Residual diagnostics
# residuals = test['New_cases1'] - test_pred
# ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
# ljung_pvalue = ljung_box['lb_pvalue'].iloc[0]

# # Display metrics
# st.subheader("Model Evaluation Metrics (Test Set)")
# st.write(f"Mean Absolute Error (MAE): {mae:.2f} cases")
# st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} cases")
# st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# st.write(f"Ljung-Box Test p-value (lag 10): {ljung_pvalue:.4f} {'(Residuals appear random)' if ljung_pvalue > 0.05 else '(Residuals may have autocorrelation)'}")

# # Forecast beyond dataset
# future_dates = pd.date_range(start='2022-11-01', periods=forecast_horizon, freq='D')
# future_data = pd.DataFrame(index=future_dates)
# future_data['Temperature'] = np.random.uniform(20, 30, forecast_horizon)
# future_data['Humidity'] = np.random.uniform(60, 90, forecast_horizon)

# # Initialize lagged features with the last known values
# last_known = data.iloc[-1][[f'lag_{i}' for i in range(1, 8)]].values
# future_predictions = []

# # Iterative forecasting
# current_lags = list(last_known)
# for i in range(forecast_horizon):
#     input_data = pd.DataFrame({
#         'Temperature': [future_data['Temperature'].iloc[i]],
#         'Humidity': [future_data['Humidity'].iloc[i]],
#         **{f'lag_{j+1}': [current_lags[j]] for j in range(7)}
#     })
#     pred = model.predict(input_data)[0]
#     future_predictions.append(pred)
#     current_lags = [pred] + current_lags[:-1]  # Update lags with new prediction

# forecast_mean = pd.Series(future_predictions, index=future_dates)
# # Approximate confidence intervals (simple approach: using standard deviation of test residuals)
# std_residuals = residuals.std()
# forecast_ci = pd.DataFrame({
#     'lower': forecast_mean - 1.96 * std_residuals,
#     'upper': forecast_mean + 1.96 * std_residuals
# }, index=future_dates)

# # Plot actual vs predicted
# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=train.index, y=train['New_cases1'], mode='lines', name='Train', line=dict(color='#1f77b4')))
# fig1.add_trace(go.Scatter(x=test.index, y=test['New_cases1'], mode='lines', name='Test Actual', line=dict(color='#2ca02c')))
# fig1.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='#d62728')))
# fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci['upper'], mode='lines', name='Upper CI', line=dict(color='rgba(128, 128, 128, 0.2)'), showlegend=False))
# fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci['lower'], mode='lines', name='Lower CI', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', line=dict(color='rgba(128, 128, 128, 0.2)')))
# fig1.update_layout(title="COVID-19 New Cases: Actual vs Predicted (XGBoost)", xaxis_title="Date", yaxis_title="New Cases", template="plotly_white")
# st.plotly_chart(fig1)











# # Arima
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from statsmodels.stats.diagnostic import acorr_ljungbox
# import warnings
# warnings.filterwarnings("ignore")

# # Streamlit app title
# st.title("COVID-19 Case Prediction & Evaluation in India (Sep-Dec 2022) using ARIMA")

# # COVID-19 dataset
# data = pd.DataFrame({
#     'Date_reported1': pd.date_range(start='2022-09-01', end='2022-12-30', freq='D'),
#     'New_cases1': [
#         7946, 6168, 7211, 6817, 5887, 4417, 5379, 6395, 6093, 5554, 5076, 5221, 4369, 5108,
#         6422, 6298, 5747, 5664, 4858, 4043, 4510, 5443, 5383, 4912, 5263, 4129, 3230, 3615,
#         4272, 3947, 3805, 3375, 3011, 1968, 2468, 2529, 1997, 2797, 2756, 2424, 1957, 2139,
#         2786, 2678, 2430, 2401, 2060, 1542, 1946, 2141, 2119, 2112, 1994, 1334, 862, 830,
#         1112, 2208, 1574, 1604, 1326, 1046, 1190, 1321, 1216, 1082, 1132, 937, 625, 811,
#         1016, 842, 833, 734, 547, 0, 501, 635, 656, 556, 492, 406, 294, 360, 408, 347,
#         389, 343, 291, 215, 279, 291, 275, 253, 226, 226, 165, 166, 241, 249, 210, 173,
#         159, 114, 152, 200, 162, 167, 176, 135, 112, 131, 185, 163, 201, 227, 196, 157,
#         188, 268, 243
#     ]
# })
# data.set_index('Date_reported1', inplace=True)

# # Simulate weather data (not used in ARIMA model, included for consistency)
# np.random.seed(42)
# data['Temperature'] = np.random.uniform(20, 30, len(data))
# data['Humidity'] = np.random.uniform(60, 90, len(data))

# # Display dataset
# st.subheader("COVID-19 Dataset")
# st.dataframe(data[['New_cases1']].head())

# # Train-test split starting from 2022-11-01
# train_end_date = '2022-11-01'
# train = data.loc[:train_end_date]
# test = data.loc['2022-11-01':]
# st.write(f"Training data: {len(train)} days (until {train.index[-1].date()})")
# st.write(f"Test data: {len(test)} days (from {test.index[0].date()})")

# # User input for forecast horizon
# st.subheader("Forecast Settings")
# forecast_horizon = st.slider("Select forecast horizon (days)", min_value=1, max_value=100, value=10)

# # Train ARIMA model
# model = ARIMA(train['New_cases1'], order=(1, 1, 1))
# results = model.fit()

# # Predict on test set
# test_pred = results.get_forecast(steps=len(test))
# test_pred_mean = test_pred.predicted_mean
# test_pred_ci = test_pred.conf_int()

# # Calculate metrics
# mae = mean_absolute_error(test['New_cases1'], test_pred_mean)
# rmse = np.sqrt(mean_squared_error(test['New_cases1'], test_pred_mean))
# mape = np.mean(np.abs((test['New_cases1'] - test_pred_mean) / test['New_cases1'].replace(0, np.nan))) * 100 if not test['New_cases1'].eq(0).all() else np.nan

# # Residual diagnostics
# residuals = test['New_cases1'] - test_pred_mean
# ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
# ljung_pvalue = ljung_box['lb_pvalue'].iloc[0]

# # Display metrics
# st.subheader("Model Evaluation Metrics (Test Set)")
# st.write(f"Mean Absolute Error (MAE): {mae:.2f} cases")
# st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} cases")
# st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# st.write(f"Ljung-Box Test p-value (lag 10): {ljung_pvalue:.4f} {'(Residuals appear random)' if ljung_pvalue > 0.05 else '(Residuals may have autocorrelation)'}")

# # Forecast beyond dataset starting from 2022-11-01
# future_dates = pd.date_range(start='2022-11-01', periods=forecast_horizon, freq='D')
# forecast = results.get_forecast(steps=forecast_horizon)
# forecast_mean = forecast.predicted_mean
# forecast_ci = forecast.conf_int()

# # Plot actual vs predicted
# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=train.index, y=train['New_cases1'], mode='lines', name='Train', line=dict(color='#1f77b4')))
# fig1.add_trace(go.Scatter(x=test.index, y=test['New_cases1'], mode='lines', name='Test Actual', line=dict(color='#2ca02c')))
# fig1.add_trace(go.Scatter(x=test.index, y=test_pred_mean, mode='lines', name='Test Predicted', line=dict(color='#ff7f0e')))
# fig1.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='#d62728')))
# fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', name='Upper CI', line=dict(color='rgba(128, 128, 128, 0.2)'), showlegend=False))
# fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', name='Lower CI', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', line=dict(color='rgba(128, 128, 128, 0.2)')))
# fig1.update_layout(title="COVID-19 New Cases: Actual vs Predicted (ARIMA)", xaxis_title="Date", yaxis_title="New Cases", template="plotly_white")
# st.plotly_chart(fig1)













# prophet

# import streamlit as st
# import pandas as pd
# import numpy as np
# from prophet import Prophet
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # Streamlit app title
# st.title("COVID-19 New Cases Forecast for Next 30 Days (Nov 16 - Dec 15, 2022)")

# # Load and preprocess data
# @st.cache_data
# def load_data():

# # Data preparation
#     dates = pd.to_datetime([
#     '01-Jan-21', '01-Feb-21', '01-Mar-21', '01-Apr-21', '01-May-21', '01-Jun-21',
#     '01-Jul-21', '01-Aug-21', '01-Sep-21', '01-Oct-21', '01-Nov-21', '01-Dec-21',
#     '02-Dec-21', '03-Dec-21', '04-Dec-21', '05-Dec-21', '06-Dec-21', '07-Dec-21',
#     '08-Dec-21', '09-Dec-21', '10-Dec-21', '11-Dec-21', '12-Dec-21', '13-Dec-21',
#     '14-Dec-21', '15-Dec-21', '16-Dec-21', '17-Dec-21', '18-Dec-21', '19-Dec-21',
#     '20-Dec-21', '21-Dec-21', '22-Dec-21', '23-Dec-21', '24-Dec-21', '25-Dec-21',
#     '26-Dec-21', '27-Dec-21', '28-Dec-21', '29-Dec-21', '30-Dec-21', '31-Dec-21',
#     '01-Jan-22', '02-Jan-22', '03-Jan-22', '04-Jan-22', '05-Jan-22', '06-Jan-22',
#     '07-Jan-22', '08-Jan-22', '09-Jan-22', '10-Jan-22', '11-Jan-22', '12-Jan-22',
#     '13-Jan-22', '14-Jan-22', '15-Jan-22', '16-Jan-22', '17-Jan-22', '18-Jan-22',
#     '19-Jan-22', '20-Jan-22', '21-Jan-22', '22-Jan-22', '23-Jan-22', '24-Jan-22',
#     '25-Jan-22', '26-Jan-22', '27-Jan-22', '28-Jan-22', '29-Jan-22', '30-Jan-22',
#     '31-Jan-22', '01-Feb-22', '02-Feb-22', '03-Feb-22', '04-Feb-22', '05-Feb-22',
#     '06-Feb-22', '07-Feb-22', '08-Feb-22', '09-Feb-22', '10-Feb-22', '11-Feb-22',
#     '12-Feb-22', '13-Feb-22', '14-Feb-22', '15-Feb-22', '16-Feb-22', '17-Feb-22',
#     '18-Feb-22', '19-Feb-22', '20-Feb-22', '21-Feb-22', '22-Feb-22', '23-Feb-22',
#     '24-Feb-22', '25-Feb-22', '26-Feb-22', '27-Feb-22', '28-Feb-22', '01-Mar-22',
#     '02-Mar-22', '03-Mar-22', '04-Mar-22', '05-Mar-22', '06-Mar-22', '07-Mar-22',
#     '08-Mar-22', '09-Mar-22', '10-Mar-22', '11-Mar-22', '12-Mar-22', '13-Mar-22',
#     '14-Mar-22', '15-Mar-22', '16-Mar-22', '17-Mar-22', '18-Mar-22', '19-Mar-22',
#     '20-Mar-22', '21-Mar-22', '22-Mar-22', '23-Mar-22', '24-Mar-22', '25-Mar-22',
#     '26-Mar-22', '27-Mar-22', '28-Mar-22', '29-Mar-22', '30-Mar-22', '31-Mar-22',
#     '01-Apr-22', '02-Apr-22', '03-Apr-22', '04-Apr-22', '05-Apr-22', '06-Apr-22',
#     '07-Apr-22', '08-Apr-22', '09-Apr-22', '10-Apr-22', '11-Apr-22', '12-Apr-22',
#     '13-Apr-22', '14-Apr-22', '15-Apr-22', '16-Apr-22', '17-Apr-22', '18-Apr-22',
#     '19-Apr-22', '20-Apr-22', '21-Apr-22', '22-Apr-22', '23-Apr-22', '24-Apr-22',
#     '25-Apr-22', '26-Apr-22', '27-Apr-22', '28-Apr-22', '29-Apr-22', '30-Apr-22',
#     '01-May-22', '02-May-22', '03-May-22', '04-May-22', '05-May-22', '06-May-22',
#     '07-May-22', '08-May-22', '09-May-22', '10-May-22', '11-May-22', '12-May-22',
#     '13-May-22', '14-May-22', '15-May-22', '16-May-22', '17-May-22', '18-May-22',
#     '19-May-22', '20-May-22', '21-May-22', '22-May-22', '23-May-22', '24-May-22',
#     '25-May-22', '26-May-22', '27-May-22', '28-May-22', '29-May-22', '30-May-22',
#     '31-May-22', '01-Jun-22', '02-Jun-22', '03-Jun-22', '04-Jun-22', '05-Jun-22',
#     '06-Jun-22', '07-Jun-22', '08-Jun-22', '09-Jun-22', '10-Jun-22', '11-Jun-22',
#     '12-Jun-22', '13-Jun-22', '14-Jun-22', '15-Jun-22', '16-Jun-22', '17-Jun-22',
#     '18-Jun-22', '19-Jun-22', '20-Jun-22', '21-Jun-22', '22-Jun-22', '23-Jun-22',
#     '24-Jun-22', '25-Jun-22', '26-Jun-22', '27-Jun-22', '28-Jun-22', '29-Jun-22',
#     '30-Jun-22', '01-Jul-22', '02-Jul-22', '03-Jul-22', '04-Jul-22', '05-Jul-22',
#     '06-Jul-22', '07-Jul-22', '08-Jul-22', '09-Jul-22', '10-Jul-22', '11-Jul-22',
#     '12-Jul-22', '13-Jul-22', '14-Jul-22', '15-Jul-22', '16-Jul-22', '17-Jul-22',
#     '18-Jul-22', '19-Jul-22', '20-Jul-22', '21-Jul-22', '22-Jul-22', '23-Jul-22',
#     '24-Jul-22', '25-Jul-22', '26-Jul-22', '27-Jul-22', '28-Jul-22', '29-Jul-22',
#     '30-Jul-22', '31-Jul-22', '01-Aug-22', '02-Aug-22', '03-Aug-22', '04-Aug-22',
#     '05-Aug-22', '06-Aug-22', '07-Aug-22', '08-Aug-22', '09-Aug-22', '10-Aug-22',
#     '11-Aug-22', '12-Aug-22', '13-Aug-22', '14-Aug-22', '15-Aug-22', '16-Aug-22',
#     '17-Aug-22', '18-Aug-22', '19-Aug-22', '20-Aug-22', '21-Aug-22', '22-Aug-22',
#     '23-Aug-22', '24-Aug-22', '25-Aug-22', '26-Aug-22', '27-Aug-22', '28-Aug-22',
#     '29-Aug-22', '30-Aug-22', '31-Aug-22', '01-Sep-22', '02-Sep-22', '03-Sep-22',
#     '04-Sep-22', '05-Sep-22', '06-Sep-22', '07-Sep-22', '08-Sep-22', '09-Sep-22',
#     '10-Sep-22', '11-Sep-22', '12-Sep-22', '13-Sep-22', '14-Sep-22', '15-Sep-22',
#     '16-Sep-22', '17-Sep-22', '18-Sep-22', '19-Sep-22', '20-Sep-22', '21-Sep-22',
#     '22-Sep-22', '23-Sep-22', '24-Sep-22', '25-Sep-22', '26-Sep-22', '27-Sep-22',
#     '28-Sep-22', '29-Sep-22', '30-Sep-22', '01-Oct-22', '02-Oct-22', '03-Oct-22',
#     '04-Oct-22', '05-Oct-22', '06-Oct-22', '07-Oct-22', '08-Oct-22', '09-Oct-22',
#     '10-Oct-22', '11-Oct-22', '12-Oct-22', '13-Oct-22', '14-Oct-22', '15-Oct-22',
#     '16-Oct-22', '17-Oct-22', '18-Oct-22', '19-Oct-22', '20-Oct-22', '21-Oct-22',
#     '22-Oct-22', '23-Oct-22', '24-Oct-22', '25-Oct-22', '26-Oct-22', '27-Oct-22',
#     '28-Oct-22', '29-Oct-22', '30-Oct-22', '31-Oct-22', '01-Nov-22', '02-Nov-22',
#     '03-Nov-22', '04-Nov-22', '05-Nov-22', '06-Nov-22', '07-Nov-22', '08-Nov-22',
#     '09-Nov-22', '10-Nov-22', '11-Nov-22', '12-Nov-22', '13-Nov-22', '14-Nov-22',
#     '15-Nov-22'
# ],format='%d-%b-%y')

#     new_cases = [
#     20035, 11427, 15510, 72330, 401993, 127510, 48786, 41831, 41965, 26727, 12514, 8954, 9765,
#     9216, 8603, 8895, 8306, 6822, 8439, 9419, 8503, 7992, 7774, 7350, 5784, 6984, 7974, 7447,
#     7145, 7081, 6563, 5326, 6317, 7495, 6650, 7189, 6987, 6531, 6358, 9195, 13154, 16764,
#     22775, 27553, 33750, 37379, 58097, 90928, 117100, 141986, 159632, 179723, 168063, 194720,
#     247417, 264202, 268833, 271202, 258089, 238018, 282970, 317532, 347254, 337704, 333533,
#     306064, 255874, 285914, 286384, 251209, 235532, 234281, 209918, 167059, 161386, 172433,
#     149394, 127952, 107474, 83876, 67597, 71365, 67084, 58077, 50407, 44877, 34113, 27409,
#     30615, 30757, 25920, 22270, 19968, 16051, 13405, 15102, 14148, 13166, 11499, 10273, 8013,
#     6915, 7554, 6561, 6396, 5921, 5476, 4362, 3993, 4575, 4184, 4194, 3614, 3116, 2503, 2568,
#     2876, 2539, 2528, 2075, 1761, 1549, 1581, 1778, 1938, 1685, 1660, 1421, 1270, 1259, 1233,
#     1225, 1335, 1260, 1096, 913, 795, 1086, 1033, 1109, 1150, 1054, 861, 796, 1088, 1007, 949,
#     975, 1150, 2183, 1247, 2067, 2380, 2451, 2527, 2593, 2541, 2483, 2927, 3303, 3377, 3688,
#     3324, 3157, 2568, 3205, 3275, 3545, 3805, 3451, 3207, 2288, 2897, 2827, 2841, 2858, 2487,
#     2202, 1569, 1829, 2364, 2259, 2323, 2226, 2022, 1675, 2124, 2628, 2710, 2685, 2828, 2706,
#     2338, 2745, 3712, 4041, 3962, 4270, 4518, 3714, 5233, 7240, 7584, 8329, 8582, 8084, 6594,
#     8822, 12213, 12847, 13216, 12899, 12781, 9923, 12249, 13313, 17336, 15940, 11739, 17073,
#     11793, 14506, 18819, 17070, 17092, 16103, 16135, 13086, 16159, 18930, 18815, 18840, 18257,
#     16678, 13615, 16906, 20139, 20038, 20044, 20528, 16935, 15528, 20557, 21566, 21880, 21411,
#     20279, 16866, 14830, 18313, 20557, 20409, 20408, 19673, 16464, 13734, 17135, 19893, 20551,
#     19406, 18738, 16167, 12751, 16047, 16299, 16561, 15815, 14092, 14917, 8813, 9062, 12608,
#     15754, 13272, 11539, 9531, 8586, 10649, 10725, 10256, 9520, 9436, 7591, 5439, 7231, 7946,
#     6168, 7211, 6817, 5910, 4417, 5379, 6395, 6093, 5554, 5076, 5221, 4369, 5108, 6422, 6298,
#     5747, 5664, 4858, 4043, 4510, 5443, 5383, 4912, 4777, 4129, 3230, 3615, 4272, 3947, 3805,
#     3375, 3011, 1968, 2468, 2529, 1997, 2797, 2756, 2424, 1957, 2139, 2786, 2678, 2430, 2401,
#     2060, 1542, 1946, 2141, 2119, 2112, 1994, 1334, 862, 830, 1112, 2208, 1574, 1604, 1326,
#     1046, 1190, 1321, 1216, 1082, 1132, 937, 625, 811, 1016, 842, 833, 734, 547, 749
# ]

#     Temp=[
#     10.2, 13.4, 13.9, 16.2, 17.1, 16.5, 16.3, 13.5, 14.6, 11.4, 11.2, 10.5, 10, 11.2, 13.1, 
#     11.1, 10.3, 14.1, 12.9, 13.3, 14.2, 11.6, 13.2, 12.3, 12, 12.6, 12.9, 12.4, 12.3, 12.5, 
#     13.2, 16.4, 17.1, 18.2, 16.5, 14.3, 15, 15.1, 15.6, 16.4, 18.4, 16.9, 18.1, 18.4, 18.7, 
#     19.8, 20.6, 18.9, 19.2, 18.1, 18.2, 18.5, 19.3, 20.9, 21.7, 22.6, 23.6, 24.6, 23.8, 20.9, 
#     20.9, 21.3, 23.1, 22.6, 22.9, 24.5, 25.2, 26.1, 25.1, 25.7, 24.6, 24, 23.9, 24.3, 24.8, 
#     25.6, 25.9, 26.6, 25.2, 26.3, 26.4, 24.8, 25.4, 23.4, 24, 26.1, 27.9, 30.2, 29.5, 27.9, 
#     27.1, 25.8, 26, 27.1, 29.9, 30.4, 29.7, 27.6, 27.7, 29, 29.4, 29.8, 30.5, 31.5, 31.3, 
#     28.8, 26.7, 28.3, 30, 28, 26.1, 27.4, 26.6, 26.8, 28.2, 30.1, 32.1, 33.2, 33.5, 33.3, 
#     33.7, 32.2, 32.5, 33.1, 32.6, 29.5, 28.7, 31.8, 31.3, 30.6, 32.1, 31, 28, 30.4, 30.6, 
#     32.7, 32.8, 27.8, 22.4, 25.2, 26.2, 28, 27.7, 29.5, 32, 32.8, 32.6, 32.2, 32.2, 30.9, 
#     32.8, 26.9, 30, 31.8, 31, 30.1, 31.6, 34, 35.8, 36.8, 34.2, 31.6, 31.2, 29.4, 32.3, 
#     31.9, 29.8, 31.3, 30.7, 30.3, 29.4, 32.1, 34.7, 35.1, 32.2, 32.3, 31.4, 33.6, 34.5, 
#     36.3, 37.7, 37.5, 34, 31.6, 33, 33.8, 34.8, 36.4, 36.1, 33, 34.5, 33.7, 31.7, 29.2, 
#     28.1, 30.5, 33.1, 33.1, 30.6, 26.3, 27.6, 29.5, 29.4, 30.5, 31.5, 32.4, 30.9, 28.5, 
#     26.7, 28.3, 27.7, 29.8, 28.5, 28.8, 30, 30.6, 30.7, 31.2, 29.5, 28.2, 29, 31, 31.2, 
#     32, 32.1, 32, 32.1, 32.8, 33.1, 33, 32.3, 27.8, 27.4, 29.4, 30.5, 31.3, 32.2, 30.9, 
#     31.2, 31, 30.1, 29.4, 26.7, 25.7, 27.7, 29.3, 28.1, 30, 31.1, 29.8, 29.4, 29.8, 28.3, 
#     25.1, 26.8, 27.6, 28.8, 29.7, 26.2, 27.7, 28.6, 29.9, 30.1, 29.6, 27.5, 27.9, 28.2, 
#     28.7, 28.1, 29.6, 30.3, 30.1, 29.7, 29.6, 30.3, 30.2, 30.6, 29.8, 29.5, 28.8, 28.9, 
#     29.2, 29.2, 29.5, 29, 28, 26, 26.3, 27.3, 25, 21.9, 24.7, 25.2, 24.8, 25.3, 25.1, 23.8, 
#     22.3, 22.2, 22, 21.9, 22.5, 22.3, 22.5, 22.1, 22.5, 22.4, 21, 20.6, 20.9, 21, 21, 21.4, 
#     21.5, 19.4, 19.7, 19.7, 19.3, 18.9, 18.6, 18.1, 19.3, 19.2, 19.8, 19.3, 18.9, 18, 18.2, 
#     18.9, 19, 17.7, 18.6, 18.2, 17.8, 17.7, 16.6, 17.2, 18.5, 18.8, 19.3, 17.5, 16.4, 16.3, 
#     16.3, 15.5, 15.3, 14.9, 15.1, 15.2, 14.6, 13.7, 11.1, 11.1, 12, 13.3, 14.2, 14.6, 15.2, 
#     15.5, 15.7, 15.4, 14.2, 14, 11.9, 11.2, 11.9, 12.7, 13.8, 15.8, 13.7, 15.2, 16.2, 15.8, 
#     13.8, 12.5, 11.4, 11.5, 11.4, 10.5, 10, 10.7, 11.2, 10.6, 11.9, 12.7, 13.4, 13, 12.6, 
#     11.7, 10, 11.2, 12.2, 12.7, 13.7, 14.6, 14.3, 13.4, 14.7, 13, 13.1, 12.5, 14.1, 17.4, 
#     18.5, 16.5, 15.7, 15.5, 16, 17.4, 17.9, 18.8, 18.1, 18.4, 19.1, 18.8, 17.9, 18, 19.1, 
#     19.2, 19.6, 20.1, 18.4, 18.7, 18.1, 18.3, 19.4, 21.8, 20.4, 19.5, 20.3, 22.1, 22.5, 23.1, 
#     22.9, 22.7, 23.1, 24.5, 25.6, 26.9, 27.6, 27.7, 28.2, 28.5, 29.6, 28.8, 28.3, 27.8, 28, 
#     27.6, 27, 28, 29.6, 30.3, 30.5, 30.7, 29.9, 29.9, 30.3, 30.2, 30.2, 30.7, 31.4, 32.1, 33, 
#     33.4, 33.7, 32.7, 32.8, 32.3, 32.4, 32.1, 32.6, 33.4, 34.3, 33.8, 31.4, 31.5, 31.7, 32.4, 
#     33.7, 33.5, 33.9, 34.6, 35.8, 35.6, 33.9, 33.9, 34.1, 33.2, 31.1, 30, 32.3, 33.4, 33.6, 
#     33.8, 33.8, 34.6, 35.9, 35, 37.2, 37, 37.4, 34.6, 35.8, 36.2, 37, 34.8, 32.2, 25.7, 26.3, 
#     29.1, 30.6, 32.4, 34.4, 34, 31.4, 32.9, 35, 35.4, 35.6, 36.5, 37.3, 36.8, 37.3, 37.3, 
#     37.5, 37.8, 37.7, 37.7, 37.6, 35.3, 36.1, 34.1, 29.2, 28.9, 28.3, 29, 30.2, 31, 33.1, 
#     33.3, 34.3, 35, 35.7, 35.9, 34.9, 27.7, 28.2, 31, 33.1, 33.5, 33.9, 34.4, 34.6, 34.9, 
#     32, 31, 31.8, 30.3, 31.5, 32.3, 32.9, 29.4, 29.9, 33.2, 32.4, 29.7, 29, 29.4, 29.7, 30, 
#     30, 29.4, 30, 28.6, 28, 28.4, 28.7, 29.7, 31, 30.8, 29.4, 27.9, 30.2, 27.6, 30.1, 32, 
#     32.5, 29.6, 31.2, 30.9, 28.9, 29.3, 29.1, 29.2, 30.3, 31.1, 29.6, 30.9, 31.5, 30.7, 29.1, 
#     30.1, 30.7, 30.7, 30.6, 30.3, 30.2, 31.5, 32.1, 32.1, 32.2, 32.3, 31.9, 31.7, 32.1, 32.2, 
#     32.2, 31.6, 32.2, 31.8, 30, 28.5, 28.5, 29.5, 29.3, 28.8, 27.1, 25.4, 25, 25.3, 27, 28.1, 
#     28.6, 29.2, 28.8, 28.9, 29.1, 29.5, 29, 29.1, 28.7, 26, 23.9, 22.5, 22, 22, 24.8, 25.3, 
#     25.4, 25.5, 25.8, 26.2, 26.5, 26, 26.5, 26.2, 26.2, 25.2, 23.9, 23.6, 23.8, 24.1, 24.3, 
#     24, 23.8, 23.3, 23.4, 24.6, 24, 23.9, 24.4, 24, 24.8, 25, 23.7, 23.4, 23.5, 21.8, 21, 
#     20.5, 22.6, 22, 20.6, 18.7, 18.6, 18.7, 18.7, 18.6, 18.6, 18.5, 18.3, 18.3, 18.2, 17.6, 
#     17.8
# ]
#     humidity = [
#         79.2, 77.8, 90.1, 94.2, 93.8, 95.2, 90.6, 92.2, 90.5, 86.8, 86.2, 85.5, 84.4, 84.1, 85.6,
#         89.4, 90.6, 86.9, 88.5, 83.9, 75.4, 87.4, 84.1, 92.7, 88.7, 77, 76.2, 75.4, 77.2, 72.7,
#         67.7, 62.6, 70.9, 76, 87.6, 80.6, 80.2, 81.8, 78.7, 75.8, 72.1, 80.7, 79, 79.1, 74.4,
#         74.4, 74.8, 73.8, 73.6, 77.6, 76.7, 75.2, 72.3, 67.3, 65.3, 67.1, 64.4, 59.4, 52.8,
#         58.8, 57.7, 57.6, 57.1, 60.8, 60.2, 61.5, 63.7, 67.5, 60.8, 63.5, 52.6, 59.7, 61.8,
#         60.5, 61.7, 55.3, 52.5, 48, 50.7, 51.7, 46.5, 48.9, 45.4, 55.8, 55, 57, 50.9, 41.5,
#         39.4, 35, 22.5, 27.5, 23, 27.9, 28.7, 30.4, 31.6, 26.4, 25.5, 22.9, 23.8, 28, 28.5,
#         25, 23.9, 39.5, 54, 40.8, 33.8, 34.7, 47.7, 45.3, 44.6, 33.8, 32.2, 30.2, 30.1, 27.9,
#         26.8, 28.9, 28.5, 30.4, 30.2, 31.7, 38, 51.5, 51.8, 40.6, 42.4, 41.4, 40.4, 39.8,
#         54.9, 47.2, 39.8, 32.1, 31.2, 60.6, 94.6, 63.4, 76.9, 58.5, 58, 45.6, 33.1, 28.4,
#         35.5, 50.6, 50.1, 51.4, 43.7, 68.7, 58, 52, 60.9, 61.3, 58.3, 47.4, 43.8, 40.8, 60,
#         66.5, 67.7, 70.8, 61.5, 60.5, 70.1, 67.8, 70.8, 70.3, 77.6, 66.8, 49.8, 42.5, 63.4,
#         53, 60.3, 55.4, 54.2, 46.7, 36.8, 38.1, 51, 61.9, 55.7, 50, 51.1, 46.2, 44.4, 58.7,
#         59.2, 65.4, 72.8, 87.4, 91.4, 79.1, 73.9, 70.8, 84.4, 99.3, 95.2, 85.3, 83.6, 82.3,
#         81.7, 75.1, 83, 94.6, 96.8, 92.5, 94.8, 85.4, 92.5, 91.5, 63.7, 78.5, 80.5, 78.2,
#         87.3, 91.8, 89.2, 76.8, 72.4, 67.6, 67.8, 66, 63.8, 65.5, 68.2, 65.9, 70.9, 91.6,
#         95.3, 87.5, 85.5, 82.1, 70.6, 66.5, 66.6, 70.8, 82.7, 83.8, 96.1, 98.2, 92.4, 84.6, 89.7,
#         82.8, 83.2, 86.8, 84.3, 84.2, 91, 98, 94.6, 93.5, 85.6, 80.6, 93.7, 83.6, 85.1, 80.3,
#         80.9, 82.8, 89.4, 87, 85.4, 84.8, 88.6, 82.3, 79.3, 79.6, 80.6, 81.1, 80.5, 79, 72.5,
#         74.3, 76.9, 74.7, 67, 61.2, 60.4, 57.8,60,3, 51.7, 55, 59.2, 66.1, 87.2, 94.9, 82.4,
#         73.2, 55.7, 60.6, 66.8, 73.5, 78.7, 71.7, 68.7, 67.3, 68, 64.3, 64.8, 60.9, 62.8,
#         70, 79.1, 80.3, 71.7, 70.4, 62.7, 59.1, 63.3, 71.5, 75.3, 63.7, 56.6, 61, 63.2,
#         63.3, 67.5, 68.9, 63.3, 60.3, 64.9, 53.8, 65.7, 69.2, 71.2, 77.4, 81.3, 73.8,
#         70, 75.6, 89.3, 87.3, 75.1, 77.6, 79.6, 76.7, 73, 76.5, 73.7, 66.6, 67.1, 71.7,
#         77.4, 75.3, 71, 70.4, 75.1, 67.6, 64.9, 65.4, 65.4, 70.3, 76.6, 80, 88, 80.5,
#         88.7, 82.9, 70, 76.4, 80.7, 80.9, 76.1, 71.1, 91.8, 88.8, 90.2, 96.9, 97.2,
#         92.3, 91.5, 88.6, 91.2, 94.6, 93.9, 87.5, 88.5, 92.2, 89.4, 92.4, 90.4, 94.3,
#         94.4, 93.5, 94.5, 92.5, 83.9, 76, 73.8, 78.2, 80.5, 89.2, 83.2, 90.1, 87.3,
#         82.4, 72, 72.1, 77.3, 84.6, 74.5, 69.7, 66.7, 65.5, 64.6, 58.3, 68.5, 66.4,
#         61.6, 64, 69.4, 57.4, 58.6, 78.2, 72, 71.7, 70.8, 70.2, 62.7, 68.6, 68.7,
#         63.7, 73.3, 71.6, 70, 63.8, 62.9, 56.2, 57, 65.2, 58.9, 61.9, 68.1, 69.5,
#         65.1, 66.2, 70.7, 64.3, 49.7, 54, 51.2, 50.1, 42, 51.5, 45.2, 41.5, 34.8,
#         31.9, 34.4, 34.4, 29, 28.1, 26.1, 24.6, 26.7, 23.9, 25.2, 23.1, 23.5, 20.9,
#         22.8, 22.3, 26, 23.1, 27.1, 30, 28.2, 27, 25.8, 22.6, 28.7, 29.3, 25.8,
#         23.6, 23.5, 22.6, 23.7, 23.3, 20.2, 17.8, 32.4, 32.4, 41.7, 49.2, 54.8,
#         53.2, 44.5, 36.2, 39.6, 45.3, 45.9, 44.4, 41.8, 45.9, 28.4, 21.3, 18.1,
#         31.2, 31, 32.6, 30.6, 30.8, 44.9, 71.2, 75.1, 66.6, 59.1, 50.5, 46.6,
#         50.4, 59.4, 49.2, 38.7, 32.8, 28.5, 27, 20.8, 23.8, 22.2, 21.5, 24.6,
#         26.1, 22.1, 24.7, 29.2, 38, 39.3, 46.1, 70.8, 74.1, 74.4, 65.6, 61.2,
#         59.3, 48.1, 38.6, 34.8, 48, 51.1, 53.1, 53.8, 88.2, 88.6, 72.9, 57.4,
#         64.1, 64.5, 63, 59.3, 54.8, 66.5, 73.6, 67.9, 80.1, 73.3, 67.6, 63.7,
#         80.4, 81.9, 68.8, 73.5, 84, 84, 86.7, 85.6, 82.5, 80.3, 81, 80, 86.3,
#         89.2, 86.9, 84.3, 79.6, 75.9, 76.1, 84.6, 92.1, 84.5, 93.3, 79.9, 75.3,
#         68.9, 73.9, 71.1, 73.4, 86.6, 81.3, 78.7, 76.8, 76.1, 71.6, 80.8, 76.7,
#         74.3, 71.4, 80.9, 72.3, 73, 70.2, 65.9, 66.9, 77.4, 66.9, 66.3, 62.9,
#         58.2, 56.6, 58.7, 59.4, 59.3, 61.8, 64.8, 71.9, 61.3, 62.8, 70.4, 78.6,
#         82.2, 73.6, 77.3, 77.1, 87.1, 95.9, 98.2, 97.2, 84.8, 82.4, 76.5, 71.1,
#         71.6, 73.5, 72, 64.1, 59.4, 63.3, 69.7, 78.3, 87.3, 96.9, 99, 97.2, 87.1,
#         81.7, 72.7, 71.6, 69.8, 61, 56, 55.6, 50.8, 54.1, 55.3, 56.8, 55, 60.1,
#         53.7, 60.9, 59.7, 64.5, 68.2, 62, 66, 62.6, 64.6, 65.1, 66.9, 78.3,
#         72.3, 64.1, 69.5, 60.6, 64.6, 61.9, 53.4, 56.7, 55.4
#     ]

#     wind_speed=[
#     11.2, 16.6, 16.6, 18.4, 14.8, 12.6, 11.2, 9.3, 11.2, 16.6, 16.6, 16.2, 11.9, 7.6, 7.6, 
#     10.8, 20.5, 13, 14.8, 19.4, 20.5, 11.5, 16.6, 14.8, 13, 13.7, 14.8, 9.4, 18.4, 17.8, 
#     9.4, 11.3, 9.4, 14.8, 15.7, 11.2, 18.4, 18.4, 13, 9.4, 13, 16.6, 11.2, 8.6, 6.1, 12.6, 
#     7.6, 9, 8.5, 15.9, 11.9, 15.2, 15.1, 12.6, 13.3, 16.5, 16.6, 17.7, 14.1, 21.6, 26.5, 
#     15.6, 9.7, 19.3, 15.8, 13, 16.6, 16.6, 14.8, 13.8, 27.7, 11.2, 17.7, 18.4, 15.1, 14.8, 
#     11.2, 14.8, 18.3, 14.8, 27.7, 24.1, 20.5, 17.8, 18.4, 42.3, 10, 18.4, 31.4, 21.3, 22.8, 
#     12.4, 16.4, 18.3, 13.4, 14.8, 18.4, 15.7, 18.4, 16.6, 18.4, 18.4, 18.5, 18.4, 19.8, 
#     37.1, 21.2, 16.6, 18.4, 37.1, 16.6, 16.6, 24.1, 22.3, 22.3, 18.4, 20.5, 22.3, 18.4, 
#     18.4, 22.3, 18.4, 18.4, 16.6, 20.5, 20.5, 11.2, 13, 18.4, 20.5, 22.6, 27.7, 21.8, 
#     14.8, 29.5, 11.2, 14.8, 24.1, 22.3, 22.3, 22.3, 37.1, 22.3, 27.7, 20.5, 20.5, 18.4, 
#     18.4, 23.9, 28.8, 9.9, 27.7, 18.4, 13, 29.1, 20.4, 11.2, 18.4, 25.7, 18.3, 18.4, 
#     18.4, 27.7, 20.5, 14.8, 23.3, 14.8, 18.4, 20.5, 18.4, 15.7, 13, 14.8, 18.4, 18.4, 
#     27.7, 22.1, 13, 13, 24.1, 27.7, 23.7, 32.7, 22.3, 20.5, 14.8, 14.8, 18.4, 22.3, 
#     22.3, 18.4, 22.3, 22.3, 15.5, 27.7, 16.6, 16.6, 13, 18.4, 13, 18.4, 11.1, 13, 14.8, 
#     13, 14.8, 14.8, 16.6, 14.8, 16.6, 27.7, 14.8, 14.8, 18.4, 24.1, 18.4, 18.4, 9.4, 
#     13, 16.6, 13, 22.3, 25.9, 20.5, 18.4, 18.7, 18.4, 16.7, 16.6, 14.8, 13, 18.4, 
#     18.4, 14.8, 11.2, 14.8, 28.2, 21.5, 18.4, 14.8, 14.8, 14.8, 14.8, 20.5, 13, 14.8, 
#     14.8, 11.4, 10.8, 16.6, 18.4, 17.8, 20.5, 18.4, 18.4, 23.6, 20.3, 18.4, 11.2, 
#     16.6, 14.8, 14.8, 14.8, 14.8, 15.5, 16.6, 14.8, 9.4, 14.8, 9.4, 16.6, 16.6, 11.2, 
#     11.8, 14.8, 18.3, 14.8, 16.6, 14.1, 16.6, 18.4, 14.8, 14.8, 11.2, 13, 18.4, 20.5, 
#     10.5, 7.6, 27.7, 14.8, 20.5, 16.6, 18.3, 18.4, 13, 18.4, 18.4, 18.4, 14.8, 13, 
#     9.4, 16.6, 12.7, 11.2, 14.8, 9.4, 9.4, 9.4, 11.6, 11.2, 12.3, 13, 10.6, 9.4, 9.4, 
#     9.4, 9.4, 7.7, 11.2, 9.4, 7.6, 11.2, 9.4, 14.4, 24.1, 13, 9.4, 7.6, 7.6, 11.2, 
#     9.4, 13, 11.2, 7.6, 13, 11.2, 11.2, 9.4, 14.8, 16.6, 13, 11.2, 18.4, 13, 9.4, 
#     9.4, 9.4, 16.6, 13, 16.6, 27.7, 16.6, 11.2, 9.4, 16.6, 13, 11.2, 7.6, 11.2, 9.4, 
#     7.6, 14.8, 9.4, 9.4, 9.4, 8, 13, 7.6, 22.3, 14.8, 20.5, 22.3, 16.6, 11.2, 16.6, 
#     10.8, 5.4, 11.2, 14.8, 9.4, 7.6, 14.8, 11.2, 13, 9.3, 20.5, 20.5, 7.6, 13, 14.8, 
#     18.4, 22.3, 14.9, 27.7, 13, 13, 14.8, 22.7, 14.8, 8.2, 10.6, 9.4, 9.4, 14.8, 
#     18.4, 26, 21, 7.6, 10.9, 13, 10.1, 8.6, 13, 12.3, 21.6, 16.6, 27.7, 22.3, 16.1, 
#     20.5, 27.7, 13.2, 14.8, 12.3, 14.4, 16.6, 16.6, 20.5, 10.6, 11.2, 18.4, 18.4, 
#     21.1, 22.3, 22.3, 16.6, 14.8, 18.4, 24.4, 9.4, 14.5, 11.2, 13, 18.4, 25.9, 13, 
#     19.5, 18.4, 27.7, 16.9, 18.4, 18.4, 9.5, 13.2, 14.8, 11.2, 18.4, 18.4, 19.6, 
#     18.4, 15.2, 18.4, 16.6, 18.4, 18.4, 18.4, 14.8, 27.7, 14.8, 20.5, 9.4, 9.4, 
#     18.4, 19.8, 18.4, 27.7, 18.4, 21, 46.4, 16.6, 20.5, 18.4, 16.7, 14.8, 22.3, 
#     22.3, 22.3, 20.5, 27.7, 9.8, 17.9, 11.2, 18.4, 20.5, 22.3, 14.8, 16.6, 18.4, 
#     18.3, 20, 20.6, 13, 27.7, 14.8, 27.7, 26.9, 20.5, 40.7, 63, 13, 11.2, 28.2, 
#     13, 14.8, 16.6, 9.4, 11.2, 10.8, 15.4, 16.6, 17.4, 18, 23.2, 29.6, 16.3, 30.4, 
#     27.4, 21.1, 16.6, 21.4, 23, 23.2, 19.3, 18, 16.6, 14.4, 18.4, 18.4, 20.5, 14.8, 
#     14.8, 24.1, 18.4, 16.6, 18.4, 20.5, 13, 15.3, 15.8, 11.2, 13.7, 18.4, 18.4, 
#     14.8, 22.3, 18.4, 16.9, 14.7, 13, 16.6, 20.5, 22.3, 15.3, 13, 17.1, 22.3, 11.2, 
#     15.4, 18.4, 22.5, 21.5, 20.5, 16.6, 11.2, 11.2, 16.6, 11.2, 13, 29, 14.8, 18.4, 
#     13, 11.2, 27.7, 9.4, 16.6, 15.5, 18.5, 20.5, 16.6, 11.2, 18.4, 37.1, 22.3, 11.9, 
#     17.4, 13, 11.2, 22.3, 27.7, 16.6, 11.2, 14.8, 20.1, 17.9, 18.4, 16.6, 14.8, 11.2, 
#     13, 22.3, 16.2, 13.2, 14.8, 11.2, 9.4, 16.6, 14.8, 13, 18.4, 18.4, 20.9, 11.2, 
#     9.4, 13, 11.2, 16.6, 16.6, 13, 20.5, 11.2, 13, 11.2, 13, 14.8, 13, 13, 16.6, 
#     24.4, 11.2, 9.4, 16.6, 16.9, 18.4, 18.4, 14.8, 9.4, 9.4, 17.6, 13, 13, 7.6, 
#     10.2, 7.6, 9.4, 7.6, 7.6, 13, 9.4, 11.2, 16.6, 7.6, 9.4, 5.4, 9.2, 9.4, 6.4, 
#     7.6, 18.4, 9.4, 7.6, 13, 13, 9.4, 13, 7.6, 13, 20.5, 16.6, 16.6, 11.2
# ]

#     conditions=[
#     "Clear", "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Partially cloudy", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Rain, Partially cloudy", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Clear", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Clear", "Rain, Partially cloudy", "Rain, Partially cloudy", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", 
#     "Clear", "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Clear", "Clear", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Clear", "Partially cloudy", "Rain, Overcast", "Rain, Overcast", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Clear", "Partially cloudy", "Rain", "Rain, Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Rain, Overcast", 
#     "Rain, Overcast", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Overcast", 
#     "Rain, Overcast", "Rain, Partially cloudy", "Rain, Overcast", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Overcast", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Overcast", "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Overcast", 
#     "Rain, Overcast", "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Clear", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Clear", "Clear", "Clear", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Partially cloudy", "Clear", "Partially cloudy", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Clear", 
#     "Rain, Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Partially cloudy", 
#     "Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", 
#     "Clear", "Clear", "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Rain, Overcast", "Partially cloudy", 
#     "Partially cloudy", "Rain, Overcast", "Rain, Overcast", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Overcast", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", "Clear", 
#     "Clear", "Clear", "Partially cloudy", "Partially cloudy", "Clear", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Clear", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Clear", "Clear", "Clear", 
#     "Clear", "Clear", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Clear", "Clear", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Overcast", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Rain, Overcast", "Rain, Overcast", 
#     "Rain, Overcast", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Overcast", "Rain, Overcast", "Rain, Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Partially cloudy", "Partially cloudy", "Rain, Partially cloudy", "Rain, Partially cloudy", 
#     "Rain, Partially cloudy", "Rain, Overcast", "Rain, Overcast", "Rain, Overcast", 
#     "Partially cloudy", "Partially cloudy", "Partially cloudy", "Clear", "Clear", 
#     "Partially cloudy", "Clear", "Clear", "Clear", "Partially cloudy", "Partially cloudy", 
#     "Partially cloudy", "Rain, Overcast", "Rain, Overcast", "Rain, Overcast", 
#     "Rain, Overcast", "Rain, Partially cloudy", "Partially cloudy", "Clear", "Clear", 
#     "Clear"

# ]
    
#     lengths = {
#         'dates': len(dates),
#         'new_cases': len(new_cases),
#         'Temp': len(Temp),
#         'humidity': len(humidity),
#         'wind_speed': len(wind_speed),
#         'conditions': len(conditions)
#     }
#     print("Lengths before truncation:", lengths)

#     min_len = min(lengths.values())

#     # Truncate all arrays to min length
#     dates = dates[:min_len]
#     new_cases = new_cases[:min_len]
#     Temp = Temp[:min_len]
#     humidity = humidity[:min_len]
#     wind_speed = wind_speed[:min_len]
#     conditions = conditions[:min_len]

#     lengths_after = {
#         'dates': len(dates),
#         'new_cases': len(new_cases),
#         'Temp': len(Temp),
#         'humidity': len(humidity),
#         'wind_speed': len(wind_speed),
#         'conditions': len(conditions)
#     }
#     print("Lengths after truncation:", lengths_after)

#     # Create DataFrame
#     df = pd.DataFrame({
#         'ds': dates,
#         'y': new_cases,
#         'Temperature': Temp,
#         'Humidity': humidity,
#         'Wind_Speed': wind_speed,
#         'Conditions': conditions
#     })

#     # Handle wind speed outliers by clipping to 99th percentile
#     wind_speed_99th = np.percentile(df['Wind_Speed'], 99)
#     df['Wind_Speed'] = np.clip(df['Wind_Speed'], a_min=None, a_max=wind_speed_99th)

#     # One-hot encode Conditions
#     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     conditions_encoded = encoder.fit_transform(df[['Conditions']])
#     conditions_df = pd.DataFrame(conditions_encoded, columns=encoder.get_feature_names_out(['Conditions']))
#     df = pd.concat([df, conditions_df], axis=1).drop(columns=['Conditions'])

#     return df, encoder


# # Load data and encoder
# df, encoder = load_data()

# # Split data
# train_end = '2022-10-15'
# val_end = '2022-11-15'
# train_df = df[df['ds'] <= train_end]
# val_df = df[(df['ds'] > train_end) & (df['ds'] <= val_end)]

# # Train Prophet model
# st.subheader("Model Training")
# with st.spinner("Training Prophet model..."):
#     model = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=True,
#         daily_seasonality=False,
#         seasonality_mode='multiplicative'
#     )

#     # Add regressors
#     regressor_cols = ['Temperature', 'Humidity', 'Wind_Speed'] + list(df.columns[df.columns.str.startswith('Conditions_')])
#     for col in regressor_cols:
#         model.add_regressor(col)

#     # Fit model
#     model.fit(train_df)

# # Evaluate on validation set
# val_forecast = model.predict(val_df.drop('y', axis=1))
# mae = mean_absolute_error(val_df['y'], val_forecast['yhat'])
# rmse = np.sqrt(mean_squared_error(val_df['y'], val_forecast['yhat']))
# st.write(f"Validation MAE: {mae:.2f}")
# st.write(f"Validation RMSE: {rmse:.2f}")

# # Forecast next 30 days
# st.subheader("30-Day Forecast (Nov 16 - Dec 15, 2022)")
# future_dates = pd.date_range(start='2022-11-16', end='2022-12-15', freq='D')
# future_df = pd.DataFrame({'ds': future_dates})

# # Assume weather variables repeat last known values (simplified)
# last_weather = df.iloc[-1][regressor_cols]
# for col in regressor_cols:
#     future_df[col] = last_weather[col]

# # Make forecast
# forecast = model.predict(future_df)

# # Plot results
# st.subheader("Forecast Visualization")
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(train_df['ds'], train_df['y'], label='Actual (Train)', color='blue')
# ax.plot(val_df['ds'], val_df['y'], label='Actual (Validation)', color='blue', linestyle='--')
# ax.plot(val_forecast['ds'], val_forecast['yhat'], label='Predicted (Validation)', color='orange')
# ax.plot(forecast['ds'], forecast['yhat'], label='Forecast (Next 30 Days)', color='red')
# ax.set_xlabel('Date')
# ax.set_ylabel('New Cases')
# ax.set_title('COVID-19 New Cases: Actual vs Forecast')
# ax.legend()
# ax.grid(True)
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.setp(ax.get_xticklabels(), rotation=45)
# st.pyplot(fig)

# # Display forecast table
# st.subheader("Forecasted New Cases")
# forecast_table = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted_New_Cases'})
# forecast_table['Predicted_New_Cases'] = forecast_table['Predicted_New_Cases'].round(0).astype(int)
# st.dataframe(forecast_table)

# # Feature importance (approximated by regressor coefficients)
# st.subheader("Feature Importance")
# if hasattr(model, 'params'):
#     regressor_coefs = pd.DataFrame({
#         'Feature': model.extra_regressors.keys(),
#         'Coefficient': model.params['beta'][0][-len(model.extra_regressors):]
#     })
#     st.dataframe(regressor_coefs.sort_values(by='Coefficient', ascending=False))
      


# # Actual cases for Nov 16 - Nov 29, 2022
# actual_dates = pd.date_range(start='2022-11-16', end='2022-11-29', freq='D')
# actual_cases = [501, 635, 656, 556, 492, 406, 294, 360, 408, 347, 389, 343, 291, 215]
# actual_df = pd.DataFrame({'ds': actual_dates, 'y': actual_cases})

# # Create DataFrame for training
# df = pd.DataFrame({'ds': pd.to_datetime(dates, format='%d-%b-%y'), 'y': new_cases})
# df.set_index('ds', inplace=True)

# # Initialize and fit ARIMA model
# st.subheader("ARIMA Model Forecast")
# # Use ARIMA(5,1,0) based on differencing to make series stationary
# model = ARIMA(df['y'], order=(5, 1, 0))
# model_fit = model.fit()

# # Forecast for Nov 16 - Dec 15, 2022 (30 days)
# forecast_steps = 30
# forecast = model_fit.forecast(steps=forecast_steps)
# forecast_dates = pd.date_range(start='2022-11-16', end='2022-12-15', freq='D')

# # Calculate confidence intervals (approximate 95% CI)
# forecast_summary = model_fit.get_forecast(steps=forecast_steps)
# conf_int = forecast_summary.conf_int(alpha=0.05)
# conf_int.columns = ['yhat_lower', 'yhat_upper']

# # Create forecast DataFrame
# forecast_df = pd.DataFrame({
#     'ds': forecast_dates,
#     'yhat': forecast,
#     'yhat_lower': conf_int['yhat_lower'].values,
#     'yhat_upper': conf_int['yhat_upper'].values
# })

# # Filter forecast for comparison with actual data (Nov 16 - Nov 29, 2022)
# comparison = forecast_df[forecast_df['ds'].isin(actual_dates)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# comparison = comparison.merge(actual_df, on='ds', how='left').rename(columns={'y': 'actual'})

# # Calculate error metrics
# mae = np.mean(np.abs(comparison['actual'] - comparison['yhat']))
# mape = np.mean(np.abs((comparison['actual'] - comparison['yhat']) / comparison['actual'])) * 100

# # Calculate total predicted and actual cases
# total_predicted_cases = forecast_df['yhat'].sum()
# lower_bound = forecast_df['yhat_lower'].sum()
# upper_bound = forecast_df['yhat_upper'].sum()
# total_actual_cases = actual_df['y'].sum()

# # Display prediction and error results
# st.subheader("Prediction Results")
# st.write(f"**Predicted Total New Cases (Nov 16 - Dec 15, 2022):** {int(total_predicted_cases):,}")
# st.write(f"**95% Confidence Interval (Nov 16 - Dec 15, 2022):** {int(lower_bound):,} - {int(upper_bound):,}")
# st.write(f"**Actual Total New Cases (Nov 16 - Nov 29, 2022):** {int(total_actual_cases):,}")
# st.write(f"**Mean Absolute Error (MAE, Nov 16 - Nov 29):** {mae:.2f} cases")
# st.write(f"**Mean Absolute Percentage Error (MAPE, Nov 16 - Nov 29):** {mape:.2f}%")

# # Plot historical data, forecast, and actual cases
# st.subheader("Historical Data, Forecast, and Actual Cases Plot")
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(df.index, df['y'], 'b-', label='Historical Cases', linewidth=1.5)
# ax.plot(forecast_df['ds'], forecast_df['yhat'], 'r--', label='Forecast', linewidth=1.5)
# ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='r', alpha=0.1, label='95% CI')
# ax.plot(actual_df['ds'], actual_df['y'], 'g.-', label='Actual Cases', linewidth=1.5, markersize=10)
# ax.set_xlabel('Date')
# ax.set_ylabel('New Cases')
# ax.set_title('Daily New COVID-19 Cases, ARIMA Forecast, and Actual (Jan 2021 - Dec 2022)')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
# ax.tick_params(axis='x', rotation=45)
# ax.grid(True, linestyle='--', alpha=0.7)
# ax.legend()
# plt.tight_layout()
# st.pyplot(fig) 



















# # xgboost
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import xgboost as xgb

# # Load the dataset
# # Assuming the dataset is already provided as mm.csv with Date_reported as index
# df = pd.read_csv('mm2.csv', index_col='Date_reported', parse_dates=True)

# # For demonstration, assuming df is already loaded with the provided data
# # dfconfirmIndia and dfdeathIndia will use the same dataset
# dfconfirmIndia = pd.DataFrame({
#     'Date_reported': pd.to_datetime(df['Date_reported']),
#     'New_cases': df['New_cases'],
#     'Cumulative_cases': df['Cumulative_cases'],
#     'New_deaths': df['New_deaths'],
#     'Cumulative_deaths': df['Cumulative_deaths']
# }).set_index('Date_reported')

# dfdeathIndia = dfconfirmIndia.copy()

# def Confirem(df, str_title, str_forecast):
#     # Drop unnecessary columns (adapt to dataset)
#     df = df.drop(['Cumulative_cases', 'New_deaths', 'Cumulative_deaths'], axis=1)
#     df_full2 = df.copy()

#     # Filter data up to the dataset's end date (July 31, 2022)
#     split_date = '2022-07-31'
#     df = df.loc[df.index <= split_date].copy()

#     # Split into train and test sets
#     split_date_train_test = '2022-05-31'  # Adjusted to have enough test data
#     df_train = df.loc[df.index <= split_date_train_test].copy()
#     df_test = df.loc[df.index > split_date_train_test].copy()

#     def create_features(df, label=None):
#         """
#         Creates time series features from datetime index
#         """
#         df['date'] = df.index
#         df['hour'] = df['date'].dt.hour
#         df['dayofweek'] = df['date'].dt.dayofweek
#         df['quarter'] = df['date'].dt.quarter
#         df['month'] = df['date'].dt.month
#         df['year'] = df['date'].dt.year
#         df['dayofyear'] = df['date'].dt.dayofyear
#         df['dayofmonth'] = df['date'].dt.day
#         df['weekofyear'] = df['date'].dt.isocalendar().week  # Updated for compatibility
        
#         X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
#         if label:
#             y = df[label]
#             return X, y
#         return X

#     # Create features for train, test, and full dataset
#     X_train, y_train = create_features(df_train, label='New_cases')
#     X_test, y_test = create_features(df_test, label='New_cases')
#     X_all, y_all = create_features(df, label='New_cases')

#     # Train XGBoost model
#     reg = xgb.XGBRegressor(n_estimators=1000)
#     reg.fit(X_train, y_train, verbose=False)

#     # Predict on test set
#     df_test['New_cases_Prediction'] = reg.predict(X_test)
#     df_all = pd.concat([df_test, df_train], sort=False)

#     # Plot train, test, and predictions
#     plt.figure(figsize=(16, 9))
#     df_train['New_cases'].plot(legend=True)
#     df_test['New_cases'].plot(legend=True)
#     df_all['New_cases_Prediction'].plot(legend=True)
#     plt.title(str_title, size=30)
#     plt.xlabel('Days', size=30)
#     plt.ylabel('Number of Confirmed Cases', size=30)
#     plt.legend(['Train Case', 'Test Case', 'XGBoost Prediction'], prop={'size': 20})
#     plt.xticks(size=20)
#     plt.yticks(size=20)
#     plt.grid(True, color='k')
#     plt.show()

#     # Predict on full dataset
#     df_full2['fullNew_cases_Prediction'] = reg.predict(X_all)
#     df_full2['fullNew_cases_Prediction'] = df_full2['fullNew_cases_Prediction'].abs().round().astype('int')

#     # Plot full dataset and predictions
#     style.use('fivethirtyeight')
#     plt.figure(figsize=(16, 9))
#     df_full2['New_cases'].plot(legend=True, linestyle='dotted')
#     df_full2['fullNew_cases_Prediction'].plot(legend=True, linestyle='dotted')
#     plt.title(str_title, size=30)
#     plt.xlabel('Days', size=30)
#     plt.ylabel('Number of Confirmed Cases', size=30)
#     plt.legend(['Total Confirmed Cases', 'XGBoost Prediction from Beginning'], prop={'size': 20})
#     plt.xticks(size=20)
#     plt.yticks(size=20)
#     plt.grid(True, color='k')
#     plt.show()

#     # Evaluation metrics for test set
#     mae = mean_absolute_error(y_true=df_test['New_cases'], y_pred=df_test['New_cases_Prediction'])
#     print("MAE: % f" % (mae))
#     print('MSE:', mean_squared_error(y_true=df_test['New_cases'], y_pred=df_test['New_cases_Prediction']))
#     rmse = np.sqrt(mean_squared_error(y_true=df_test['New_cases'], y_pred=df_test['New_cases_Prediction']))
#     print("RMSE: % f" % (rmse))
#     print("R2 Score:", r2_score(y_true=df_test['New_cases'], y_pred=df_test['New_cases_Prediction']))

#     # Evaluation metrics for full dataset
#     mae = mean_absolute_error(y_true=df_full2['New_cases'], y_pred=df_full2['fullNew_cases_Prediction'])
#     print("Full MAE: % f" % (mae))
#     print('Full MSE:', mean_squared_error(y_true=df_full2['New_cases'], y_pred=df_full2['fullNew_cases_Prediction']))
#     rmse = np.sqrt(mean_squared_error(y_true=df_full2['New_cases'], y_pred=df_full2['fullNew_cases_Prediction']))
#     print("Full RMSE: % f" % (rmse))
#     print("Full R2 Score:", r2_score(y_true=df_full2['New_cases'], y_pred=df_full2['fullNew_cases_Prediction']))

#     # Create features for future dates
#     def create_features1(df, target_variable):
#         df['Date'] = pd.to_datetime(df['Date'])
#         df['hour'] = df['Date'].dt.hour
#         df['dayofweek'] = df['Date'].dt.dayofweek
#         df['quarter'] = df['Date'].dt.quarter
#         df['month'] = df['Date'].dt.month
#         df['year'] = df['Date'].dt.year
#         df['dayofyear'] = df['Date'].dt.dayofyear
#         df['dayofmonth'] = df['Date'].dt.day
#         df['weekofyear'] = df['Date'].dt.isocalendar().week
#         X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
#         if target_variable:
#             y = df[target_variable]
#             return X, y
#         return X

#     # Generate future dates for forecasting
#     dti = pd.date_range("2022-11-01", periods=20, freq="D")
#     df_future_dates = pd.DataFrame(dti, columns=['Date'])
#     df_future_dates['Irr'] = np.nan
#     df_future_dates.index = pd.to_datetime(df_future_dates['Date'], format='%Y-%m-%d')
#     testX_future, testY_future = create_features1(df_future_dates, target_variable='Irr')
    
#     # Predict future values
#     yhat1 = reg.predict(testX_future)
#     poly_df = pd.DataFrame({'Date': dti, str_forecast: yhat1})
#     poly_df[str_forecast] = poly_df[str_forecast].abs().round().astype('int')
    
#     # Plot forecasted values
#     poly_df.plot(x='Date', y=str_forecast, figsize=(15, 5), title=str_forecast)
#     plt.show()
    
#     return poly_df

# def Death(df, str_title, str_forecast):
#     # Drop unnecessary columns (adapt to dataset)
#     df = df.drop(['Cumulative_cases', 'New_cases', 'Cumulative_deaths'], axis=1)
#     df_full2 = df.copy()

#     # Filter data up to the dataset's end date (July 31, 2022)
#     split_date = '2022-07-31'
#     df = df.loc[df.index <= split_date].copy()

#     # Split into train and test sets
#     split_date_train_test = '2022-05-31'  # Adjusted to have enough test data
#     df_train = df.loc[df.index <= split_date_train_test].copy()
#     df_test = df.loc[df.index > split_date_train_test].copy()

#     def create_features(df, label=None):
#         """
#         Creates time series features from datetime index
#         """
#         df['date'] = df.index
#         df['hour'] = df['date'].dt.hour
#         df['dayofweek'] = df['date'].dt.dayofweek
#         df['quarter'] = df['date'].dt.quarter
#         df['month'] = df['date'].dt.month
#         df['year'] = df['date'].dt.year
#         df['dayofyear'] = df['date'].dt.dayofyear
#         df['dayofmonth'] = df['date'].dt.day
#         df['weekofyear'] = df['date'].dt.isocalendar().week
        
#         X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
#         if label:
#             y = df[label]
#             return X, y
#         return X

#     # Create features for train, test, and full dataset
#     X_train, y_train = create_features(df_train, label='New_deaths')
#     X_test, y_test = create_features(df_test, label='New_deaths')
#     X_all, y_all = create_features(df, label='New_deaths')

#     # Train XGBoost model
#     reg = xgb.XGBRegressor(n_estimators=1000)
#     reg.fit(X_train, y_train, verbose=False)

#     # Predict on test set
#     df_test['New_deaths_Prediction'] = reg.predict(X_test)
#     df_all = pd.concat([df_test, df_train], sort=False)

#     # Plot train, test, and predictions
#     plt.figure(figsize=(16, 9))
#     df_train['New_deaths'].plot(legend=True)
#     df_test['New_deaths'].plot(legend=True)
#     df_all['New_deaths_Prediction'].plot(legend=True)
#     plt.title(str_title, size=30)
#     plt.xlabel('Days', size=30)
#     plt.ylabel('Number of Death Cases', size=30)
#     plt.legend(['Train Case', 'Test Case', 'XGBoost Prediction'], prop={'size': 20})
#     plt.xticks(size=20)
#     plt.yticks(size=20)
#     plt.grid(True, color='k')
#     plt.show()

#     # Predict on full dataset
#     df_full2['fullNew_deaths_Prediction'] = reg.predict(X_all)
#     df_full2['fullNew_deaths_Prediction'] = df_full2['fullNew_deaths_Prediction'].abs().round().astype('int')

#     # Plot full dataset and predictions
#     style.use('fivethirtyeight')
#     plt.figure(figsize=(16, 9))
#     df_full2['New_deaths'].plot(legend=True, linestyle='dotted')
#     df_full2['fullNew_deaths_Prediction'].plot(legend=True, linestyle='dotted')
#     plt.title(str_title, size=30)
#     plt.xlabel('Days', size=30)
#     plt.ylabel('Number of Death Cases', size=30)
#     plt.legend(['Total Death Cases', 'XGBoost Prediction from Beginning'], prop={'size': 20})
#     plt.xticks(size=20)
#     plt.yticks(size=20)
#     plt.grid(True, color='k')
#     plt.show()

#     # Evaluation metrics for test set
#     mae = mean_absolute_error(y_true=df_test['New_deaths'], y_pred=df_test['New_deaths_Prediction'])
#     print("MAE: % f" % (mae))
#     print('MSE:', mean_squared_error(y_true=df_test['New_deaths'], y_pred=df_test['New_deaths_Prediction']))
#     rmse = np.sqrt(mean_squared_error(y_true=df_test['New_deaths'], y_pred=df_test['New_deaths_Prediction']))
#     print("RMSE: % f" % (rmse))
#     print("R2 Score:", r2_score(y_true=df_test['New_deaths'], y_pred=df_test['New_deaths_Prediction']))

#     # Evaluation metrics for full dataset
#     mae = mean_absolute_error(y_true=df_full2['New_deaths'], y_pred=df_full2['fullNew_deaths_Prediction'])
#     print("Full MAE: % f" % (mae))
#     print('Full MSE:', mean_squared_error(y_true=df_full2['New_deaths'], y_pred=df_full2['fullNew_deaths_Prediction']))
#     rmse = np.sqrt(mean_squared_error(y_true=df_full2['New_deaths'], y_pred=df_full2['fullNew_deaths_Prediction']))
#     print("Full RMSE: % f" % (rmse))
#     print("Full R2 Score:", r2_score(y_true=df_full2['New_deaths'], y_pred=df_full2['fullNew_deaths_Prediction']))

#     # Create features for future dates
#     def create_features1(df, target_variable):
#         df['Date'] = pd.to_datetime(df['Date'])
#         df['hour'] = df['Date'].dt.hour
#         df['dayofweek'] = df['Date'].dt.dayofweek
#         df['quarter'] = df['Date'].dt.quarter
#         df['month'] = df['Date'].dt.month
#         df['year'] = df['Date'].dt.year
#         df['dayofyear'] = df['Date'].dt.dayofyear
#         df['dayofmonth'] = df['Date'].dt.day
#         df['weekofyear'] = df['Date'].dt.isocalendar().week
#         X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
#         if target_variable:
#             y = df[target_variable]
#             return X, y
#         return X

#     # Generate future dates for forecasting
#     dti = pd.date_range("2022-11-01", periods=20, freq="D")
#     df_future_dates = pd.DataFrame(dti, columns=['Date'])
#     df_future_dates['Irr'] = np.nan
#     df_future_dates.index = pd.to_datetime(df_future_dates['Date'], format='%Y-%m-%d')
#     testX_future, testY_future = create_features1(df_future_dates, target_variable='Irr')
    
#     # Predict future values
#     yhat1 = reg.predict(testX_future)
#     poly_df = pd.DataFrame({'Date': dti, str_forecast: yhat1})
#     poly_df[str_forecast] = poly_df[str_forecast].abs().round().astype('int')
    
#     # Plot forecasted values
#     poly_df.plot(x='Date', y=str_forecast, figsize=(15, 5), title=str_forecast)
#     plt.show()
    
#     return poly_df

# # Execute the functions
# forecasted_cases = Confirem(dfconfirmIndia, 'Number of Coronavirus Cases Over Time in India', 'Forecasted Confirmed Cases in India')
# forecasted_deaths = Death(dfdeathIndia, 'Number of Coronavirus Deaths Over Time in India', 'Forecasted Death Cases in India')

# # Print forecasted results
# print("Forecasted Confirmed Cases:")
# print(forecasted_cases)
# print("\nForecasted Death Cases:")
# print(forecasted_deaths)




# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
# from datetime import datetime, timedelta
# import warnings
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import LSTM, Dense # type: ignore
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from statsmodels.stats.diagnostic import acorr_ljungbox

# warnings.filterwarnings("ignore")

# # Custom CSS for styling with theme support
# def apply_theme(theme):
#     if theme == "Dark":
#         css = """
#             <style>
#             .main { background-color: #2c3e50; color: #ecf0f1; font-family: 'Poppins', sans-serif; }
#             .stButton>button { background-color: #ff4b5c; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; transition: 0.3s; }
#             .stButton>button:hover { background-color: #e04352; box-shadow: 0 2px 5px rgba(255,255,255,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(255,255,255,0.1); }
#             .stMetric { background-color: #34495e; color: #ecf0f1; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(255,255,255,0.1); margin: 10px 0; }
#             .card { background-color: #34495e; color: #ecf0f1; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 5px rgba(255,255,255,0.1); }
#             h1, h2, h3, h4 { color: #ecf0f1; }
#             .sidebar .sidebar-content { background-color: #2c3e50; color: #ecf0f1; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: #34495e; }
#             .dataframe th { background-color: #1a73e8; color: white; padding: 12px; }
#             .dataframe td { padding: 12px; border-bottom: 1px solid #ddd; }
#             .dataframe tr:nth-child(even) { background-color: #f9f9f9; }
#             .dataframe tr:hover { background-color: #e6f0fa; }
#             @keyframes pulse {
#                 0% { transform: scale(1); }
#                 50% { transform: scale(1.1); }
#                 100% { transform: scale(1); }
#             }
#             .stButton>button { animation: pulse 2s infinite; }
#             @keyframes fadeIn {
#                 0% { opacity: 0; transform: translateY(20px); }
#                 100% { opacity: 1; transform: translateY(0); }
#             }
#             .fade-in { animation: fadeIn 1s ease-in-out; }
#             </style>
#         """
#     else:  # Light theme
#         css = """
#             <style>
#             .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Poppins', sans-serif; }
#             .stButton>button { background-color: #1a73e8; color: white; border-radius: 25px; border: none; padding: 10px 30px; font-weight: bold; transition: 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#             .stButton>button:hover { background-color: #ff6f61; transform: scale(1.05); box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
#             .stSelectbox, .stFileUploader, .stDateInput, .stTextInput { background-color: white; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#             .stMetric { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 10px 0; }
#             .card { background-color: white; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
#             h1, h2, h3, h4 { color: #2c3e50; }
#             .sidebar .sidebar-content { background-color: #34495e; color: white; }
#             .sidebar .stButton>button { background-color: #1abc9c; }
#             .sidebar .stButton>button:hover { background-color: #16a085; }
#             .plotly-chart { border-radius: 10px; overflow: hidden; background-color: white; }
#             .dataframe { border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; }
#             .dataframe th { background-color: #1a73e8; color: white; padding: 12px; }
#             .dataframe td { padding: 12px; border-bottom: 1px solid #ddd; }
#             .dataframe tr:nth-child(even) { background-color: #f9f9f9; }
#             .dataframe tr:hover { background-color: #e6f0fa; }
#             @keyframes pulse {
#                 0% { transform: scale(1); }
#                 50% { transform: scale(1.1); }
#                 100% { transform: scale(1); }
#             }
#             .stButton>button { animation: pulse 2s infinite; }
#             @keyframes fadeIn {
#                 0% { opacity: 0; transform: translateY(20px); }
#                 100% { opacity: 1; transform: translateY(0); }
#             }
#             .fade-in { animation: fadeIn 1s ease-in-out; }
#             </style>
#         """
#     st.markdown(css, unsafe_allow_html=True)

# # Initialize session state
# if 'df_covid' not in st.session_state:
#     st.session_state.df_covid = None
# if 'df_malaria' not in st.session_state:
#     st.session_state.df_malaria = None
# if 'user_profile' not in st.session_state:
#     st.session_state.user_profile = {'name': '', 'email': '', 'analyses_run': 0}
# if 'settings' not in st.session_state:
#     st.session_state.settings = {'theme': 'Light', 'look_back_period': 60}

# # Apply theme
# apply_theme(st.session_state.settings['theme'])

# # Sidebar navigation
# st.sidebar.title("🦠 Disease Prediction Dashboard")
# st.sidebar.markdown("Analyze and predict COVID-19 and malaria cases in India.")
# page = st.sidebar.radio("Go to:", ["Profile", "Home", "COVID-19 Analysis","COVID-19 2022", "Malaria Analysis", "Combined Insights", "Settings"])

# # Header
# st.markdown("""
#     <div class="card">
#         <h1>🦠 Disease Prediction Dashboard</h1>
#         <p style="font-size: 18px;">Analyze and predict COVID-19 and malaria cases across India using LSTM, ARIMA, and ARIMAX models.</p>
#     </div>
# """, unsafe_allow_html=True)

# # LSTM Forecasting Function
# @st.cache_data
# def lstm_forecast(series, steps, look_back=None):
#     try:
#         look_back = look_back if look_back is not None else st.session_state.settings['look_back_period']
#         if series is None or len(series) == 0:
#             raise ValueError("Input series is None or empty")
#         if series.isna().all():
#             raise ValueError("Input series contains only NaN values")
#         if len(series) < look_back:
#             raise ValueError(f"Input series has {len(series)} data points, but look_back={look_back} is required")

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         series_values = series.values.reshape(-1, 1)
#         scaled_series = scaler.fit_transform(series_values)

#         def create_sequences(data, look_back):
#             X, y = [], []
#             for i in range(len(data) - look_back):
#                 X.append(data[i:(i + look_back), 0])
#                 y.append(data[i + look_back, 0])
#             return np.array(X), np.array(y)

#         X, y = create_sequences(scaled_series, look_back)
#         if len(X) == 0 or len(y) == 0:
#             st.warning(f"No sequences created. Series length={len(series)}, look_back={look_back}. Returning flat forecast.")
#             return pd.Series(
#                 [series.iloc[-1] if not pd.isna(series.iloc[-1]) else 0] * steps,
#                 index=pd.date_range(start=series.index[-1] + timedelta(days=1), periods=steps, freq='D')
#             ), float('nan'), float('nan')

#         X = X.reshape((X.shape[0], X.shape[1], 1))
#         train_size = int(len(X) * 0.8)
#         if train_size == 0:
#             raise ValueError("Training set is empty after splitting. Increase data size or reduce look_back.")

#         X_train, X_val = X[:train_size], X[train_size:]
#         y_train, y_val = y[:train_size], y[train_size:]

#         model = Sequential()
#         model.add(LSTM(50, activation='relu', input_shape=(look_back, 1), return_sequences=True))
#         model.add(LSTM(50, activation='relu'))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_val, y_val))

#         last_sequence = scaled_series[-look_back:].reshape((1, look_back, 1))
#         forecast = []
#         current_sequence = last_sequence.copy()
#         for _ in range(steps):
#             pred = model.predict(current_sequence, verbose=0)
#             forecast.append(pred[0, 0])
#             current_sequence = np.roll(current_sequence, -1, axis=1)
#             current_sequence[0, -1, 0] = pred[0, 0]

#         forecast = np.array(forecast).reshape(-1, 1)
#         forecast = scaler.inverse_transform(forecast).flatten()
#         forecast_series = pd.Series(
#             forecast,
#             index=pd.date_range(start=series.index[-1] + timedelta(days=1), periods=steps, freq='D')
#         )

#         mse, r2 = float('nan'), float('nan')
#         if len(X_val) > 0:
#             val_pred = model.predict(X_val, verbose=0)
#             val_pred = scaler.inverse_transform(val_pred).flatten()
#             val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
#             mse = mean_squared_error(val_true, val_pred)
#             r2 = r2_score(val_true, val_pred)

#         return forecast_series.clip(lower=0), mse, r2
#     except Exception as e:
#         st.error(f"Error in LSTM forecasting: {str(e)}")
#         default_start_date = pd.to_datetime('2025-05-27') if series is None or series.empty else series.index[-1]
#         return pd.Series(
#             [0] * steps,
#             index=pd.date_range(start=default_start_date + timedelta(days=1), periods=steps, freq='D')
#         ), float('nan'), float('nan')

# # COVID-19 Data Parsing
# @st.cache_data
# def parse_covid_data(raw_data):
#     try:
#         lines = raw_data.strip().split('\n')
#         headers = lines[0].split()
#         data = [line.split(maxsplit=len(headers)-1) for line in lines[1:]]
#         df = pd.DataFrame(data, columns=headers)
#         df['Date_reported'] = pd.to_datetime(df['Date_reported'], format='%d-%m-%Y')
#         for col in ['New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
#         df['New_cases'] = df['New_cases'].clip(lower=0)
#         df['New_deaths'] = df['New_deaths'].clip(lower=0)
#         return df
#     except Exception as e:
#         st.error(f"Error parsing COVID-19 data: {str(e)}")
#         return None

# # COVID-19 Data Simulation
# @st.cache_data
# def generate_simulated_covid_data(initial_data):
#     try:
#         df = initial_data.copy()
#         last_date = df['Date_reported'].max()
#         last_cumulative_cases = df['Cumulative_cases'].iloc[-1]
#         last_cumulative_deaths = df['Cumulative_deaths'].iloc[-1]
#         end_date = pd.to_datetime('2025-05-27')  # Updated to current date

#         def generate_wave(start_day, peak_day, end_day, start_cases, peak_cases):
#             days = []
#             for day in range(start_day, peak_day + 1):
#                 progress = (day - start_day) / (peak_day - start_day)
#                 cases = int(start_cases + (peak_cases - start_cases) * progress * progress)
#                 days.append(cases)
#             for day in range(peak_day + 1, end_day + 1):
#                 progress = (day - peak_day) / (end_day - peak_day)
#                 cases = int(peak_cases * (1 - progress * progress))
#                 days.append(cases)
#             return days

#         wave1 = generate_wave(0, 150, 300, 100, 100000)
#         wave2 = generate_wave(0, 40, 100, 30000, 400000)
#         wave3 = generate_wave(0, 30, 90, 10000, 350000)
#         wave4 = generate_wave(0, 20, 60, 5000, 50000)
#         wave5 = generate_wave(0, 15, 45, 2000, 25000)
#         wave6 = generate_wave(0, 15, 40, 1000, 15000)
#         wave7 = generate_wave(0, 10, 30, 500, 8000)

#         def get_new_cases(date):
#             year = date.year
#             month = date.month
#             total_days = (date - pd.to_datetime('2020-03-01')).days
#             if year == 2020 or (year == 2021 and month < 3):
#                 day_in_wave = total_days % len(wave1)
#                 return wave1[day_in_wave]
#             elif year == 2021 and 3 <= month <= 6:
#                 day_in_wave = (total_days - (pd.to_datetime('2021-03-01') - pd.to_datetime('2020-03-01')).days) % len(wave2)
#                 return wave2[day_in_wave]
#             elif (year == 2021 and month >= 12) or (year == 2022 and month <= 2):
#                 day_in_wave = (total_days - (pd.to_datetime('2021-12-01') - pd.to_datetime('2020-03-01')).days) % len(wave3)
#                 return wave3[day_in_wave]
#             elif year == 2022 and 6 <= month <= 8:
#                 day_in_wave = (total_days - (pd.to_datetime('2022-06-01') - pd.to_datetime('2020-03-01')).days) % len(wave4)
#                 return wave4[day_in_wave]
#             elif year == 2023 and 1 <= month <= 3:
#                 day_in_wave = (total_days - (pd.to_datetime('2023-01-01') - pd.to_datetime('2020-03-01')).days) % len(wave5)
#                 return wave5[day_in_wave]
#             elif year == 2023 and 9 <= month <= 11:
#                 day_in_wave = (total_days - (pd.to_datetime('2023-09-01') - pd.to_datetime('2020-03-01')).days) % len(wave6)
#                 return wave6[day_in_wave]
#             elif year == 2024 and 3 <= month <= 5:
#                 day_in_wave = (total_days - (pd.to_datetime('2024-03-01') - pd.to_datetime('2020-03-01')).days) % len(wave7)
#                 return wave7[day_in_wave]
#             elif year == 2024 and 10 <= month <= 12:
#                 day_in_wave = (total_days - (pd.to_datetime('2024-10-01') - pd.to_datetime('2020-03-01')).days) % len(wave7)
#                 return int(wave7[day_in_wave] * 0.7)
#             elif year == 2025 and 1 <= month <= 5:
#                 day_in_wave = (total_days - (pd.to_datetime('2025-01-01') - pd.to_datetime('2020-03-01')).days) % len(wave7)
#                 return int(wave7[day_in_wave] * 0.5)
#             else:
#                 return np.random.randint(400, 500)

#         current_date = last_date + timedelta(days=1)
#         new_rows = []
#         while current_date <= end_date:
#             new_cases = get_new_cases(current_date)
#             last_cumulative_cases += new_cases
#             death_rate = 0.02 if current_date <= pd.to_datetime('2021-03-01') else \
#                          0.012 if current_date <= pd.to_datetime('2021-12-01') else \
#                          0.005 if current_date <= pd.to_datetime('2022-07-01') else \
#                          0.002 if current_date <= pd.to_datetime('2023-01-01') else \
#                          0.001 if current_date <= pd.to_datetime('2024-01-01') else 0.0005
#             new_deaths = int(new_cases * death_rate)
#             last_cumulative_deaths += new_deaths
#             new_rows.append({
#                 'Date_reported': current_date,
#                 'Country': 'India',
#                 'WHO_region': 'SEAR',
#                 'New_cases': new_cases,
#                 'Cumulative_cases': last_cumulative_cases,
#                 'New_deaths': new_deaths,
#                 'Cumulative_deaths': last_cumulative_deaths
#             })
#             current_date += timedelta(days=1)
#         full_data = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
#         full_data = full_data.sort_values('Date_reported')
#         return full_data
#     except Exception as e:
#         st.error(f"Error generating simulated COVID-19 data: {str(e)}")
#         return None

# # Malaria Data Preprocessing
# @st.cache_data
# def preprocess_malaria_data(uploaded_file):
#     try:
#         df_malaria = pd.read_csv(uploaded_file)
#         if df_malaria.empty:
#             return None
#         merged_rows = []
#         skip_next = False
#         for i in range(len(df_malaria)):
#             if skip_next:
#                 skip_next = False
#                 continue
#             row = df_malaria.iloc[i].fillna("")
#             if str(row[1]).strip() in ["", "0"] and i > 0:
#                 prev_row = df_malaria.iloc[i - 1].fillna("")
#                 merged = prev_row.combine_first(row)
#                 merged_rows[-1] = merged
#             else:
#                 merged_rows.append(row)
#         df_malaria = pd.DataFrame(merged_rows).reset_index(drop=True)
#         df_malaria.columns = df_malaria.iloc[0]
#         df_malaria = df_malaria[1:].reset_index(drop=True)
#         df_malaria = df_malaria.iloc[:-1]
#         df_malaria.columns = [
#             "Sr", "STATE_UT", "BSE_2020", "Malaria_2020", "Pf_2020", "Deaths_2020",
#             "BSE_2021", "Malaria_2021", "Pf_2021", "Deaths_2021",
#             "BSE_2022", "Malaria_2022", "Pf_2022", "Deaths_2022",
#             "BSE_2023", "Malaria_2023", "Pf_2023", "Deaths_2023",
#             "BSE_2024", "Malaria_2024", "Pf_2024", "Deaths_2024"
#         ]
#         num_cols = df_malaria.columns[2:]
#         df_malaria[num_cols] = df_malaria[num_cols].apply(pd.to_numeric, errors="coerce")
#         df_malaria.drop("Sr", axis=1, inplace=True)
#         return df_malaria
#     except Exception as e:
#         st.error(f"Error preprocessing malaria data: {str(e)}")
#         return None

# # Malaria Visualizations
# @st.cache_data
# def generate_malaria_visualizations(df_malaria):
#     try:
#         visualizations = {}
#         numeric_cols = df_malaria.select_dtypes(include=[np.number]).columns
#         df_numeric = df_malaria[numeric_cols]
#         corr_matrix = df_numeric.corr()
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig1, ax1 = plt.subplots(figsize=(12, 8))
#         sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Spectral", linewidths=0.5, ax=ax1)
#         ax1.set_title("Correlation Matrix for Malaria Cases", fontsize=14, pad=15)
#         visualizations['corr_matrix'] = fig1

#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig2, ax2 = plt.subplots(figsize=(12, 6))
#         palette = sns.color_palette("Set2", 6)
#         for i, year in enumerate(range(2020, 2025)):
#             col = f"Malaria_{year}"
#             if col in df_malaria.columns:
#                 sns.kdeplot(df_malaria[col], label=f"{year} (Actual)", fill=True, alpha=0.5, color=palette[i], ax=ax2)
#         if "Predicted_Malaria_2025" in df_malaria.columns:
#             sns.kdeplot(df_malaria["Predicted_Malaria_2025"], label="2025 (Forecast)", fill=True, alpha=0.5, 
#                         color=palette[5], linestyle='--', ax=ax2)
#         ax2.set_xlabel("Number of Malaria Cases", fontsize=12)
#         ax2.set_ylabel("Density", fontsize=12)
#         ax2.legend(title="Year")
#         ax2.set_title("Malaria Cases Distribution (2020-2025)", fontsize=14, pad=15)
#         visualizations['kde_plot'] = fig2

#         malaria_cols = [col for col in df_malaria.columns if "Malaria_" in col]
#         df_malaria['Avg_Malaria'] = df_malaria[malaria_cols].mean(axis=1)
#         top_states = df_malaria.nlargest(10, 'Avg_Malaria')[['STATE_UT', 'Avg_Malaria']]
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig3, ax3 = plt.subplots(figsize=(10, 6))
#         sns.barplot(data=top_states, y='STATE_UT', x='Avg_Malaria', palette='magma', ax=ax3)
#         ax3.set_title("Top 10 States by Average Malaria Cases (2020-2024)", fontsize=14, pad=15)
#         ax3.set_xlabel("Average Malaria Cases", fontsize=12)
#         ax3.set_ylabel("State/UT", fontsize=12)
#         visualizations['top_states'] = fig3
#         return visualizations
#     except Exception as e:
#         st.error(f"Error generating malaria visualizations: {str(e)}")
#         return {}

# # Malaria Feature Engineering
# @st.cache_data
# def perform_malaria_feature_engineering(df_malaria):
#     try:
#         df_malaria = df_malaria.copy()
#         malaria_cols = [f"Malaria_{y}" for y in range(2020, 2025)]
#         df_malaria["Avg_Malaria_Cases"] = df_malaria[malaria_cols].mean(axis=1)
#         for year in range(2020, 2025):
#             df_malaria[f"Fatality_Rate_{year}"] = np.where(
#                 df_malaria[f"Malaria_{year}"] > 0,
#                 df_malaria[f"Deaths_{year}"] / df_malaria[f"Malaria_{year}"],
#                 0
#             )
#         for year in range(2021, 2025):
#             df_malaria[f"Malaria_Lag_{year}"] = df_malaria[f"Malaria_{year-1}"]
#         def categorize_risk(row):
#             avg = row["Avg_Malaria_Cases"]
#             return "High-Risk" if avg > 15000 else "Medium-Risk" if avg > 5000 else "Low-Risk"
#         df_malaria["Risk_Category"] = df_malaria.apply(categorize_risk, axis=1)
#         df_malaria["Avg_Malaria_Cases"] = df_malaria["Avg_Malaria_Cases"].round(0).astype(int)
#         visualizations = {}
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig4, ax4 = plt.subplots(figsize=(8, 5))
#         sns.countplot(data=df_malaria, x='Risk_Category', palette='Set2', ax=ax4)
#         ax4.set_title("Risk Category Distribution", fontsize=14, pad=15)
#         ax4.set_xlabel("Risk Category", fontsize=12)
#         ax4.set_ylabel("Count", fontsize=12)
#         visualizations['risk_category'] = fig4
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig5, ax5 = plt.subplots(figsize=(15, 8))
#         palette = {"High-Risk": "#e74c3c", "Medium-Risk": "#f39c12", "Low-Risk": "#2ecc71"}
#         sns.barplot(data=df_malaria, x="STATE_UT", y="Avg_Malaria_Cases", hue="Risk_Category",
#                     palette=palette, hue_order=["High-Risk", "Medium-Risk", "Low-Risk"], ax=ax5)
#         ax5.set_title("State-wise Malaria Risk Categorization", fontsize=14, pad=15)
#         ax5.set_xlabel("State/UT", fontsize=12)
#         ax5.set_ylabel("Avg. Malaria Cases", fontsize=12)
#         ax5.tick_params(axis='x', rotation=90)
#         visualizations['state_risk'] = fig5
#         return df_malaria, visualizations
#     except Exception as e:
#         st.error(f"Error in malaria feature engineering: {str(e)}")
#         return df_malaria, {}

# # Malaria ARIMA Predictions
# @st.cache_data
# def run_malaria_arima(df_malaria):
#     try:
#         df_malaria = df_malaria.copy()
#         malaria_cols = [f"Malaria_{y}" for y in range(2020, 2025)]
#         predictions = []
#         mse_total = 0
#         actuals = []
#         forecasts = []
#         for index, row in df_malaria.iterrows():
#             series = [row[col] for col in malaria_cols]
#             series = [0 if pd.isna(x) else x for x in series]
#             train = series[:4]
#             test = series[4]
#             try:
#                 model = ARIMA(train, order=(1, 1, 0))
#                 model_fit = model.fit()
#                 forecast = model_fit.forecast(steps=2)
#                 pred_2024 = forecast[0]
#                 pred_2025 = forecast[1]
#                 predictions.append(pred_2025 if pred_2025 > 0 else 0)
#                 mse_total += (test - pred_2024) ** 2
#                 actuals.append(test)
#                 forecasts.append(pred_2024)
#             except Exception:
#                 predictions.append(series[-1] if series[-1] > 0 else 0)
#         df_malaria["Predicted_Malaria_2025"] = predictions
#         mse = mse_total / len(df_malaria) if len(df_malaria) > 0 else float('nan')
#         mean_actual = np.mean(actuals) if actuals else float('nan')
#         ss_tot = sum((a - mean_actual) ** 2 for a in actuals) if actuals else float('nan')
#         ss_res = sum((a - f) ** 2 for a, f in zip(actuals, forecasts)) if actuals else float('nan')
#         r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
#         metrics = {
#             "mse": mse,
#             "r2": r2,
#             "accuracy": r2 * 100 if not np.isnan(r2) else float('nan')
#         }
#         return df_malaria, metrics
#     except Exception as e:
#         st.error(f"Error in ARIMA forecasting: {str(e)}")
#         return df_malaria, {"mse": float('nan'), "r2": float('nan'), "accuracy": float('nan')}

# # Risk Classification
# def classify_risk(cases, disease="malaria"):
#     if disease == "malaria":
#         if cases > 50000:
#             return ("🔴 High Risk", "#e74c3c")
#         elif cases > 25000:
#             return ("🟠 Medium Risk", "#f39c12")
#         elif cases > 10000:
#             return ("🟡 Moderate Risk", "#f1c40f")
#         else:
#             return ("🟢 Low Risk", "#2ecc71")
#     else:  # COVID-19
#         if cases > 10000:
#             return ("🔴 High Risk", "#e74c3c")
#         elif cases > 5000:
#             return ("🟠 Medium Risk", "#f39c12")
#         elif cases > 1000:
#             return ("🟡 Moderate Risk", "#f1c40f")
#         else:
#             return ("🟢 Low Risk", "#2ecc71")

# # Format Number
# def format_number(num):
#     if pd.isna(num):
#         return "N/A"
#     if num >= 1_000_000:
#         return f"{num / 1_000_000:.1f}M"
#     elif num >= 1_000:
#         return f"{num / 1_000:.1f}K"
#     return int(num)

# # Page: Home
# if page == "Home":
#     st.markdown("""
#         <div class="card">
#             <h2>Welcome to the Disease Prediction Dashboard</h2>
#             <p style="color: #7f8c8d;">
#                 Upload datasets for COVID-19 and/or malaria to analyze and predict cases across India using LSTM, ARIMA, and ARIMAX models.
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

#     st.subheader("Upload Datasets")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**COVID-19 Data**")
#         st.info("COVID-19 data is preloaded from January 2020 to May 2025.")
#         if st.button("🚀 Process COVID-19 Data"):
#             with st.spinner("Processing COVID-19 data..."):
#                 try:
#                     raw_data = """Date_reported Country WHO_region New_cases Cumulative_cases New_deaths Cumulative_deaths
# 04-01-2020 India SEAR 0 0 0 0
# 05-01-2020 India SEAR 1 1 0 0
# 06-01-2020 India SEAR 2 3 1 1
# """
#                     df_covid = parse_covid_data(raw_data)
#                     if df_covid is None:
#                         st.error("Failed to parse COVID-19 data.")
#                         st.stop()
#                     full_covid_data = generate_simulated_covid_data(df_covid)
#                     if full_covid_data is None or full_covid_data['New_cases'].isna().any() or (full_covid_data['New_cases'] < 0).any():
#                         st.error("Simulated COVID-19 data contains invalid values (NaN or negative).")
#                         st.stop()
#                     st.session_state.df_covid = full_covid_data
#                     st.session_state.user_profile['analyses_run'] += 1
#                     st.success("✅ COVID-19 data processed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing COVID-19 data: {str(e)}")
#                     st.stop()
#     with col2:
#         st.markdown("**Malaria Data**")
#         uploaded_file = st.file_uploader("📂 Upload malaria dataset (CSV format)", type=["csv"], key="malaria_data_uploader")
#         if uploaded_file is not None and st.button("🚀 Preprocess Malaria Dataset"):
#             with st.spinner("Processing malaria dataset..."):
#                 try:
#                     df_malaria = preprocess_malaria_data(uploaded_file)
#                     if df_malaria is None:
#                         st.error("Failed to process malaria dataset.")
#                         st.stop()
#                     st.session_state.df_malaria = df_malaria
#                     st.session_state.user_profile['analyses_run'] += 1
#                     st.success("✅ Malaria dataset preprocessed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing malaria dataset: {str(e)}")
#                     st.stop()


# # Page: COVID-19 Analysis
# if page == "COVID-19 Analysis":
#     if st.session_state.df_covid is None:
#         st.warning("COVID-19 data is not loaded. Please go to the Home page and process the COVID-19 data.")
#         st.stop()
    
#     st.markdown("<div class='card'><h2>COVID-19 Case and Death Forecast</h2></div>", unsafe_allow_html=True)
#     df_covid = st.session_state.df_covid
#     current_datetime = datetime(2025, 5, 27, 16, 13)  # Updated to current date and time
#     st.write(f"**Current Date and Time:** {current_datetime.strftime('%A, %B %d, %Y %I:%M %p IST')}")
#     st.markdown("""
#         Showing historical data from January 2020 to May 27, 2025, with a forecast for the user-specified date range.
#         **Note**: Forecast uses LSTM with a configurable look-back period. Actual numbers may vary due to changes in testing rates, public health measures, new variants, or vaccination status.
#     """)

#     historical_df = df_covid.set_index('Date_reported')[['New_cases', 'New_deaths', 'Cumulative_cases', 'Cumulative_deaths']]
#     last_historical_date = historical_df.index.max()

#     if historical_df.empty or historical_df['New_cases'].isna().all() or len(historical_df) < st.session_state.settings['look_back_period']:
#         st.error(f"Insufficient or invalid COVID-19 data for forecasting. Please ensure the dataset has at least {st.session_state.settings['look_back_period']} days of valid data.")
#         st.stop()

#     st.subheader("Select Forecast Date Range")
#     col1, col2 = st.columns(2)
#     with col1:
#         forecast_start_date = st.date_input(
#             "Forecast Start Date",
#             value=pd.to_datetime('2025-05-28'),
#             min_value=(last_historical_date + timedelta(days=1)).date(),
#             max_value=pd.to_datetime('2030-12-31').date()
#         )
#     with col2:
#         forecast_end_date = st.date_input(
#             "Forecast End Date",
#             value=pd.to_datetime('2025-06-26'),
#             min_value=(pd.to_datetime(forecast_start_date) + timedelta(days=1)).date(),
#             max_value=pd.to_datetime('2030-12-31').date()
#         )

#     forecast_start_date = pd.to_datetime(forecast_start_date)
#     forecast_end_date = pd.to_datetime(forecast_end_date)

#     if forecast_end_date <= forecast_start_date:
#         st.error("End date must be after start date.")
#         st.stop()
#     if forecast_start_date <= last_historical_date:
#         st.error(f"Start date must be after the last historical date ({last_historical_date.strftime('%Y-%m-%d')}).")
#         st.stop()

#     forecast_steps = (forecast_end_date - last_historical_date).days
#     if forecast_steps <= 0:
#         st.error("Forecast period must be in the future.")
#         st.stop()

#     with st.spinner("Generating COVID-19 forecast..."):
#         historical_days = len(historical_df['New_cases'])
#         days_to_use = min(365, historical_days)
#         if days_to_use < st.session_state.settings['look_back_period']:
#             st.error(f"Insufficient data for LSTM forecasting. Only {days_to_use} days available, need at least {st.session_state.settings['look_back_period']}.")
#             st.stop()

#         forecast_dates = pd.date_range(start=last_historical_date + timedelta(days=1), periods=forecast_steps, freq='D')
#         forecast_df = pd.DataFrame(index=forecast_dates)

#         forecast_cases, mse_cases, r2_cases = lstm_forecast(historical_df['New_cases'][-days_to_use:], forecast_steps)
#         forecast_deaths, mse_deaths, r2_deaths = lstm_forecast(historical_df['New_deaths'][-days_to_use:], forecast_steps)
#         st.session_state.user_profile['analyses_run'] += 1

#         forecast_df['New_cases'] = forecast_cases.values
#         forecast_df['New_deaths'] = forecast_deaths.values
#         last_cumulative_cases = historical_df['Cumulative_cases'].iloc[-1]
#         last_cumulative_deaths = historical_df['Cumulative_deaths'].iloc[-1]
#         forecast_df['Cumulative_cases'] = last_cumulative_cases + forecast_df['New_cases'].cumsum()
#         forecast_df['Cumulative_deaths'] = last_cumulative_deaths + forecast_df['New_deaths'].cumsum()
#         forecast_df['Type'] = 'Forecast'
#         historical_df['Type'] = 'Historical'
#         combined_df = pd.concat([historical_df, forecast_df])

#         col1, col2 = st.columns(2)
#         col1.metric("Cases MSE (Validation)", f"{mse_cases:.2f}" if not np.isnan(mse_cases) else "N/A", help="Mean Squared Error on validation set")
#         col2.metric("Cases R² Score (Validation)", f"{r2_cases:.4f}" if not np.isnan(r2_cases) else "N/A", help="R² Score indicating model fit")
#         col3, col4 = st.columns(2)
#         col3.metric("Deaths MSE (Validation)", f"{mse_deaths:.2f}" if not np.isnan(mse_deaths) else "N/A", help="Mean Squared Error on validation set")
#         col4.metric("Deaths R² Score (Validation)", f"{r2_deaths:.4f}" if not np.isnan(r2_deaths) else "N/A", help="R² Score indicating model fit")

#     last_14_days = historical_df[-14:]['New_cases']
#     last_7_days = historical_df[-7:]['New_cases']
#     avg_7_day = last_7_days.mean()
#     avg_14_day = last_14_days.mean()
#     trend = ((avg_7_day - avg_14_day) / avg_14_day * 100) if avg_14_day != 0 else 0
#     forecast_avg = forecast_df['New_cases'].mean()
#     forecast_total = forecast_df['New_cases'].sum()

#     col1, col2, col3 = st.columns(3)
#     col1.metric("7-Day Average (Current)", format_number(avg_7_day), f"{trend:.1f}% {'↑' if trend >= 0 else '↓'}",
#                 delta_color="normal" if trend >= 0 else "inverse")
#     col2.metric(f"Average Forecast ({forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')})",
#                 format_number(forecast_avg))
#     col3.metric(f"Total Forecast Cases ({forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')})",
#                 format_number(forecast_total))

#     view = st.radio("Select View", ["All Data", "Recent (90 Days)", "Forecast Focus"], horizontal=True)
#     if view == "Recent (90 Days)":
#         display_df = combined_df[-120:]
#     elif view == "Forecast Focus":
#         display_df = combined_df[-60:]
#     else:
#         display_df = combined_df

#     if display_df.empty or display_df['New_cases'].isna().all():
#         st.error("No valid data available to plot COVID-19 cases.")
#         st.stop()
#     display_df.index = pd.to_datetime(display_df.index)
#     fig_cases = go.Figure()
#     fig_cases.add_trace(go.Scatter(
#         x=display_df.index, y=display_df['New_cases'], mode='lines', name='Daily New Cases', line=dict(color='#3498db')))
#     fig_cases.add_trace(go.Scatter(
#         x=display_df[display_df['Type'] == 'Forecast'].index, y=display_df[display_df['Type'] == 'Forecast']['New_cases'],
#         mode='lines', name='Forecast Cases', line=dict(color='#e74c3c', dash='dash')))
#     fig_cases.update_layout(
#         title="COVID-19 New Cases: Historical and Forecasted (LSTM)", xaxis_title="Date", yaxis_title="New Cases",
#         height=400, template="plotly_white", title_x=0.5)
#     st.plotly_chart(fig_cases, use_container_width=True)

#     if display_df.empty or display_df['New_deaths'].isna().all():
#         st.error("No valid data available to plot COVID-19 deaths.")
#         st.stop()
#     fig_deaths = go.Figure()
#     fig_deaths.add_trace(go.Scatter(
#         x=display_df.index, y=display_df['New_deaths'], mode='lines', name='Daily New Deaths', line=dict(color='#2ecc71')))
#     fig_deaths.add_trace(go.Scatter(
#         x=display_df[display_df['Type'] == 'Forecast'].index, y=display_df[display_df['Type'] == 'Forecast']['New_deaths'],
#         mode='lines', name='Forecast Deaths', line=dict(color='#f39c12', dash='dash')))
#     fig_deaths.update_layout(
#         title="COVID-19 New Deaths: Historical and Forecasted (LSTM)", xaxis_title="Date", yaxis_title="New Deaths",
#         height=400, template="plotly_white", title_x=0.5)
#     st.plotly_chart(fig_deaths, use_container_width=True)

#     st.subheader(f"Forecast from {forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')}")
#     forecast_display = forecast_df[['New_cases', 'Cumulative_cases']].copy()
#     forecast_display['New_cases'] = forecast_display['New_cases'].round(0).astype(int)
#     forecast_display['Cumulative_cases'] = forecast_display['Cumulative_cases'].round(0).astype(int)
#     forecast_display = forecast_display.reset_index().rename(columns={'index': 'Date'})
#     forecast_display['Date'] = forecast_display['Date'].dt.strftime('%d-%m-%Y')
#     st.dataframe(forecast_display.head(10).style.set_table_styles([
#         {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#         {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#     ]), use_container_width=True)
#     if len(forecast_display) > 10:
#         st.write(f"Showing 10 of {len(forecast_display)} forecast days")




# # Page: COVID-19 2022 Analysis
# if page == "COVID-19 2022":

# # Streamlit app title
#     st.title("COVID-19 Case Prediction & Evaluation in India (Sep-Dec 2022) using SARIMA")

# # COVID-19 dataset
# data = pd.DataFrame({
#     'Date_reported1': pd.date_range(start='2022-09-01', end='2022-12-30', freq='D'),
#     'New_cases1': [
#         7946, 6168, 7211, 6817, 5887, 4417, 5379, 6395, 6093, 5554, 5076, 5221, 4369, 5108,
#         6422, 6298, 5747, 5664, 4858, 4043, 4510, 5443, 5383, 4912, 5263, 4129, 3230, 3615,
#         4272, 3947, 3805, 3375, 3011, 1968, 2468, 2529, 1997, 2797, 2756, 2424, 1957, 2139,
#         2786, 2678, 2430, 2401, 2060, 1542, 1946, 2141, 2119, 2112, 1994, 1334, 862, 830,
#         1112, 2208, 1574, 1604, 1326, 1046, 1190, 1321, 1216, 1082, 1132, 937, 625, 811,
#         1016, 842, 833, 734, 547, 0, 501, 635, 656, 556, 492, 406, 294, 360, 408, 347,
#         389, 343, 291, 215, 279, 291, 275, 253, 226, 226, 165, 166, 241, 249, 210, 173,
#         159, 114, 152, 200, 162, 167, 176, 135, 112, 131, 185, 163, 201, 227, 196, 157,
#         188, 268, 243
#     ]
# })
# data.set_index('Date_reported1', inplace=True)

# # Simulate weather data (not used in SARIMA model, included for consistency)
# np.random.seed(42)
# data['Temperature'] = np.random.uniform(20, 30, len(data))
# data['Humidity'] = np.random.uniform(60, 90, len(data))

# # Display dataset
# st.subheader("COVID-19 Dataset")
# st.dataframe(data[['New_cases1']].head())

# # Train-test split starting from 2022-11-01
# train_end_date = '2022-11-01'
# train = data.loc[:train_end_date]
# test = data.loc['2022-11-01':]
# st.write(f"Training data: {len(train)} days (until {train.index[-1].date()})")
# st.write(f"Test data: {len(test)} days (from {test.index[0].date()})")

# # User input for forecast horizon
# st.subheader("Forecast Settings")
# forecast_horizon = st.slider("Select forecast horizon (days)", min_value=1, max_value=50, value=10)

# # Train SARIMA model
# model = SARIMAX(train['New_cases1'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
# results = model.fit(disp=False)

# # Predict on test set
# test_pred = results.get_forecast(steps=len(test))
# test_pred_mean = test_pred.predicted_mean
# test_pred_ci = test_pred.conf_int()

# # Calculate metrics
# mae = mean_absolute_error(test['New_cases1'], test_pred_mean)
# rmse = np.sqrt(mean_squared_error(test['New_cases1'], test_pred_mean))
# mape = np.mean(np.abs((test['New_cases1'] - test_pred_mean) / test['New_cases1'].replace(0, np.nan))) * 100 if not test['New_cases1'].eq(0).all() else np.nan

# # Residual diagnostics
# residuals = test['New_cases1'] - test_pred_mean
# ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
# ljung_pvalue = ljung_box['lb_pvalue'].iloc[0]

# # Display metrics
# st.subheader("Model Evaluation Metrics (Test Set)")
# st.write(f"Mean Absolute Error (MAE): {mae:.2f} cases")
# st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} cases")
# st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# st.write(f"Ljung-Box Test p-value (lag 10): {ljung_pvalue:.4f} {'(Residuals appear random)' if ljung_pvalue > 0.05 else '(Residuals may have autocorrelation)'}")

# # Forecast beyond dataset starting from 2022-11-01
# future_dates = pd.date_range(start='2022-11-01', periods=forecast_horizon, freq='D')
# forecast = results.get_forecast(steps=forecast_horizon)
# forecast_mean = forecast.predicted_mean
# forecast_ci = forecast.conf_int()

# # Plot actual vs predicted
# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=train.index, y=train['New_cases1'], mode='lines', name='Train', line=dict(color='#1f77b4')))
# fig1.add_trace(go.Scatter(x=test.index, y=test['New_cases1'], mode='lines', name='Test Actual', line=dict(color='#2ca02c')))
# fig1.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='#d62728')))
# fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', name='Upper CI', line=dict(color='rgba(128, 128, 128, 0.2)'), showlegend=False))
# fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', name='Lower CI', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', line=dict(color='rgba(128, 128, 128, 0.2)')))
# fig1.update_layout(title="COVID-19 New Cases: Actual vs Predicted (SARIMA)", xaxis_title="Date", yaxis_title="New Cases", template="plotly_white")
# st.plotly_chart(fig1)





# # Page: Malaria Analysis
# if page == "Malaria Analysis":
#     if st.session_state.df_malaria is None:
#         st.warning("Malaria data is not loaded. Please go to the Home page and upload a malaria dataset.")
#         st.stop()
    
#     st.markdown("<div class='card'><h2>Malaria Case Prediction</h2></div>", unsafe_allow_html=True)
#     df_malaria = st.session_state.df_malaria

#     st.header("📈 Data Visualizations")
#     with st.spinner("Generating visualizations..."):
#         viz_data = generate_malaria_visualizations(df_malaria)
#     with st.container():
#         if 'corr_matrix' in viz_data:
#             st.subheader("Correlation Matrix")
#             st.pyplot(viz_data['corr_matrix'])
#             plt.close(viz_data['corr_matrix'])
#         if 'kde_plot' in viz_data:
#             st.subheader("Malaria Cases Distribution (2020-2025)")
#             st.pyplot(viz_data['kde_plot'])
#             plt.close(viz_data['kde_plot'])
#         if 'top_states' in viz_data:
#             st.subheader("Top 10 States by Average Malaria Cases (2020-2024)")
#             st.pyplot(viz_data['top_states'])
#             plt.close(viz_data['top_states'])

#     st.header("🔧 Feature Engineering")
#     with st.spinner("Creating features..."):
#         df_malaria, feat_viz = perform_malaria_feature_engineering(df_malaria)
#         st.session_state.df_malaria = df_malaria
#         st.session_state.user_profile['analyses_run'] += 1
#     st.success("Feature engineering completed.")
#     with st.expander("📊 Updated Dataset with Features", expanded=False):
#         st.dataframe(df_malaria.style.set_table_styles([
#             {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#             {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#         ]))
#     st.subheader("Feature Analysis Visualizations")
#     if 'risk_category' in feat_viz:
#         st.pyplot(feat_viz['risk_category'])
#         plt.close(feat_viz['risk_category'])
#     if 'state_risk' in feat_viz:
#         st.pyplot(feat_viz['state_risk'])
#         plt.close(feat_viz['state_risk'])

#     st.header("🤖 ARIMA Time Series Prediction Model")
#     with st.spinner("Running ARIMA model..."):
#         df_malaria, metrics = run_malaria_arima(df_malaria)
#         st.session_state.df_malaria = df_malaria
#         st.session_state.user_profile['analyses_run'] += 1
#     col1, col2, col3 = st.columns(3)
#     col1.metric("MSE", f"{metrics['mse']:.2f}" if not np.isnan(metrics['mse']) else "N/A", help="Mean Squared Error of the model")
#     col2.metric("R² Score", f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A", help="R² Score indicating model fit")
#     col3.metric("Accuracy", f"{metrics['accuracy']:.2f}%" if not np.isnan(metrics['accuracy']) else "N/A", help="Accuracy based on R²")

#     st.header("📊 Custom Malaria Trend Analysis")
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         selected_state_custom = st.selectbox(
#             "Select State:",
#             df_malaria['STATE_UT'].unique(),
#             index=df_malaria['STATE_UT'].tolist().index('GOA') if 'GOA' in df_malaria['STATE_UT'].tolist() else 0,
#             key='malaria_state_custom_selector')
#         year_range = st.slider(
#             "Select Year Range:",
#             min_value=2020,
#             max_value=2025,
#             value=(2020, 2023),
#             step=1,
#             key='malaria_year_range_selector')
#         st.subheader(f"Data Table")
#         state_data_custom = df_malaria[df_malaria['STATE_UT'] == selected_state_custom].iloc[0]
#         years = list(range(2020, 2026))
#         cases = [
#             state_data_custom['Malaria_2020'],
#             state_data_custom['Malaria_2021'],
#             state_data_custom['Malaria_2022'],
#             state_data_custom['Malaria_2023'],
#             state_data_custom['Malaria_2024'],
#             state_data_custom['Predicted_Malaria_2025']
#         ]
#         cases = [0 if pd.isna(x) else x for x in cases]
#         filtered_years = [y for y in years if year_range[0] <= y <= year_range[1]]
#         filtered_cases = [cases[i] for i, y in enumerate(years) if year_range[0] <= y <= year_range[1]]
#         filtered_types = ["Actual" if y < 2025 else "Forecast" for y in filtered_years]
#         display_data_custom = {
#             "Year": filtered_years,
#             "Cases": filtered_cases,
#             "Type": filtered_types
#         }
#         st.dataframe(pd.DataFrame(display_data_custom).style.set_table_styles([
#             {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#             {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#         ]))
#     with col2:
#         st.subheader(f"Trend Visualization")
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig_custom = plt.figure(figsize=(10, 5))
#         ax_custom = fig_custom.add_subplot(111)
#         actual_years = [y for y in filtered_years if y < 2025]
#         actual_cases = [cases[i] for i, y in enumerate(years) if year_range[0] <= y < 2025 and y <= year_range[1]]
#         forecast_years = [y for y in filtered_years if y >= 2025]
#         forecast_cases = [cases[i] for i, y in enumerate(years) if y >= 2025 and year_range[0] <= y <= year_range[1]]
#         if actual_years:
#             ax_custom.plot(actual_years, actual_cases, marker='o', linestyle='-', color='#3498db', label='Actual', linewidth=2)
#         plot_years = actual_years[-1:] + forecast_years if actual_years and forecast_years else forecast_years
#         plot_cases = [actual_cases[-1]] + forecast_cases if actual_years and forecast_years else forecast_cases
#         if plot_years:
#             ax_custom.plot(plot_years, plot_cases, marker='o', linestyle='--', color='#e74c3c', label='Forecast', linewidth=2)
#         ax_custom.set_title(f"Malaria Cases Trend ({year_range[0]}-{year_range[1]})", fontsize=14, pad=15)
#         ax_custom.set_xlabel("Year", fontsize=12)
#         ax_custom.set_ylabel("Number of Cases", fontsize=12)
#         ax_custom.grid(True, linestyle='--', alpha=0.7)
#         ax_custom.legend()
#         if 2025 in filtered_years:
#             ax_custom.axvspan(2024.5, 2025.5, color='#f1c40f', alpha=0.1)
#             ax_custom.text(2024.8, max(cases)*0.9, 'Forecast', color='#e74c3c', fontsize=10)
#         st.pyplot(fig_custom)
#         plt.close(fig_custom)

#     st.header("🔮 Top Predictions for 2025")
#     tab1, tab2 = st.tabs(["📊 Top 10 States", "📋 All States"])
#     with tab1:
#         top10 = df_malaria[['STATE_UT', 'Predicted_Malaria_2025']].sort_values(
#             by='Predicted_Malaria_2025', ascending=False).head(10)
#         sns.set_style('darkgrid')  # Fixed: Use Seaborn's darkgrid style
#         fig_top, ax_top = plt.subplots(figsize=(10, 5))
#         sns.barplot(data=top10, x='Predicted_Malaria_2025', y='STATE_UT', palette='coolwarm', ax=ax_top)
#         ax_top.set_title("Top 10 Predicted Malaria Cases for 2025", fontsize=14, pad=15)
#         ax_top.set_xlabel("Predicted Cases", fontsize=12)
#         ax_top.set_ylabel("State/UT", fontsize=12)
#         st.pyplot(fig_top)
#         plt.close(fig_top)
#     with tab2:
#         df_display = df_malaria[['STATE_UT', 'Malaria_2020', 'Malaria_2021', 'Malaria_2022',
#                                  'Malaria_2023', 'Malaria_2024', 'Predicted_Malaria_2025']].copy()
#         df_display[['Malaria_2020', 'Malaria_2021', 'Malaria_2022',
#                     'Malaria_2023', 'Malaria_2024', 'Predicted_Malaria_2025']] = df_display[[
#             'Malaria_2020', 'Malaria_2021', 'Malaria_2022',
#             'Malaria_2023', 'Malaria_2024', 'Predicted_Malaria_2025'
#         ]].apply(lambda x: x.fillna(0).astype(int))
#         st.dataframe(df_display.sort_values(by='Predicted_Malaria_2025', ascending=False).style.set_table_styles([
#             {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white')]},
#             {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
#         ]))

#     st.header("📊 Annual Malaria Risk by State")
#     selected_state = st.selectbox(
#         "Select State/UT:",
#         options=sorted(df_malaria["STATE_UT"].unique()),
#         key='malaria_risk_state_selector')
#     available_years = ['2020', '2021', '2022', '2023', '2024', '2025']
#     selected_year = st.selectbox(
#         "Select Year:",
#         options=available_years,
#         key='malaria_risk_year_selector')
#     state_data = df_malaria[df_malaria['STATE_UT'] == selected_state].iloc[0]
#     years = ['2020', '2021', '2022', '2023', '2024', '2025']
#     cases = [state_data[f'Malaria_{year}'] for year in years[:-1]] + [state_data['Predicted_Malaria_2025']]
#     cases = [0 if pd.isna(x) else x for x in cases]
#     risk_data = [classify_risk(c, "malaria") for c in cases]
#     risk_df = pd.DataFrame({
#         'Year': years,
#         'Cases': cases,
#         'Risk Category': [r[0] for r in risk_data],
#         'Color': [r[1] for r in risk_data]
#     })
#     st.subheader(f"Risk Assessment for {selected_state} in {selected_year}")
#     selected_year_data = risk_df[risk_df['Year'] == selected_year].iloc[0]
#     st.write(f"Year: {selected_year_data['Year']}")
#     st.write(f"Number of Cases: {format_number(selected_year_data['Cases'])}")
#     st.markdown(f"Risk Category: <span style='color:{selected_year_data['Color']}'>{selected_year_data['Risk Category']}</span>", unsafe_allow_html=True)

#     st.header("🗺️ Predicted 2025 Malaria Risk by State")
#     df_malaria['Risk_2025'] = df_malaria['Predicted_Malaria_2025'].apply(lambda x: classify_risk(x, "malaria")[0])
#     color_map = {
#         "🔴 High Risk": "#e74c3c",
#         "🟠 Medium Risk": "#f39c12",
#         "🟡 Moderate Risk": "#f1c40f",
#         "🟢 Low Risk": "#2ecc71"
#     }
#     try:
#         fig = px.choropleth(
#             df_malaria,
#             geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
#             featureidkey='properties.ST_NM',
#             locations='STATE_UT',
#             color='Risk_2025',
#             color_discrete_map=color_map,
#             category_orders={"Risk_2025": ["🔴 High Risk", "🟠 Medium Risk", "🟡 Moderate Risk", "🟢 Low Risk"]},
#             title='Predicted Malaria Risk for 2025',
#             hover_data={'Predicted_Malaria_2025': ':,.0f', 'STATE_UT': True},
#             labels={'Risk_2025': 'Risk Category'}
#         )
#         fig.update_geos(
#             fitbounds="locations",
#             visible=False,
#             projection_type="mercator"
#         )
#         fig.update_layout(
#             height=600,
#             margin={"r":0,"t":40,"l":0,"b":0},
#             legend_title_text='Risk Category',
#             title_font_size=20,
#             title_x=0.5
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     except Exception as e:
#         st.error(f"Error rendering choropleth map: {str(e)}")
#     st.caption("""
#         **Malaria Risk Classification:**  
#         🔴 High Risk (>50,000 cases) | 🟠 Medium Risk (25,000-50,000)  
#         🟡 Moderate Risk (10,000-25,000) | 🟢 Low Risk (<10,000)
#     """)

# # Page: Combined Insights
# if page == "Combined Insights":
#     st.markdown("<div class='card'><h2>Combined Insights: COVID-19 and Malaria</h2></div>", unsafe_allow_html=True)
#     if st.session_state.df_covid is None and st.session_state.df_malaria is None:
#         st.warning("Please upload and process datasets for COVID-19 and/or malaria on the Home page to view combined insights.")
#         st.stop()

#     if st.session_state.df_covid is not None and st.session_state.df_malaria is not None:
#         df_covid = st.session_state.df_covid
#         df_malaria = st.session_state.df_malaria

#         df_covid['Year'] = df_covid['Date_reported'].dt.year
#         covid_yearly = df_covid.groupby('Year')['New_cases'].sum().reset_index()
#         covid_yearly['Disease'] = 'COVID-19'

#         malaria_cols = [f'Malaria_{y}' for y in range(2020, 2025)] + ['Predicted_Malaria_2025']
#         years = list(range(2020, 2026))
#         malaria_yearly = pd.DataFrame({
#             'Year': years,
#             'New_cases': [df_malaria[col].sum() if col in df_malaria.columns else 0 for col in malaria_cols]
#         })
#         malaria_yearly['Disease'] = 'Malaria'
#         malaria_yearly['New_cases'] = malaria_yearly['New_cases'].round(0).astype(int)

#         combined_yearly = pd.concat([covid_yearly, malaria_yearly])

#         st.subheader("Annual Case Trends: COVID-19 vs. Malaria")
#         try:
#             fig_combined = px.line(
#                 combined_yearly,
#                 x='Year',
#                 y='New_cases',
#                 color='Disease',
#                 title='Annual Case Trends (2020-2025)',
#                 labels={'New_cases': 'Total Cases', 'Year': 'Year'},
#                 color_discrete_map={'COVID-19': '#3498db', 'Malaria': '#2ecc71'}
#             )
#             malaria_2025 = combined_yearly[(combined_yearly['Disease'] == 'Malaria') & (combined_yearly['Year'] >= 2024)]
#             fig_combined.add_trace(go.Scatter(
#                 x=malaria_2025['Year'],
#                 y=malaria_2025['New_cases'],
#                 mode='lines',
#                 name='Malaria (Forecast)',
#                 line=dict(color='#2ecc71', dash='dash'),
#                 showlegend=False
#             ))
#             fig_combined.update_layout(
#                 height=400,
#                 template="plotly_white",
#                 title_x=0.5,
#                 yaxis=dict(title='Total Cases (Log Scale)', type='log')
#             )
#             st.plotly_chart(fig_combined, use_container_width=True)
#         except Exception as e:
#             st.error(f"Error plotting combined trends: {str(e)}")

#         st.subheader("Risk Comparison for 2024")
#         covid_2024_cases = df_covid[df_covid['Year'] == 2024]['New_cases'].sum()
#         malaria_2024_cases = df_malaria['Malaria_2024'].sum().round(0).astype(int)
#         covid_risk = classify_risk(covid_2024_cases, "covid")[0]
#         malaria_risk = classify_risk(malaria_2024_cases, "malaria")[0]

#         col1, col2 = st.columns(2)
#         col1.metric("COVID-19 Risk (2024)", covid_risk, f"{format_number(covid_2024_cases)} cases")
#         col2.metric("Malaria Risk (2024)", malaria_risk, f"{format_number(malaria_2024_cases)} cases")

#         st.subheader("Model Accuracy Comparison")
#         historical_df = df_covid.set_index('Date_reported')['New_cases']
#         forecast_steps = 30
#         _, mse_cases, r2_cases = lstm_forecast(historical_df[-365:], forecast_steps)
#         covid_accuracy = r2_cases * 100 if not np.isnan(r2_cases) else float('nan')
#         _, malaria_metrics = run_malaria_arima(df_malaria)
#         malaria_accuracy = malaria_metrics['accuracy']
#         st.session_state.user_profile['analyses_run'] += 1

#         col1, col2 = st.columns(2)
#         col1.metric("COVID-19 LSTM Accuracy", f"{covid_accuracy:.2f}%" if not np.isnan(covid_accuracy) else "N/A",
#                     help="Accuracy based on R² score for LSTM model on validation set")
#         col2.metric("Malaria ARIMA Accuracy", f"{malaria_accuracy:.2f}%" if not np.isnan(malaria_accuracy) else "N/A",
#                     help="Accuracy based on R² score for ARIMA model")
#     else:
#         st.info("Please upload both datasets to view combined insights.")

# # Page: Profile
# if page == "Profile":
#     st.markdown("<div class='card'><h2>👤 User Profile</h2></div>", unsafe_allow_html=True)
#     st.subheader("User Information")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write(f"Name: {st.session_state.user_profile['name'] or 'Not set'}")
#         st.write(f"Email: {st.session_state.user_profile['email'] or 'Not set'}")
#     with col2:
#         st.write(f"Analyses Run: {st.session_state.user_profile['analyses_run']}")
#     with st.form("profile_form"):
#         st.subheader("Edit Profile")
#         name = st.text_input("Name", value=st.session_state.user_profile['name'])
#         email = st.text_input("Email", value=st.session_state.user_profile['email'])
#         submitted = st.form_submit_button("💾 Save Profile")
#         if submitted:
#             if email and '@' not in email:
#                 st.error("Please enter a valid email address.")
#             else:
#                 st.session_state.user_profile['name'] = name
#                 st.session_state.user_profile['email'] = email
#                 st.success("Profile updated successfully!")
#                 st.rerun()
#     st.subheader("Analysis History")
#     if st.session_state.user_profile['analyses_run'] > 0:
#         st.write(f"You have run {st.session_state.user_profile['analyses_run']} analyses.")
#         st.info("Detailed analysis history is not yet implemented. Future updates may include logs of past forecasts.")
#     else:
#         st.write("No analyses have been run yet.")


import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Suppress TensorFlow oneDNN warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Streamlit app title
st.title("COVID-19 Case Forecast (June 15, 2025 - July 15, 2025)")

# Data (full dataset)
dates = pd.to_datetime([
    '01-Jan-21', '01-Feb-21', '01-Mar-21', '01-Apr-21', '01-May-21', '01-Jun-21',
    '01-Jul-21', '01-Aug-21', '01-Sep-21', '01-Oct-21', '01-Nov-21', '01-Dec-21',
    '02-Dec-21', '03-Dec-21', '04-Dec-21', '05-Dec-21', '06-Dec-21', '07-Dec-21',
    '08-Dec-21', '09-Dec-21', '10-Dec-21', '11-Dec-21', '12-Dec-21', '13-Dec-21',
    '14-Dec-21', '15-Dec-21', '16-Dec-21', '17-Dec-21', '18-Dec-21', '19-Dec-21',
    '20-Dec-21', '21-Dec-21', '22-Dec-21', '23-Dec-21', '24-Dec-21', '25-Dec-21',
    '26-Dec-21', '27-Dec-21', '28-Dec-21', '29-Dec-21', '30-Dec-21', '31-Dec-21',
    '01-Jan-22', '02-Jan-22', '03-Jan-22', '04-Jan-22', '05-Jan-22', '06-Jan-22',
    '07-Jan-22', '08-Jan-22', '09-Jan-22', '10-Jan-22', '11-Jan-22', '12-Jan-22',
    '13-Jan-22', '14-Jan-22', '15-Jan-22', '16-Jan-22', '17-Jan-22', '18-Jan-22',
    '19-Jan-22', '20-Jan-22', '21-Jan-22', '22-Jan-22', '23-Jan-22', '24-Jan-22',
    '25-Jan-22', '26-Jan-22', '27-Jan-22', '28-Jan-22', '29-Jan-22', '30-Jan-22',
    '31-Jan-22', '01-Feb-22', '02-Feb-22', '03-Feb-22', '04-Feb-22', '05-Feb-22',
    '06-Feb-22', '07-Feb-22', '08-Feb-22', '09-Feb-22', '10-Feb-22', '11-Feb-22',
    '12-Feb-22', '13-Feb-22', '14-Feb-22', '15-Feb-22', '16-Feb-22', '17-Feb-22',
    '18-Feb-22', '19-Feb-22', '20-Feb-22', '21-Feb-22', '22-Feb-22', '23-Feb-22',
    '24-Feb-22', '25-Feb-22', '26-Feb-22', '27-Feb-22', '28-Feb-22', '01-Mar-22',
    '02-Mar-22', '03-Mar-22', '04-Mar-22', '05-Mar-22', '06-Mar-22', '07-Mar-22',
    '08-Mar-22', '09-Mar-22', '10-Mar-22', '11-Mar-22', '12-Mar-22', '13-Mar-22',
    '14-Mar-22', '15-Mar-22', '16-Mar-22', '17-Mar-22', '18-Mar-22', '19-Mar-22',
    '20-Mar-22', '21-Mar-22', '22-Mar-22', '23-Mar-22', '24-Mar-22', '25-Mar-22',
    '26-Mar-22', '27-Mar-22', '28-Mar-22', '29-Mar-22', '30-Mar-22', '31-Mar-22',
    '01-Apr-22', '02-Apr-22', '03-Apr-22', '04-Apr-22', '05-Apr-22', '06-Apr-22',
    '07-Apr-22', '08-Apr-22', '09-Apr-22', '10-Apr-22', '11-Apr-22', '12-Apr-22',
    '13-Apr-22', '14-Apr-22', '15-Apr-22', '16-Apr-22', '17-Apr-22', '18-Apr-22',
    '19-Apr-22', '20-Apr-22', '21-Apr-22', '22-Apr-22', '23-Apr-22', '24-Apr-22',
    '25-Apr-22', '26-Apr-22', '27-Apr-22', '28-Apr-22', '29-Apr-22', '30-Apr-22',
    '01-May-22', '02-May-22', '03-May-22', '04-May-22', '05-May-22', '06-May-22',
    '07-May-22', '08-May-22', '09-May-22', '10-May-22', '11-May-22', '12-May-22',
    '13-May-22', '14-May-22', '15-May-22', '16-May-22', '17-May-22', '18-May-22',
    '19-May-22', '20-May-22', '21-May-22', '22-May-22', '23-May-22', '24-May-22',
    '25-May-22', '26-May-22', '27-May-22', '28-May-22', '29-May-22', '30-May-22',
    '31-May-22', '01-Jun-22', '02-Jun-22', '03-Jun-22', '04-Jun-22', '05-Jun-22',
    '06-Jun-22', '07-Jun-22', '08-Jun-22', '09-Jun-22', '10-Jun-22', '11-Jun-22',
    '12-Jun-22', '13-Jun-22', '14-Jun-22', '15-Jun-22', '16-Jun-22', '17-Jun-22',
    '18-Jun-22', '19-Jun-22', '20-Jun-22', '21-Jun-22', '22-Jun-22', '23-Jun-22',
    '24-Jun-22', '25-Jun-22', '26-Jun-22', '27-Jun-22', '28-Jun-22', '29-Jun-22',
    '30-Jun-22', '01-Jul-22', '02-Jul-22', '03-Jul-22', '04-Jul-22', '05-Jul-22',
    '06-Jul-22', '07-Jul-22', '08-Jul-22', '09-Jul-22', '10-Jul-22', '11-Jul-22',
    '12-Jul-22', '13-Jul-22', '14-Jul-22', '15-Jul-22', '16-Jul-22', '17-Jul-22',
    '18-Jul-22', '19-Jul-22', '20-Jul-22', '21-Jul-22', '22-Jul-22', '23-Jul-22',
    '24-Jul-22', '25-Jul-22', '26-Jul-22', '27-Jul-22', '28-Jul-22', '29-Jul-22',
    '30-Jul-22', '31-Jul-22', '01-Aug-22', '02-Aug-22', '03-Aug-22', '04-Aug-22',
    '05-Aug-22', '06-Aug-22', '07-Aug-22', '08-Aug-22', '09-Aug-22', '10-Aug-22',
    '11-Aug-22', '12-Aug-22', '13-Aug-22', '14-Aug-22', '15-Aug-22', '16-Aug-22',
    '17-Aug-22', '18-Aug-22', '19-Aug-22', '20-Aug-22', '21-Aug-22', '22-Aug-22',
    '23-Aug-22', '24-Aug-22', '25-Aug-22', '26-Aug-22', '27-Aug-22', '28-Aug-22',
    '29-Aug-22', '30-Aug-22', '31-Aug-22', '01-Sep-22', '02-Sep-22', '03-Sep-22',
    '04-Sep-22', '05-Sep-22', '06-Sep-22', '07-Sep-22', '08-Sep-22', '09-Sep-22',
    '10-Sep-22', '11-Sep-22', '12-Sep-22', '13-Sep-22', '14-Sep-22', '15-Sep-22',
    '16-Sep-22', '17-Sep-22', '18-Sep-22', '19-Sep-22', '20-Sep-22', '21-Sep-22',
    '22-Sep-22', '23-Sep-22', '24-Sep-22', '25-Sep-22', '26-Sep-22', '27-Sep-22',
    '28-Sep-22', '29-Sep-22', '30-Sep-22', '01-Oct-22', '02-Oct-22', '03-Oct-22',
    '04-Oct-22', '05-Oct-22', '06-Oct-22', '07-Oct-22', '08-Oct-22', '09-Oct-22',
    '10-Oct-22', '11-Oct-22', '12-Oct-22', '13-Oct-22', '14-Oct-22', '15-Oct-22',
    '16-Oct-22', '17-Oct-22', '18-Oct-22', '19-Oct-22', '20-Oct-22', '21-Oct-22',
    '22-Oct-22', '23-Oct-22', '24-Oct-22', '25-Oct-22', '26-Oct-22', '27-Oct-22',
    '28-Oct-22', '29-Oct-22', '30-Oct-22', '31-Oct-22', '01-Nov-22', '02-Nov-22',
    '03-Nov-22', '04-Nov-22', '05-Nov-22', '06-Nov-22', '07-Nov-22', '08-Nov-22',
    '09-Nov-22', '10-Nov-22', '11-Nov-22', '12-Nov-22', '13-Nov-22', '14-Nov-22',
    '15-Nov-22'
], format='%d-%b-%y')

new_cases = [
    20035, 11427, 15510, 72330, 401993, 127510, 48786, 41831, 41965, 26727, 12514, 8954, 9765,
    9216, 8603, 8895, 8306, 6822, 8439, 9419, 8503, 7992, 7774, 7350, 5784, 6984, 7974, 7447,
    7145, 7081, 6563, 5326, 6317, 7495, 6650, 7189, 6987, 6531, 6358, 9195, 13154, 16764,
    22775, 27553, 33750, 37379, 58097, 90928, 117100, 141986, 159632, 179723, 168063, 194720,
    247417, 264202, 268833, 271202, 258089, 238018, 282970, 317532, 347254, 337704, 333533,
    306064, 255874, 285914, 286384, 251209, 235532, 234281, 209918, 167059, 161386, 172433,
    149394, 127952, 107474, 83876, 67597, 71365, 67084, 58077, 50407, 44877, 34113, 27409,
    30615, 30757, 25920, 22270, 19968, 16051, 13405, 15102, 14148, 13166, 11499, 10273, 8013,
    6915, 7554, 6561, 6396, 5921, 5476, 4362, 3993, 4575, 4184, 4194, 3614, 3116, 2503, 2568,
    2876, 2539, 2528, 2075, 1761, 1549, 1581, 1778, 1938, 1685, 1660, 1421, 1270, 1259, 1233,
    1225, 1335, 1260, 1096, 913, 795, 1086, 1033, 1109, 1150, 1054, 861, 796, 1088, 1007, 949,
    975, 1150, 2183, 1247, 2067, 2380, 2451, 2527, 2593, 2541, 2483, 2927, 3303, 3377, 3688,
    3324, 3157, 2568, 3205, 3275, 3545, 3805, 3451, 3207, 2288, 2897, 2827, 2841, 2858, 2487,
    2202, 1569, 1829, 2364, 2259, 2323, 2226, 2022, 1675, 2124, 2628, 2710, 2685, 2828, 2706,
    2338, 2745, 3712, 4041, 3962, 4270, 4518, 3714, 5233, 7240, 7584, 8329, 8582, 8084, 6594,
    8822, 12213, 12847, 13216, 12899, 12781, 9923, 12249, 13313, 17336, 15940, 11739, 17073,
    11793, 14506, 18819, 17070, 17092, 16103, 16135, 13086, 16159, 18930, 18815, 18840, 18257,
    16678, 13615, 16906, 20139, 20038, 20044, 20528, 16935, 15528, 20557, 21566, 21880, 21411,
    20279, 16866, 14830, 18313, 20557, 20409, 20408, 19673, 16464, 13734, 17135, 19893, 20551,
    19406, 18738, 16167, 12751, 16047, 16299, 16561, 15815, 14092, 14917, 8813, 9062, 12608,
    15754, 13272, 11539, 9531, 8586, 10649, 10725, 10256, 9520, 9436, 7591, 5439, 7231, 7946,
    6168, 7211, 6817, 5910, 4417, 5379, 6395, 6093, 5554, 5076, 5221, 4369, 5108, 6422, 6298,
    5747, 5664, 4858, 4043, 4510, 5443, 5383, 4912, 4777, 4129, 3230, 3615, 4272, 3947, 3805,
    3375, 3011, 1968, 2468, 2529, 1997, 2797, 2756, 2424, 1957, 2139, 2786, 2678, 2430, 2401,
    2060, 1542, 1946, 2141, 2119, 2112, 1994, 1334, 862, 830, 1112, 2208, 1574, 1604, 1326,
    1046, 1190, 1321, 1216, 1082, 1132, 937, 625, 811, 1016, 842, 833, 734, 547, 749
]

# Create DataFrame
df = pd.DataFrame({'date': dates, 'new_cases': new_cases})
df.set_index('date', inplace=True)

# Apply 7-day moving average to smooth data
df['smoothed_cases'] = df['new_cases'].rolling(window=7, min_periods=1).mean()

# Select last 90 days for modeling
last_90_days = df['smoothed_cases'][-90:]

# Fit ARIMA(1,1,1) model
model = ARIMA(last_90_days, order=(1,1,1))
model_fit = model.fit()

# Forecast for 31 days to match June 15, 2025 - July 15, 2025
forecast_steps = 31
forecast = model_fit.forecast(steps=forecast_steps)
forecast_summary = model_fit.get_forecast(steps=forecast_steps)
conf_int = forecast_summary.conf_int(alpha=0.05)  # 95% CI

# Create forecast dates
forecast_dates = pd.date_range(start='2022-11-16', end='2022-12-16', freq='D')

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Predicted Cases': forecast,
    'Lower CI': conf_int.iloc[:, 0],
    'Upper CI': conf_int.iloc[:, 1]
})

# Display forecast table
st.subheader("Forecasted COVID-19 Cases")
st.dataframe(forecast_df.round(0))

# Display total predicted cases
total_cases = int(forecast_df['Predicted Cases'].sum())
lower_ci_total = int(forecast_df['Lower CI'].sum())
upper_ci_total = int(forecast_df['Upper CI'].sum())
st.write(f"**Total Predicted Cases**: {total_cases}")
st.write(f"**95% Confidence Interval**: {lower_ci_total} - {upper_ci_total}")

# Plot the forecast
st.subheader("Forecast Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(last_90_days.index, last_90_days, label='Historical Smoothed Cases', color='blue')
ax.plot(forecast_df['Date'], forecast_df['Predicted Cases'], label='Forecasted Cases', color='red')
ax.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'], 
                color='red', alpha=0.2, label='95% Confidence Interval')
ax.set_title('COVID-19 Case Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Daily New Cases (Smoothed)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Instructions for users
st.markdown("""
**Note**: This forecast is based on data up to November 15, 2022, and assumes the trend continues. 
Real-world factors like new variants or policy changes may affect accuracy. 
For more recent data, consult WHO or CDC dashboards.
""")