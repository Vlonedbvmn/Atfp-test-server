import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST, NHITS
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np
import datetime
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import streamlit as st
# import gettext
import glob
import pickle
import os
from streamlit_pills import pills
import io


# Define your supported languages
# languages = {"English": "en", "Español": "es"}
# # selected_language = st.sidebar.selectbox("Choose your language", list(languages.keys()))
# lang_code = languages[selected_language]
#
# # Load the appropriate translation (assuming your locale files are in the 'locales' folder)
# translation = gettext.translation('messages', localedir='locales', languages=[lang_code], fallback=True)
# translation.install()
# _ = translation.gettext

# st.write(_("Welcome to my app!"))

# st.set_page_config(
#     page_title="Модель",
#     layout="wide",
#     initial_sidebar_state="auto"
# )

means = {"Місяць": "M",
         "Година": "h",
         "Рік": "Y",
         "Хвилина": "T",
         "Секунда": "S",
         "День": "D",
         }

if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'mse' not in st.session_state:
    st.session_state.mse = None
if 'inst_name' not in st.session_state:
    st.session_state.inst_name = None
if 'model_forecast' not in st.session_state:
    st.session_state.model_forecast = None
if 'df_forpred' not in st.session_state:
    st.session_state.df_forpred = None
if 'horiz' not in st.session_state:
    st.session_state.horiz = None
if 'date_not_n' not in st.session_state:
    st.session_state.date_not_n = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'win' not in st.session_state:
    st.session_state.win = None
if 'wres' not in st.session_state:
    st.session_state.wres = None
if 'reser_size' not in st.session_state:
    st.session_state.reser_size = None
if 'inp' not in st.session_state:
    st.session_state.inp = None
if 'predicted3' not in st.session_state:
    st.session_state.predicted3 = None
if 'predicted4' not in st.session_state:
    st.session_state.predicted4 = None
if 'plotp2' not in st.session_state:
    st.session_state.plotp2 = None
if 'bp2' not in st.session_state:
    st.session_state.bp2 = None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data



def mk_fcst(datafre, ticker, models_dir, horizon, tsk="stock"):
    class SNNRegression(nn.Module):
        def __init__(self, reservoir_size, output_size):
            super(SNNRegression, self).__init__()
            self.lif1 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid())
            self.bn1 = nn.BatchNorm1d(reservoir_size)
            self.tcn1 = nn.Sequential(
                nn.Conv1d(in_channels=reservoir_size, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(128)
            )
            self.lif2 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid())
            self.bn2 = nn.BatchNorm1d(128)
            self.tcn2 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Conv1d(in_channels=256, out_channels=output_size, kernel_size=3, padding=1)
            )

        def forward(self, x):
            x = x.reshape(x.size(0), -1)
            time_steps = 150
            mem1 = self.lif1.init_leaky()
            for t in range(time_steps):
                spk1, mem1 = self.lif1(x, mem1)
            mem1 = self.bn1(mem1)
            mem1 = mem1.unsqueeze(2)
            mem1 = self.tcn1(mem1).squeeze(2)
            mem2 = self.lif2.init_leaky()
            for t in range(time_steps):
                spk2, mem2 = self.lif2(mem1, mem2)
            mem2 = self.bn2(mem2)
            mem2 = mem2.unsqueeze(2)
            out = self.tcn2(mem2).squeeze(2)
            return out

    # Set up file paths and device
    if tsk == "cryp":
        model_filename = os.path.join(models_dir, f"{ticker}-USD_intraday_model.pth")
        reservoir_filename = os.path.join(models_dir, f"{ticker}-USD_intraday_reservoir.pth")
        scaler_filename = os.path.join(models_dir, f"{ticker}-USD_intraday_scaler.pkl")
    else:
        model_filename = os.path.join(models_dir, f"{ticker}_daily_model.pth")
        reservoir_filename = os.path.join(models_dir, f"{ticker}_daily_reservoir.pth")
        scaler_filename = os.path.join(models_dir, f"{ticker}_daily_scaler.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load reservoir parameters (W_in, W_res, reservoir_size)
    reservoir_data = torch.load(reservoir_filename, map_location=device)
    W_in = reservoir_data["W_in"]
    W_res = reservoir_data["W_res"]
    reservoir_size = reservoir_data["reservoir_size"]

    # Set the output dimension (must match your training configuration; here, e.g., 20)
    output_dim = 30

    # Create the model instance and load the state dict
    model = SNNRegression(reservoir_size, output_dim).to(device)
    model_state = torch.load(model_filename, map_location=device)
    model.load_state_dict(model_state)

    # Load the scaler using pickle
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    print("Model, reservoir parameters, and scaler loaded successfully.")

    def make_prediction(input_values):
        input_values = scaler.transform(input_values)
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0)
            reservoir_state = torch.tanh(
                W_in @ input_tensor.T + W_res @ torch.rand(
                    reservoir_size, 1))
            reservoir_state = reservoir_state.T
            reservoir_state = reservoir_state.unsqueeze(2)
            prediction = model(reservoir_state).cpu().numpy()
        return prediction

    print(datafre)
    qu = int(round(len(datafre) * 0.1, 0))
    h = datafre[-qu:]["y"].tolist()
    new_sample = datafre["y"].tolist()[-50:]
    new_sample = np.array(new_sample).reshape(-1, 1)

    print(new_sample)
    result = make_prediction(new_sample)
    print("Prediction for new sample:", result)
    pr = []
    counterr = 0
    for i in result.tolist()[0]:
        if counterr < horizon:
            pr.append(i)
            h.append(i)
            counterr += 1
        else: break

    pr3dates = []
    pr4dates = []
    for i in range(1, len(h) + 1):
        pr3dates.append(i)
    for i in range(1, len(pr) + 1):
        pr4dates.append(i)

    st.session_state.predicted3 = pd.DataFrame({
        st.session_state.date: pr3dates,
        st.session_state.target: h
    })
    st.session_state.predicted4 = pd.DataFrame({
        st.session_state.date: pr4dates,
        st.session_state.target: pr
    })


if st.session_state.df is not None:
    ds_for_pred = pd.DataFrame()
    ds_for_pred["y"] = st.session_state.df[st.session_state.target]
    try:
        ds_for_pred["ds"] = st.session_state.df[st.session_state.date]
        print(ds_for_pred["ds"])
        st.session_state.date_not_n = False
        ds_for_pred['ds'] = pd.to_datetime(ds_for_pred['ds'])
    except:
        st.session_state.date_not_n = True
        ds_for_pred['ds'] = [i for i in range(1, len(ds_for_pred) + 1)]

    # if st.session_state.date_not_n:
    #     start_date = pd.to_datetime('2024-01-01')
    #     ds_for_pred['ds'] = start_date + pd.to_timedelta(ds_for_pred['ds'] - 1, rarety)

    # ds_for_pred['ds'] = pd.to_datetime(ds_for_pred['ds'])
    # ds_for_pred = ds_for_pred.set_index('ds').asfreq(rarety)
    # ds_for_pred = ds_for_pred.reset_index()
    # ds_for_pred['y'] = ds_for_pred['y'].interpolate()
    # ds_for_pred["unique_id"] = [0 for i in range(1, len(ds_for_pred) + 1)]
    # print("s;kgfoshdisdifsdf")
    # print(ds_for_pred)
    print(ds_for_pred)
    # st.session_state.df_forpred = ds_for_pred
    with st.container():
        st.title("Плагіни")

    plug = option_menu("Оберіть категорію плагінів для прогнозування",
                         ["Stock price", "Crypto"],
                         # icons=['gear', 'gear', 'gear', 'gear', 'gear', 'gear'],
                         menu_icon="no",
                         orientation="horizontal")

    # try:
    if plug == "Stock price":
        st.markdown("## Плагіни stock price")
        folder_path = 'pa/models/'
        # Get list of all .pth files in folder
        pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
        print(f"Found {len(pth_files)} CSV files.")
        # file_list = file_list[:5]
        ticks = []
        for file in pth_files:
            # Extract ticker symbol from filename (assuming filename like TICKER.csv)
            # print(file.replace("\\", "/"))
            # # fi = pd.read_csv(file.replace("\\", "/"))
            # # fi['Date'] = [i for i in range(1, len(fi) + 1)]
            # ticker = os.path.splitext(os.path.basename(file.replace("\\", "/")))[0]
            if file.split("_")[0].split("/")[-1] not in ticks:
                ticks.append(file.split("_")[0].split("/")[-1])
        selection = pills("Тикери", sorted(ticks))
        if selection is not None:
            st.markdown(f"## Ви обрали плагін: {selection}")
            with open(f'pa/models/{selection}_explanation_ukr.txt', 'r') as filee:
                content = filee.read()
                print(content)
                st.write(content)
            horizon = st.select_slider(
                "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                options=[i for i in range(1, 31)],
                key="one11"
            )
            st.button(label="Зробити прогноз", key="kan", on_click=mk_fcst,
                      args=(ds_for_pred, selection, "pa/models", horizon))
            st.divider()
            st.markdown(f"### Результати прогнозу")
            if st.session_state.predicted4 is not None:
                col3, col4 = st.columns(2)
                with col3:
                    with st.expander("Подивитись прогнозні значення:"):
                        st.write(st.session_state.predicted4)
                with col4:

                    st.download_button(
                        label="Завантажити прогноз як файл .csv",
                        data=st.session_state.predicted4.to_csv().encode("utf-8"),
                        file_name="prediction.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Завантажити прогноз як файл .xlsx",
                        data=to_excel(st.session_state.predicted4),
                        file_name="prediction.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                st.divider()

                st.markdown(f"### Дашборд прогнозу")
                st.markdown("# ")
                if st.session_state.date_not_n == True:
                    st.session_state.predicted[st.session_state.date] = [i for i in
                                                                         range(1, len(st.session_state.predicted3) + 1)]
                else:
                    pass
                last_days = st.session_state.predicted3.tail(horizon)
                rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                val = len(last_days)

                cool1, cool2 = st.columns([2, 5])

                # Create the plotly figure
                with cool1:
                    st.markdown("##### Вибір горизонту прогнозу ")
                    st.markdown("# ")
                    st.markdown("# ")
                    slid = st.select_slider(
                        "Горизонт прогнозу:",
                        options=[i for i in range(1, val + 1)])
                    st.markdown("# ")
                    st.markdown("# ")
                    st.markdown("##### Статистика прогнозу ")

                    st.write(last_days[:(slid)].describe().head(7), use_container_width=True)
                    # else:
                    #     st.write(last_days[:(slid)].describe().drop(["unique_id"], axis=1).head(7),
                    #              use_container_width=True)

                with cool2:

                    st.session_state.plotp2 = go.Figure()
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=rest_of_data[st.session_state.date],
                        y=rest_of_data[st.session_state.target],
                        mode='lines',
                        name='Дані',
                        line=dict(color='blue')
                    ))

                    # Plot the last seven days in a different color
                    # st.session_state.plotp.add_trace(go.Scatter(
                    #     x=last_days[st.session_state.date][:(slid)],
                    #     y=last_days[st.session_state.target][:(slid)],
                    #     mode='lines',
                    #     name='Прогноз',
                    #     line=dict(color='green')
                    # ))
                    q50 = last_days[st.session_state.target][:(slid)]
                    lower_forecast = q50 / 1.2  # q1: median divided by 1.5
                    upper_forecast = q50 * 1.2  # q99: median multiplied by 1.5
                    if lower_forecast is not None:
                        max_value = max(upper_forecast.tolist()) + 100
                        min_value = min(lower_forecast.tolist()) - 100
                    # First, add the upper bound trace (invisible line) to serve as the fill ceiling.
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=last_days[st.session_state.date][:(slid)],
                        y=upper_forecast,
                        mode='lines',
                        line=dict(color='rgba(0,128,0,0)'),  # fully transparent line
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Next, add the lower bound trace that fills the area up to the previous (upper bound) trace.
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=last_days[st.session_state.date][:(slid)],
                        y=lower_forecast,
                        mode='lines',
                        fill='tonexty',  # fills the area between this trace and the one above
                        fillcolor='rgba(0,128,0,0.2)',  # adjust the color and transparency as needed
                        line=dict(color='rgba(0,128,0,0)'),  # transparent line to keep the focus on the fill
                        name='Діапазон можливих значень прогнозу'
                    ))

                    # Finally, add the median (50% quantile) forecast line on top.
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=last_days[st.session_state.date][:(slid)],
                        y=q50,
                        mode='lines',
                        name='Прогноз',
                        line=dict(color='green')
                    ))
                    # Update layout (optional)
                    st.session_state.plotp2.update_layout(
                        xaxis_title='Дата',
                        # yaxis_title='Значення',
                        yaxis=dict(
                            # range=[min_value, max_value],
                            title='Спрогнозовані значення'  # Optional: add a title for clarity
                        ),
                        title="Графік прогнозу",  # Increase the overall height
                    )

                    # Show the plot
                    st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                    # Plot the data except the last seven days

                    st.session_state.bp2 = go.Figure()

                    st.session_state.bp2.add_trace(go.Bar(
                        x=last_days[st.session_state.date][:(slid)],
                        y=last_days[st.session_state.target][:(slid)],
                        name='Прогноз',
                        marker_color='green'
                    ))

                    # Customize layout
                    st.session_state.bp2.update_layout(
                        title='Барплот прогнозу',
                        xaxis_title='Дата',
                        yaxis_title='Значення',
                        template='plotly_white'
                    )

                    # Display the Plotly chart in Streamlit
                    st.plotly_chart(st.session_state.bp2, use_container_width=True)
    if plug == "Crypto":
        st.markdown("## Плагіни stock price")
        folder_path = 'pa/crypto_models/'
        # Get list of all .pth files in folder
        pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
        print(f"Found {len(pth_files)} CSV files.")
        # file_list = file_list[:5]
        ticks = []
        for file in pth_files:
            # Extract ticker symbol from filename (assuming filename like TICKER.csv)
            # print(file.replace("\\", "/"))
            # # fi = pd.read_csv(file.replace("\\", "/"))
            # # fi['Date'] = [i for i in range(1, len(fi) + 1)]
            # ticker = os.path.splitext(os.path.basename(file.replace("\\", "/")))[0]
            if file.split("/")[2].split("-")[0] not in ticks:
                ticks.append(file.split("/")[2].split("-")[0])
        selection = pills("Tickers", sorted(ticks))
        if selection is not None:
            st.markdown(f"Ви обрали плагін: {selection}.")
            horizon = st.select_slider(
                "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                options=[i for i in range(1, 31)],
                key="one11"
            )
            st.button(label="Зробити прогноз", key="kan", on_click=mk_fcst,
                      args=(ds_for_pred, selection, "pa/crypto_models", horizon, "cryp"))
            st.divider()
            st.markdown(f"### Результати прогнозу")
            if st.session_state.predicted4 is not None:
                col3, col4 = st.columns(2)
                with col3:
                    with st.expander("Подивитись прогнозні значення:"):
                        st.write(st.session_state.predicted4)
                with col4:

                    st.download_button(
                        label="Завантажити прогноз як файл .csv",
                        data=st.session_state.predicted4.to_csv().encode("utf-8"),
                        file_name="prediction.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Завантажити прогноз як файл .xlsx",
                        data=to_excel(st.session_state.predicted4),
                        file_name="prediction.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                st.divider()

                st.markdown(f"### Дашборд прогнозу")
                st.markdown("# ")
                if st.session_state.date_not_n == True:
                    st.session_state.predicted[st.session_state.date] = [i for i in
                                                                         range(1, len(st.session_state.predicted3) + 1)]
                else:
                    pass
                last_days = st.session_state.predicted3.tail(horizon)
                rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                val = len(last_days)

                cool1, cool2 = st.columns([2, 5])

                # Create the plotly figure
                with cool1:
                    st.markdown("##### Вибір горизонту прогнозу ")
                    st.markdown("# ")
                    st.markdown("# ")
                    slid = st.select_slider(
                        "Горизонт прогнозу:",
                        options=[i for i in range(1, val + 1)])
                    st.markdown("# ")
                    st.markdown("# ")
                    st.markdown("##### Статистика прогнозу ")

                    st.write(last_days[:(slid)].describe().head(7), use_container_width=True)
                    # else:
                    #     st.write(last_days[:(slid)].describe().drop(["unique_id"], axis=1).head(7),
                    #              use_container_width=True)

                with cool2:

                    st.session_state.plotp2 = go.Figure()
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=rest_of_data[st.session_state.date],
                        y=rest_of_data[st.session_state.target],
                        mode='lines',
                        name='Дані',
                        line=dict(color='blue')
                    ))

                    # Plot the last seven days in a different color
                    # st.session_state.plotp.add_trace(go.Scatter(
                    #     x=last_days[st.session_state.date][:(slid)],
                    #     y=last_days[st.session_state.target][:(slid)],
                    #     mode='lines',
                    #     name='Прогноз',
                    #     line=dict(color='green')
                    # ))
                    q50 = last_days[st.session_state.target][:(slid)]
                    lower_forecast = q50 / 1.2  # q1: median divided by 1.5
                    upper_forecast = q50 * 1.2  # q99: median multiplied by 1.5
                    if lower_forecast is not None:
                        max_value = max(upper_forecast.tolist()) + 100
                        min_value = min(lower_forecast.tolist()) - 100
                    # First, add the upper bound trace (invisible line) to serve as the fill ceiling.
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=last_days[st.session_state.date][:(slid)],
                        y=upper_forecast,
                        mode='lines',
                        line=dict(color='rgba(0,128,0,0)'),  # fully transparent line
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Next, add the lower bound trace that fills the area up to the previous (upper bound) trace.
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=last_days[st.session_state.date][:(slid)],
                        y=lower_forecast,
                        mode='lines',
                        fill='tonexty',  # fills the area between this trace and the one above
                        fillcolor='rgba(0,128,0,0.2)',  # adjust the color and transparency as needed
                        line=dict(color='rgba(0,128,0,0)'),  # transparent line to keep the focus on the fill
                        name='Діапазон можливих значень прогнозу'
                    ))

                    # Finally, add the median (50% quantile) forecast line on top.
                    st.session_state.plotp2.add_trace(go.Scatter(
                        x=last_days[st.session_state.date][:(slid)],
                        y=q50,
                        mode='lines',
                        name='Прогноз',
                        line=dict(color='green')
                    ))
                    # Update layout (optional)
                    st.session_state.plotp2.update_layout(
                        xaxis_title='Дата',
                        # yaxis_title='Значення',
                        yaxis=dict(
                            # range=[min_value, max_value],
                            title='Спрогнозовані значення'  # Optional: add a title for clarity
                        ),
                        title="Графік прогнозу",  # Increase the overall height
                    )

                    # Show the plot
                    st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                    # Plot the data except the last seven days

                    st.session_state.bp2 = go.Figure()

                    st.session_state.bp2.add_trace(go.Bar(
                        x=last_days[st.session_state.date][:(slid)],
                        y=last_days[st.session_state.target][:(slid)],
                        name='Прогноз',
                        marker_color='green'
                    ))

                    # Customize layout
                    st.session_state.bp2.update_layout(
                        title='Барплот прогнозу',
                        xaxis_title='Дата',
                        yaxis_title='Значення',
                        template='plotly_white'
                    )

                    # Display the Plotly chart in Streamlit
                    st.plotly_chart(st.session_state.bp2, use_container_width=True)


else:
    st.warning('Для початки роботи з моделями, оберіть дані', icon="⚠️")
