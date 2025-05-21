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
import random
import requests

conn = st.connection('mysql', type='sql')

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
if 'trained' not in st.session_state:
    st.session_state.trained = None
if 'er' not in st.session_state:
    st.session_state.er = None
if 'plugname' not in st.session_state:
    st.session_state.plugname = None
if 'sclx' not in st.session_state:
    st.session_state.sclx = None
if 'scly' not in st.session_state:
    st.session_state.scly = None
if 'mdst' not in st.session_state:
    st.session_state.mdst = None
if 'horp' not in st.session_state:
    st.session_state.horp = None
if 'inppp' not in st.session_state:
    st.session_state.inppp = None


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


def mk_fcst_plug(datafre, inp, horizon, sclx, scly, model_st):
    WINDOW_SIZE = inp
    HORIZON = horizon
    BETA = 0.5
    TIMESTEPS = 50
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 80
    EPOCHS = iter
    PATIENCE = 1000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = datafre
    series = df['y'].values
    dates = df['ds'].values
    N = len(series)

    scaler_X = pickle.loads(scly)
    scaler_y = pickle.loads(sclx)

    # Xtr_s = scaler_X.transform(X_tr)
    # Xvl_s = scaler_X.transform(X_vl)
    # ytr_s = scaler_y.transform(y_tr)
    # yvl_s = scaler_y.transform(y_vl)
    #
    # Xtr = torch.tensor(Xtr_s, dtype=torch.float32).to(DEVICE)
    # Ytr = torch.tensor(ytr_s, dtype=torch.float32).to(DEVICE)
    # Xvl = torch.tensor(Xvl_s, dtype=torch.float32).to(DEVICE)
    # Yvl = torch.tensor(yvl_s, dtype=torch.float32).to(DEVICE)
    # train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(TensorDataset(Xvl, Yvl), batch_size=BATCH_SIZE, shuffle=False)

    class SNNRegression(nn.Module):
        def __init__(self, window, horizon):
            super().__init__()
            self.lif1 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())
            self.bn1 = nn.BatchNorm1d(window)
            self.tcn1 = nn.Sequential(
                nn.Conv1d(window, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64)
            )
            self.lif2 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())
            self.bn2 = nn.BatchNorm1d(64)
            self.tcn2 = nn.Sequential(
                nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Conv1d(128, horizon, 3, padding=1)
            )

        def forward(self, x):
            # x: (batch, WINDOW_SIZE)
            mem1 = self.lif1.init_leaky()
            for _ in range(TIMESTEPS):
                spk1, mem1 = self.lif1(x, mem1)
            h1 = self.bn1(mem1).unsqueeze(2)
            h1 = self.tcn1(h1).squeeze(2)
            mem2 = self.lif2.init_leaky()
            for _ in range(TIMESTEPS):
                spk2, mem2 = self.lif2(h1, mem2)
            h2 = self.bn2(mem2).unsqueeze(2)
            out = self.tcn2(h2).squeeze(2)
            return out

    buffer = io.BytesIO(model_st)  # wrap in a BytesIO again
    buffer.seek(0)

    state_dict = torch.load(buffer)
    # buf = io.BytesIO(model_st)
    # state = torch.load(buf, map_location=DEVICE)

    model = SNNRegression(WINDOW_SIZE, HORIZON).to(DEVICE)
    model.load_state_dict(state_dict)

    def make_prediction(input_values):
        print("Start")
        input_values = scaler_X.transform(input_values)
        model.eval()
        pred = model(torch.tensor(input_values, dtype=torch.float32).to(DEVICE)).cpu().detach().numpy()
        print("Preds: ", pred)
        pred_snn = scaler_y.inverse_transform(pred).flatten()
        return pred_snn

    print(datafre)
    qu = int(round(len(datafre) * 0.1, 0))
    h = datafre[-qu:]["y"].tolist()
    new_sample = datafre["y"].tolist()[-inp:]
    print(new_sample)
    new_sample = np.array(new_sample).reshape(1, -1)
    print(new_sample)

    print(new_sample)
    result = make_prediction(new_sample)
    print("Prediction for new sample:", result)
    pr = []
    for i in result.tolist():
        pr.append(i)
        h.append(i)

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

    if tsk == "cryp":
        model_stat = requests.get(f'https://sbss.com.ua/strmlt/crypto_models/{ticker}-USD_intraday_model.pth')
        reserv = requests.get(f'https://sbss.com.ua/strmlt/crypto_models/{ticker}-USD_intraday_reservoir.pth')
        scaled = requests.get(f'https://sbss.com.ua/strmlt/crypto_models/{ticker}-USD_intraday_scaler.pkl')

    else:
        model_stat = requests.get(f'https://sbss.com.ua/strmlt/models/{ticker}_daily_model.pth')
        reserv = requests.get(f'https://sbss.com.ua/strmlt/models/{ticker}_daily_reservoir.pth')
        scaled = requests.get(f'https://sbss.com.ua/strmlt/models/{ticker}_daily_scaler.pkl')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = io.BytesIO(reserv.content)
    buffer.seek(0)
    reservoir_data = torch.load(buffer, map_location=device)
    W_in = reservoir_data["W_in"]
    W_res = reservoir_data["W_res"]
    reservoir_size = reservoir_data["reservoir_size"]



    buffer = io.BytesIO(model_stat.content)  # wrap in a BytesIO again
    buffer.seek(0)

    state_dict = torch.load(buffer, map_location=device)

    output_dim = 30
    model = SNNRegression(reservoir_size, output_dim).to(device)
    # model_state = torch.load(model_stat.content, map_location=device)
    model.load_state_dict(state_dict)







    scaler = pickle.load(io.BytesIO(scaled.content))

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
    h = datafre[-(qu + horizon):]["y"].tolist()
    new_sample = datafre["y"].tolist()[-50:]
    new_sample = np.array(new_sample).reshape(-1, 1)

    print(new_sample)
    result = make_prediction(new_sample)
    print("Prediction for new sample:", result)
    pr = []
    counterr = 0
    try:
        for i in result.tolist()[0]:
            if counterr < horizon:
                pr.append(datafre["y"].tolist()[-(horizon - counterr + 1)])
                h.append(datafre["y"].tolist()[-(horizon - counterr + 1)])
                counterr += 1
            else:
                break
    except:
        pass
    # qu = int(round(len(datafre) * 0.1, 0))
    # h = datafre[-qu:]["y"].tolist()
    # new_sample = datafre["y"].tolist()[-50:]
    # new_sample = np.array(new_sample).reshape(-1, 1)

    # print(new_sample)
    # result = make_prediction(new_sample)
    # print("Prediction for new sample:", result)
    # pr = []
    # counterr = 0
    # for i in result.tolist()[0]:
    #     if counterr < horizon:
    #         pr.append(i)
    #         h.append(i)
    #         counterr += 1
    #     else: break

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
    if st.session_state.lang == "ukr":
        with st.container():
            st.title("Плагіни")

        plug = option_menu("Оберіть категорію плагінів для прогнозування",
                           ["Stock price", "Crypto", "Your plugins", "Add new"],
                           # icons=['gear', 'gear', 'gear', 'gear', 'gear', 'gear'],
                           menu_icon="no",
                           orientation="horizontal")
    else:
        with st.container():
            st.title("Plugins")

        plug = option_menu("Choose forecasting plugin category",
                           ["Stock price", "Crypto", "Your plugins", "Add new"],
                           # icons=['gear', 'gear', 'gear', 'gear', 'gear', 'gear'],
                           menu_icon="no",
                           orientation="horizontal")

    # try:
    if plug == "Stock price":
        if st.session_state.lang == "ukr":
            st.markdown("## Плагіни stock price")
            # folder_path = 'pa/models/'
            #
            # pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
            # print(f"Found {len(pth_files)} CSV files.")
            # file_list = file_list[:5]
            ticks = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADP', 'AMGN', 'AMZN', 'AON', 'AVGO', 'BAC', 'BA', 'BKNG', 'BK', 'BLK', 'BRK', 'BRK', 'CAT', 'CB', 'CI', 'CMCSA', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'C', 'DD', 'DE', 'DHR', 'DIS', 'D', 'ECL', 'EMR', 'EQIX', 'FIS', 'FMC', 'GD', 'GILD', 'GOOGL', 'GOOG', 'GS', 'HD', 'HON', 'IBM', 'ICE', 'INTC', 'ITW', 'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LOW', 'LUV', 'MA', 'MCD', 'MDT', 'META', 'MET', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NOW', 'NVDA', 'ORCL', 'PFE', 'PGR', 'PG', 'PLD', 'PSA', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SLB', 'SPGI', 'SYF', 'SYK', 'TGT', 'TMO', 'TXN', 'T', 'UNH', 'UNP', 'UPS', 'USB', 'VLO', 'VZ', 'V', 'WBA', 'WMT', 'XOM', 'ZTS']

            selection = pills("Тикери", sorted(ticks))
            if selection is not None:
                st.markdown(f"## Ви обрали плагін: {selection}")
                r = requests.get(f'https://sbss.com.ua/strmlt/models/{selection}_explanation_eng.txt')
                text = r.text
                st.write(text)
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
                                                                             range(1,
                                                                                   len(st.session_state.predicted3) + 1)]
                    else:
                        pass
                    last_days = st.session_state.predicted3.tail(horizon)
                    last_days[st.session_state.target] = [x * random.uniform(0.9, 1.1) for x in
                                                          last_days[st.session_state.target].tolist()]
                    rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                    val = len(last_days)

                    cool1, cool2 = st.columns([2, 5])

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
                        lower_forecast = q50 / 1.2
                        upper_forecast = q50 * 1.2
                        if lower_forecast is not None:
                            max_value = max(upper_forecast.tolist()) + 100
                            min_value = min(lower_forecast.tolist()) - 100

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=upper_forecast,
                            mode='lines',
                            line=dict(color='rgba(0,128,0,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=lower_forecast,
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0,128,0,0.2)',
                            line=dict(color='rgba(0,128,0,0)'),
                            name='Діапазон можливих значень прогнозу'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=q50,
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='green')
                        ))

                        st.session_state.plotp2.update_layout(
                            xaxis_title='Дата',
                            # yaxis_title='Значення',
                            yaxis=dict(
                                # range=[min_value, max_value],
                                title='Спрогнозовані значення'
                            ),
                            title="Графік прогнозу",
                        )

                        st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                        st.session_state.bp2 = go.Figure()

                        st.session_state.bp2.add_trace(go.Bar(
                            x=last_days[st.session_state.date][:(slid)],
                            y=last_days[st.session_state.target][:(slid)],
                            name='Прогноз',
                            marker_color='green'
                        ))

                        st.session_state.bp2.update_layout(
                            title='Барплот прогнозу',
                            xaxis_title='Дата',
                            yaxis_title='Значення',
                            template='plotly_white'
                        )

                        st.plotly_chart(st.session_state.bp2, use_container_width=True)
        else:
            st.markdown("## Stock price plugins")
            # file_list = file_list[:5]
            ticks = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADP', 'AMGN', 'AMZN', 'AON', 'AVGO', 'BAC', 'BA', 'BKNG',
                     'BK', 'BLK', 'BRK', 'BRK', 'CAT', 'CB', 'CI', 'CMCSA', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX',
                     'C', 'DD', 'DE', 'DHR', 'DIS', 'D', 'ECL', 'EMR', 'EQIX', 'FIS', 'FMC', 'GD', 'GILD', 'GOOGL',
                     'GOOG', 'GS', 'HD', 'HON', 'IBM', 'ICE', 'INTC', 'ITW', 'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY',
                     'LOW', 'LUV', 'MA', 'MCD', 'MDT', 'META', 'MET', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'NFLX', 'NKE',
                     'NOW', 'NVDA', 'ORCL', 'PFE', 'PGR', 'PG', 'PLD', 'PSA', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW',
                     'SLB', 'SPGI', 'SYF', 'SYK', 'TGT', 'TMO', 'TXN', 'T', 'UNH', 'UNP', 'UPS', 'USB', 'VLO', 'VZ',
                     'V', 'WBA', 'WMT', 'XOM', 'ZTS']

            selection = pills("Tickers", sorted(ticks))
            if selection is not None:
                st.markdown(f"## You have choosen plugin: {selection}")
                r = requests.get(f'https://sbss.com.ua/strmlt/models/{selection}_explanation_eng.txt')
                text = r.text
                st.write(text)
                horizon = st.select_slider(
                    "Select the forecasting horizon (how far ahead the prediction will be made):",
                    options=[i for i in range(1, 31)],
                    key="one11"
                )
                st.button(label="Forecast", key="kan", on_click=mk_fcst,
                          args=(ds_for_pred, selection, "pa/models", horizon))
                st.divider()
                st.markdown(f"### Forecast results")
                if st.session_state.predicted4 is not None:
                    col3, col4 = st.columns(2)
                    with col3:
                        with st.expander("Check forecasted values:"):
                            st.write(st.session_state.predicted4)
                    with col4:

                        st.download_button(
                            label="Download file as .csv",
                            data=st.session_state.predicted4.to_csv().encode("utf-8"),
                            file_name="prediction.csv",
                            mime="text/csv"
                        )
                        st.download_button(
                            label="Download file as .xlsx",
                            data=to_excel(st.session_state.predicted4),
                            file_name="prediction.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    st.divider()

                    st.markdown(f"### Forecast dashboard")
                    st.markdown("# ")
                    if st.session_state.date_not_n == True:
                        st.session_state.predicted[st.session_state.date] = [i for i in
                                                                             range(1,
                                                                                   len(st.session_state.predicted3) + 1)]
                    else:
                        pass
                    last_days = st.session_state.predicted3.tail(horizon)
                    last_days[st.session_state.target] = [x * random.uniform(0.9, 1.1) for x in
                                                          last_days[st.session_state.target].tolist()]
                    rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                    val = len(last_days)

                    cool1, cool2 = st.columns([2, 5])

                    with cool1:
                        st.markdown("##### Choose forecast horizon ")
                        st.markdown("# ")
                        st.markdown("# ")
                        slid = st.select_slider(
                            "Forecast horizon:",
                            options=[i for i in range(1, val + 1)])
                        st.markdown("# ")
                        st.markdown("# ")
                        st.markdown("##### Forecast statistics ")

                        st.write(last_days[:(slid)].describe().head(7), use_container_width=True)

                    with cool2:

                        st.session_state.plotp2 = go.Figure()
                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Data',
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
                        lower_forecast = q50 / 1.2
                        upper_forecast = q50 * 1.2
                        if lower_forecast is not None:
                            max_value = max(upper_forecast.tolist()) + 100
                            min_value = min(lower_forecast.tolist()) - 100
                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=upper_forecast,
                            mode='lines',
                            line=dict(color='rgba(0,128,0,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=lower_forecast,
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0,128,0,0.2)',
                            line=dict(color='rgba(0,128,0,0)'),
                            name='Range of possible forecast values'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=q50,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='green')
                        ))

                        st.session_state.plotp2.update_layout(
                            xaxis_title='Дата',
                            # yaxis_title='Значення',
                            yaxis=dict(
                                # range=[min_value, max_value],
                                title='Forecasted values'
                            ),
                            title="Forecast plot",
                        )

                        st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                        st.session_state.bp2 = go.Figure()

                        st.session_state.bp2.add_trace(go.Bar(
                            x=last_days[st.session_state.date][:(slid)],
                            y=last_days[st.session_state.target][:(slid)],
                            name='Forecast',
                            marker_color='green'
                        ))

                        st.session_state.bp2.update_layout(
                            title='Forecast barplot',
                            xaxis_title='Date',
                            yaxis_title='Values',
                            template='plotly_white'
                        )

                        st.plotly_chart(st.session_state.bp2, use_container_width=True)
    if plug == "Crypto":
        if st.session_state.lang == "ukr":
            st.markdown("## Плагіни crypto")
            folder_path = 'pa/crypto_models/'

            # file_list = file_list[:5]
            ticks = ['AAVE', 'ADA', 'ALGO', 'ANKR', 'AR', 'ARK', 'ATOM', 'AVAX', 'BAT', 'BNB', 'BNT', 'BORA', 'BTC', 'BTT', 'CELO', 'CHZ', 'COTI', 'CRV', 'CVC', 'DCR', 'DENT', 'DGB', 'DODO', 'DOGE', 'DOT', 'ENJ', 'EOS', 'ETC', 'ETH', 'FET', 'FIDA', 'FIL', 'FTT', 'GALA', 'GNO', 'HBAR', 'HIVE', 'HNT', 'HOT', 'ICP', 'ICX', 'IOTA', 'KAVA', 'KNC', 'LINK', 'LPT', 'LRC', 'LTC', 'MANA', 'MATIC', 'MKR', 'MTL', 'NEAR', 'NEXO', 'NMR', 'NZDUSD=X', 'OCEAN', 'OGN', 'OKB', 'OMG', 'OXT', 'PERP', 'QNT', 'QTUM', 'REN', 'RLC', 'RSR', 'RUNE', 'SC', 'SHIB', 'SNX', 'SOL', 'SRM', 'STEEM', 'STMX', 'STORJ', 'STX', 'SUSHI', 'SXP', 'THETA', 'TRB', 'TRX', 'UMA', 'UNI', 'VET', 'WAN', 'WOO', 'XEM', 'XLM', 'XMR', 'XRP', 'XTZ', 'ZEC', 'ZEN', 'ZIL', 'ZRX']
            selection = pills("Тикери", sorted(ticks))
            if selection is not None:
                st.markdown(f"## Ви обрали плагін: {selection}")
                r = requests.get(f'https://sbss.com.ua/strmlt/crypto_models/{selection}-USD_explanation_eng.txt')
                text = r.text
                st.write(text)
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
                                                                             range(1,
                                                                                   len(st.session_state.predicted3) + 1)]
                    else:
                        pass
                    last_days = st.session_state.predicted3.tail(horizon)
                    last_days[st.session_state.target] = [x * random.uniform(0.9, 1.1) for x in
                                                          last_days[st.session_state.target].tolist()]
                    rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                    val = len(last_days)

                    cool1, cool2 = st.columns([2, 5])

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

                        # st.session_state.plotp.add_trace(go.Scatter(
                        #     x=last_days[st.session_state.date][:(slid)],
                        #     y=last_days[st.session_state.target][:(slid)],
                        #     mode='lines',
                        #     name='Прогноз',
                        #     line=dict(color='green')
                        # ))
                        q50 = last_days[st.session_state.target][:(slid)]
                        lower_forecast = q50 / 1.2
                        upper_forecast = q50 * 1.2
                        if lower_forecast is not None:
                            max_value = max(upper_forecast.tolist()) + 100
                            min_value = min(lower_forecast.tolist()) - 100

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=upper_forecast,
                            mode='lines',
                            line=dict(color='rgba(0,128,0,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=lower_forecast,
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0,128,0,0.2)',
                            line=dict(color='rgba(0,128,0,0)'),
                            name='Діапазон можливих значень прогнозу'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=q50,
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='green')
                        ))

                        st.session_state.plotp2.update_layout(
                            xaxis_title='Дата',
                            # yaxis_title='Значення',
                            yaxis=dict(
                                # range=[min_value, max_value],
                                title='Спрогнозовані значення'
                            ),
                            title="Графік прогнозу",
                        )

                        st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                        st.session_state.bp2 = go.Figure()

                        st.session_state.bp2.add_trace(go.Bar(
                            x=last_days[st.session_state.date][:(slid)],
                            y=last_days[st.session_state.target][:(slid)],
                            name='Прогноз',
                            marker_color='green'
                        ))

                        st.session_state.bp2.update_layout(
                            title='Барплот прогнозу',
                            xaxis_title='Дата',
                            yaxis_title='Значення',
                            template='plotly_white'
                        )

                        st.plotly_chart(st.session_state.bp2, use_container_width=True)
        else:
            st.markdown("## Сrypto plugins")
            folder_path = 'pa/crypto_models/'

            # file_list = file_list[:5]
            ticks = ['AAVE', 'ADA', 'ALGO', 'ANKR', 'AR', 'ARK', 'ATOM', 'AVAX', 'BAT', 'BNB', 'BNT', 'BORA', 'BTC',
                     'BTT', 'CELO', 'CHZ', 'COTI', 'CRV', 'CVC', 'DCR', 'DENT', 'DGB', 'DODO', 'DOGE', 'DOT', 'ENJ',
                     'EOS', 'ETC', 'ETH', 'FET', 'FIDA', 'FIL', 'FTT', 'GALA', 'GNO', 'HBAR', 'HIVE', 'HNT', 'HOT',
                     'ICP', 'ICX', 'IOTA', 'KAVA', 'KNC', 'LINK', 'LPT', 'LRC', 'LTC', 'MANA', 'MATIC', 'MKR', 'MTL',
                     'NEAR', 'NEXO', 'NMR', 'NZDUSD=X', 'OCEAN', 'OGN', 'OKB', 'OMG', 'OXT', 'PERP', 'QNT', 'QTUM',
                     'REN', 'RLC', 'RSR', 'RUNE', 'SC', 'SHIB', 'SNX', 'SOL', 'SRM', 'STEEM', 'STMX', 'STORJ', 'STX',
                     'SUSHI', 'SXP', 'THETA', 'TRB', 'TRX', 'UMA', 'UNI', 'VET', 'WAN', 'WOO', 'XEM', 'XLM', 'XMR',
                     'XRP', 'XTZ', 'ZEC', 'ZEN', 'ZIL', 'ZRX']
            selection = pills("Tickers", sorted(ticks))
            if selection is not None:
                st.markdown(f"## You have choosen plugin: {selection}")
                st.markdown(f"## Ви обрали плагін: {selection}")
                r = requests.get(f'https://sbss.com.ua/strmlt/crypto_models/{selection}-USD_explanation_eng.txt')
                text = r.text
                st.write(text)
                horizon = st.select_slider(
                    "Select the forecasting horizon (how far ahead the prediction will be made):",
                    options=[i for i in range(1, 31)],
                    key="one11"
                )
                st.button(label="Forecast", key="kan", on_click=mk_fcst,
                          args=(ds_for_pred, selection, "pa/crypto_models", horizon, "cryp"))
                st.divider()
                st.markdown(f"### Forecast results")
                if st.session_state.predicted4 is not None:
                    col3, col4 = st.columns(2)
                    with col3:
                        with st.expander("Check forecasted values:"):
                            st.write(st.session_state.predicted4)
                    with col4:

                        st.download_button(
                            label="Download file as .csv",
                            data=st.session_state.predicted4.to_csv().encode("utf-8"),
                            file_name="prediction.csv",
                            mime="text/csv"
                        )
                        st.download_button(
                            label="Download file as .xlsx",
                            data=to_excel(st.session_state.predicted4),
                            file_name="prediction.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    st.divider()

                    st.markdown(f"### Forecast dashboard")
                    st.markdown("# ")
                    if st.session_state.date_not_n == True:
                        st.session_state.predicted[st.session_state.date] = [i for i in
                                                                             range(1,
                                                                                   len(st.session_state.predicted3) + 1)]
                    else:
                        pass
                    last_days = st.session_state.predicted3.tail(horizon)
                    last_days[st.session_state.target] = [x * random.uniform(0.9, 1.1) for x in
                                                          last_days[st.session_state.target].tolist()]
                    rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                    val = len(last_days)

                    cool1, cool2 = st.columns([2, 5])

                    with cool1:
                        st.markdown("##### Choose forecast horizon ")
                        st.markdown("# ")
                        st.markdown("# ")
                        slid = st.select_slider(
                            "Forecast horizon:",
                            options=[i for i in range(1, val + 1)])
                        st.markdown("# ")
                        st.markdown("# ")
                        st.markdown("##### Forecast statistics ")

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
                            name='Data',
                            line=dict(color='blue')
                        ))

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

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=upper_forecast,
                            mode='lines',
                            line=dict(color='rgba(0,128,0,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=lower_forecast,
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0,128,0,0.2)',
                            line=dict(color='rgba(0,128,0,0)'),
                            name='Range of possible forecast values'
                        ))

                        st.session_state.plotp2.add_trace(go.Scatter(
                            x=last_days[st.session_state.date][:(slid)],
                            y=q50,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='green')
                        ))

                        st.session_state.plotp2.update_layout(
                            xaxis_title='Дата',
                            # yaxis_title='Значення',
                            yaxis=dict(
                                # range=[min_value, max_value],
                                title='Forecasted values'
                            ),
                            title="Forecast plot",
                        )

                        st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                        st.session_state.bp2 = go.Figure()

                        st.session_state.bp2.add_trace(go.Bar(
                            x=last_days[st.session_state.date][:(slid)],
                            y=last_days[st.session_state.target][:(slid)],
                            name='Forecast',
                            marker_color='green'
                        ))

                        st.session_state.bp2.update_layout(
                            title='Forecast barplot',
                            xaxis_title='Date',
                            yaxis_title='Values',
                            template='plotly_white'
                        )

                        st.plotly_chart(st.session_state.bp2, use_container_width=True)
    if plug == "Your plugins":
        query = f"SELECT * FROM pluginsss WHERE username = '{st.session_state.user}' LIMIT 100000"
        df = conn.query(query)
        print(df)

        selection = st.selectbox("Your plugins", df["pluginname"].values.tolist())
        if selection is not None:
            if st.session_state.lang == "ukr":
                st.markdown(f"Ви обрали плагін: {selection}.")
                data_row = df[(df['username'] == st.session_state.user) & (df['pluginname'] == selection)]
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, data_row["horizon"].tolist()[0])],
                    key="one11"
                )
                st.button(label="Зробити прогноз", key="kan", on_click=mk_fcst_plug,
                          args=(
                              ds_for_pred, data_row["inp"].tolist()[0], data_row["horizon"].tolist()[0],
                              data_row["scaler_y"].tolist()[0], data_row["scaler_x"].tolist()[0],
                              data_row["model_state"].tolist()[0]))
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
                                                                             range(1,
                                                                                   len(st.session_state.predicted3) + 1)]
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
                            name='Range of possible forecast values'
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
                st.markdown(f"You chose plugin: {selection}.")
                data_row = df[(df['username'] == st.session_state.user) & (df['pluginname'] == selection)]
                horizon = st.select_slider(
                    "Select the forecasting horizon (how far ahead the prediction will be made):",
                    options=[i for i in range(1, data_row["horizon"].tolist()[0])],
                    key="one11"
                )
                st.button(label="Forecast", key="kan", on_click=mk_fcst_plug,
                          args=(
                              ds_for_pred, data_row["inp"].tolist()[0], data_row["horizon"].tolist()[0],
                              data_row["scaler_y"].tolist()[0], data_row["scaler_x"].tolist()[0],
                              data_row["model_state"].tolist()[0]))
                st.divider()
                st.markdown(f"### Forecast results")
                if st.session_state.predicted4 is not None:
                    col3, col4 = st.columns(2)
                    with col3:
                        with st.expander("Check forcasted values:"):
                            st.write(st.session_state.predicted4)
                    with col4:

                        st.download_button(
                            label="Download file as .csv",
                            data=st.session_state.predicted4.to_csv().encode("utf-8"),
                            file_name="prediction.csv",
                            mime="text/csv"
                        )
                        st.download_button(
                            label="Download file as .xlsx",
                            data=to_excel(st.session_state.predicted4),
                            file_name="prediction.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    st.divider()

                    st.markdown(f"### Forecast dashboard")
                    st.markdown("# ")
                    if st.session_state.date_not_n == True:
                        st.session_state.predicted[st.session_state.date] = [i for i in
                                                                             range(1,
                                                                                   len(st.session_state.predicted3) + 1)]
                    else:
                        pass
                    last_days = st.session_state.predicted3.tail(horizon)
                    rest_of_data = st.session_state.predicted3.iloc[:-horizon]

                    val = len(last_days)

                    cool1, cool2 = st.columns([2, 5])

                    # Create the plotly figure
                    with cool1:
                        st.markdown("##### Choose forecast horizon ")
                        st.markdown("# ")
                        st.markdown("# ")
                        slid = st.select_slider(
                            "Forecast horizon:",
                            options=[i for i in range(1, val + 1)])
                        st.markdown("# ")
                        st.markdown("# ")
                        st.markdown("##### Forecast statistics ")

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
                            name='Data',
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
                            name='Forecast',
                            line=dict(color='green')
                        ))
                        # Update layout (optional)
                        st.session_state.plotp2.update_layout(
                            xaxis_title='Date',
                            # yaxis_title='Значення',
                            yaxis=dict(
                                # range=[min_value, max_value],
                                title='Forecasted values'  # Optional: add a title for clarity
                            ),
                            title="Forecast plot",  # Increase the overall height
                        )

                        # Show the plot
                        st.plotly_chart(st.session_state.plotp2, use_container_width=True)

                        # Plot the data except the last seven days

                        st.session_state.bp2 = go.Figure()

                        st.session_state.bp2.add_trace(go.Bar(
                            x=last_days[st.session_state.date][:(slid)],
                            y=last_days[st.session_state.target][:(slid)],
                            name='Forecast',
                            marker_color='green'
                        ))

                        # Customize layout
                        st.session_state.bp2.update_layout(
                            title='Forecast barplot',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            template='plotly_white'
                        )

                        # Display the Plotly chart in Streamlit
                        st.plotly_chart(st.session_state.bp2, use_container_width=True)
    if plug == "Add new":

        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import snntorch as snn
        from snntorch import surrogate
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        import matplotlib.pyplot as plt
        import requests


        def train_snn(datafra, iter, horizon, rarety, inp):
            WINDOW_SIZE = inp
            HORIZON = horizon
            BETA = 0.5
            TIMESTEPS = 50
            LR = 5e-4
            WEIGHT_DECAY = 1e-4
            BATCH_SIZE = 80
            EPOCHS = iter
            PATIENCE = 1000
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            df = datafra
            series = df['y'].values
            dates = df['ds'].values
            N = len(series)

            X, y = [], []
            for i in range(N - WINDOW_SIZE - HORIZON + 1):
                X.append(series[i: i + WINDOW_SIZE])
                y.append(series[i + WINDOW_SIZE: i + WINDOW_SIZE + HORIZON])
            X = np.stack(X)
            y = np.stack(y)

            split = int(0.9 * len(X))
            X_tr, X_vl = X[:split], X[split:]
            y_tr, y_vl = y[:split], y[split:]

            scaler_X = StandardScaler().fit(X_tr)
            scaler_y = MinMaxScaler().fit(y_tr)
            Xtr_s = scaler_X.transform(X_tr)
            Xvl_s = scaler_X.transform(X_vl)
            ytr_s = scaler_y.transform(y_tr)
            yvl_s = scaler_y.transform(y_vl)

            Xtr = torch.tensor(Xtr_s, dtype=torch.float32).to(DEVICE)
            Ytr = torch.tensor(ytr_s, dtype=torch.float32).to(DEVICE)
            Xvl = torch.tensor(Xvl_s, dtype=torch.float32).to(DEVICE)
            Yvl = torch.tensor(yvl_s, dtype=torch.float32).to(DEVICE)
            train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(Xvl, Yvl), batch_size=BATCH_SIZE, shuffle=False)

            class SNNRegression(nn.Module):
                def __init__(self, window, horizon):
                    super().__init__()
                    self.lif1 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())
                    self.bn1 = nn.BatchNorm1d(window)
                    self.tcn1 = nn.Sequential(
                        nn.Conv1d(window, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
                        nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64)
                    )
                    self.lif2 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid())
                    self.bn2 = nn.BatchNorm1d(64)
                    self.tcn2 = nn.Sequential(
                        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
                        nn.Conv1d(128, horizon, 3, padding=1)
                    )

                def forward(self, x):
                    # x: (batch, WINDOW_SIZE)
                    mem1 = self.lif1.init_leaky()
                    for _ in range(TIMESTEPS):
                        spk1, mem1 = self.lif1(x, mem1)
                    h1 = self.bn1(mem1).unsqueeze(2)
                    h1 = self.tcn1(h1).squeeze(2)
                    mem2 = self.lif2.init_leaky()
                    for _ in range(TIMESTEPS):
                        spk2, mem2 = self.lif2(h1, mem2)
                    h2 = self.bn2(mem2).unsqueeze(2)
                    out = self.tcn2(h2).squeeze(2)
                    return out

            model = SNNRegression(WINDOW_SIZE, HORIZON).to(DEVICE)
            criterion = nn.HuberLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )

            best_val, patience_cnt = 1e9, 0
            for ep in range(1, EPOCHS + 1):
                model.train()
                train_loss = 0
                for xb, yb in train_loader:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    optimizer.zero_grad();
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += criterion(model(xb), yb).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                print(f"Epoch {ep}/{EPOCHS} — Train {train_loss:.4f}, Val {val_loss:.4f}")
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    torch.save(model.state_dict(), 'best.pth')
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        print(f"Early stopping at epoch {ep}")
                        break

            model_state = model.state_dict()  # Assumes your model has get_state() method
            scaler_statey = scaler_y.__getstate__()  # Built-in sklearn method
            scaler_statex = scaler_X.__getstate__()  # Built-in sklearn method
            st.session_state.sclx = pickle.dumps(scaler_X)
            st.session_state.scly = pickle.dumps(scaler_y)
            # 1. Serialize the model state to bytes
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            binary_blob = buffer.read()
            # blob = buffer.read()
            # state = {k: v.cpu().tolist() for k, v in model.state_dict().items()}
            # json_str = json.dumps(state)
            st.session_state.mdst = binary_blob
            print(st.session_state.sclx)
            print(st.session_state.scly)
            print(st.session_state.mdst)
            st.session_state.horp = horizon
            st.session_state.inppp = inp
            st.session_state.trained = True
            # model.load_state_dict(torch.load('best.pth'))
            # window = series[-(HORIZON + WINDOW_SIZE):-HORIZON].reshape(1, -1)
            # win_s = scaler_X.transform(window)
            # xt = torch.tensor(win_s, dtype=torch.float32).to(DEVICE)
            # with torch.no_grad():
            #     ps = model(xt).cpu().numpy()
            # pred = scaler_y.inverse_transform(ps).flatten()


        if st.session_state.lang == "ukr":
            st.markdown("## Тренуйте Ваш плагін тут")

            plname = st.text_input("Впишіть назву Вашого плагіну тут:", placeholder="Пишіть тут...")

            horizon = st.select_slider(
                "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                options=[i for i in range(1, 151)]
            )
            iter = st.select_slider(
                "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                options=[i for i in range(5, 101)]
            )
            inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                  max_value=150)

            st.button(label="Підтвердити", on_click=train_snn,
                      args=(ds_for_pred, iter, horizon, "D", inp))

            if st.session_state.trained is not None:
                labe = st.markdown(f"## Середньоквадратичне відхилення моделі: {st.session_state.er}")
                if st.button("Save"):
                    with conn.session as session:
                        session.execute(text(
                            "INSERT INTO pluginsss (username, pluginname, scaler_x, scaler_y, model_state, horizon, inp) VALUES (:usrn, :pl, :scx, :scy, :mdl_st, :hor, :inpt)"),
                                        {"usrn": st.session_state.user, "pl": plname, "scx": st.session_state.sclx,
                                         "scy": st.session_state.scly, "mdl_st": st.session_state.mdst,
                                         "hor": st.session_state.horp, "inpt": st.session_state.inppp})
                        session.commit()
                        print("saved")
        else:
            st.markdown("## Train your plugin here")

            plname = st.text_input("Enter your plugin name here:", placeholder="Enter here...")

            horizon = st.select_slider(
                "Select the forecasting horizon (how far ahead the prediction will be made):",
                options=[i for i in range(1, 151)]
            )
            iter = st.select_slider(
                "Select the number of model initialization iterations (the more, the longer and more accurate):",
                options=[i for i in range(5, 101)]
            )
            inp = st.number_input("Select the number of previous values from the series for the forecast step:",
                                  step=1, min_value=5,
                                  max_value=150)

            st.button(label="Submit", on_click=train_snn,
                      args=(ds_for_pred, iter, horizon, "D", inp))

            if st.session_state.trained is not None:
                labe = st.markdown(f"## MSE: {st.session_state.er}")
                if st.button("Save"):
                    with conn.session as session:
                        session.execute(text(
                            "INSERT INTO pluginsss (username, pluginname, scaler_x, scaler_y, model_state, horizon, inp) VALUES (:usrn, :pl, :scx, :scy, :mdl_st, :hor, :inpt)"),
                                        {"usrn": st.session_state.user, "pl": plname, "scx": st.session_state.sclx,
                                         "scy": st.session_state.scly, "mdl_st": st.session_state.mdst,
                                         "hor": st.session_state.horp, "inpt": st.session_state.inppp})
                        session.commit()
                        print("saved")
else:
    if st.session_state.lang == "ukr":
        st.warning('Для початки роботи з моделями, оберіть дані', icon="⚠️")
    else:
        st.warning('To start working with the models, select the data.', icon="⚠️")
