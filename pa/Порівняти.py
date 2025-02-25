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
import time


if 'sel1' not in st.session_state:
    st.session_state.sel1 = []
if 'sel2' not in st.session_state:
    st.session_state.sel2 = []

if 'ready1' not in st.session_state:
    st.session_state.ready1 = []
if 'ready2' not in st.session_state:
    st.session_state.ready2 = []


means = {"Місяць": "M",
         "Година": "h",
         "Рік": "Y",
         "Хвилина": "T",
         "Секунда": "S",
         "День": "D",
         }


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3, hor=7):
    # first_date = first_date_str
    # last_date = last_date_str
    first_date = pd.to_datetime(first_date_str)
    last_date = pd.to_datetime(last_date_str)

    target_date = first_date

    dataframe.index = dataframe.pop('ds')

    dates = []
    X, Y = [], []

    while target_date <= last_date:
        df_subset = dataframe.loc[:target_date].tail(n)

        if len(df_subset) != n:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            break

        values = df_subset['y'].to_numpy()
        x = values

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=hor)].head(hor)

        if len(next_week) < hor:
            print(f'Error: Not enough data for 7-day target at date {target_date}')
            break

        y = next_week['y'].to_numpy()

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        target_date += datetime.timedelta(days=1)

    ret_df = pd.DataFrame({'Target Date': dates})

    X = np.array(X)
    for i in range(n):
        ret_df[f'Target-{n - i}'] = X[:, i]

    Y = np.array(Y)
    for i in range(hor):
        ret_df[f'Day+{i + 1}'] = Y[:, i]

    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe, hor):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]  # Extract dates

    # Extract the input matrix (X) excluding target columns
    middle_matrix = df_as_np[:, 1:-hor]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    # Extract the 7-day target matrix (Y)
    Y = df_as_np[:, -hor:]

    return dates, X.astype(np.float32), Y.astype(np.float32)



def save1(name, datafra, iter, horizon, rarety, inp):
    st.session_state.sel1 = [name, datafra, iter, horizon, rarety, inp]


def save2(name, datafra, iter, horizon, rarety, inp):
    st.session_state.sel2 = [name, datafra, iter, horizon, rarety, inp]





# @st.cache_data(show_spinner="Робимо прогнозування...")
def submit_data_KAN(datafra, iter, horizon, rarety, inp, typ):
    if st.session_state.date_not_n:
        print("no date")
        print(datafra)
        start_date = pd.to_datetime('2024-01-01')
        rarety = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(rarety)
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)
    try:
        q = int(round(len(datafra) * 0.05, 0))
        fcst = NeuralForecast(
            models=[
                KAN(h=horizon,
                    input_size=int(round(inp, 0)),
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        start = time.time()
        fcst.fit(df=Y_train_df)
        end = time.time()
        forecasts = fcst.predict(futr_df=Y_test_df)
        mse = mean_squared_error(Y_test_df["y"], forecasts["KAN"])

        print(len(forecasts["KAN"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["KAN"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["KAN"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        fig = go.Figure()

        # Plot the data except the last seven days
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["real"],
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))

        # Plot the last seven days in a different color
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["pred"],
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
        # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
        if typ == "l1":
            st.session_state.ready1 = [mse, float(end - start), fig]
        if typ == "l2":
            st.session_state.ready2 = [mse, float(end - start), fig]
    except Exception as ex:
        print(ex)
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


# @st.cache_data(show_spinner="Робимо прогнозування...")
def submit_data_SNN(datafra, iter, horizon, rarety, inp, typ):
    if st.session_state.date_not_n:
        print("no date")
        print(datafra)
        start_date = pd.to_datetime('2024-01-01')
        rarety = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra['ds'] = pd.to_datetime(datafra['ds'])

    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    datafra = datafra.set_index('ds').asfreq(rarety)
    datafra = datafra.reset_index()

    print(datafra)



    print(datafra["ds"].tolist()[inp])
    print(datafra["ds"].tolist()[-(horizon + 1)])

    windowed_df = df_to_windowed_df(datafra,
                                    datafra["ds"].tolist()[inp],
                                    datafra["ds"].tolist()[-(horizon+1)],
                                    n=inp,
                                    hor=horizon)

    dates, X, Y = windowed_df_to_date_X_y(windowed_df, hor=horizon)

    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]


    try:

        scaler = RobustScaler()
        orig_shape = X_train.shape
        print(X_train.reshape(-1, X_train.shape[-1]))
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(orig_shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        st.session_state.scaler = scaler
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(1)
        if len(y_test.shape) == 1:
            y_test = y_test.unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        def create_reservoir(input_dim, reservoir_size, spectral_radius=0.9):
            W_in = torch.randn(reservoir_size, input_dim) * 0.1
            W_res = torch.randn(reservoir_size, reservoir_size)

            eigenvalues = torch.linalg.eigvals(W_res)
            max_eigenvalue = torch.max(torch.abs(eigenvalues))
            W_res = W_res * (spectral_radius / max_eigenvalue)

            return W_in.to(device), W_res.to(device)

        input_dim = X_train.shape[1]
        print(input_dim)
        reservoir_size = 400

        W_in, W_res = create_reservoir(input_dim, reservoir_size)
        # SNN parameters
        beta = 0.5
        time_steps = 150
        spike_grad = surrogate.fast_sigmoid()

        # Define the SNN model
        class SNNRegression(nn.Module):
            def __init__(self, reservoir_size, output_size):
                super(SNNRegression, self).__init__()
                self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
                self.bn1 = nn.BatchNorm1d(reservoir_size)
                self.tcn1 = nn.Sequential(
                    nn.Conv1d(in_channels=reservoir_size, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(128)
                )

                self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
                self.bn2 = nn.BatchNorm1d(128)
                self.tcn2 = nn.Sequential(
                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Conv1d(in_channels=256, out_channels=output_size, kernel_size=3, padding=1)
                )

            def forward(self, x):
                x = x.reshape(x.size(0), -1)

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

        output_dim = y_train.shape[1]
        model = SNNRegression(reservoir_size, output_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)

        # Data loaders
        batch_size = 32
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        epochs = iter
        train_losses = []
        val_losses = []

        start = time.time()
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Reservoir computation
                reservoir_state = []
                for x in X_batch:
                    x = x.unsqueeze(0)  # Ensure x has a batch dimension
                    res_state = torch.tanh(W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
                    reservoir_state.append(res_state.squeeze(1))
                reservoir_state = torch.stack(reservoir_state).to(device)

                output = model(reservoir_state)
                loss = criterion(output, y_batch)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    reservoir_state = []
                    for x in X_batch:
                        x = x.unsqueeze(0)
                        res_state = torch.tanh(W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
                        reservoir_state.append(res_state.squeeze(1))
                    reservoir_state = torch.stack(reservoir_state).to(device)

                    output = model(reservoir_state)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        end = time.time()
        X_test, y_test = X_test.to(device), y_test.to(device)
        model.eval()
        with torch.no_grad():
            reservoir_state = []
            for x in X_test:
                x = x.unsqueeze(0)
                res_state = torch.tanh(W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
                reservoir_state.append(res_state.squeeze(1))
            reservoir_state = torch.stack(reservoir_state).to(device)

            predictions = model(reservoir_state)
            test_loss = criterion(predictions, y_test)
            loses = []
            for i in range(len(predictions)):
                loses.append(criterion(predictions[i], y_test[i]))
            print(f"{min(loses):.4f}")
            ind = 0
            for i in range(len(loses)):
                if min(loses) == loses[i]: ind = i

        print(f"Test Loss: {test_loss:.4f}")


        mse = float(f"{min(loses):.4f}")
        sample_idx = ind
        single_prediction = predictions[sample_idx].cpu().numpy()
        single_y_test = y_test[sample_idx].cpu().numpy()
        times = [i for i in range(1, len(single_prediction)-1)]
        # Create distplot with custom bin_size
        fig = go.Figure()

        # Plot the data except the last seven days
        fig.add_trace(go.Scatter(
            x=times,
            y=single_y_test,
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))

        # Plot the last seven days in a different color
        fig.add_trace(go.Scatter(
            x=times,
            y=single_prediction,
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
        # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        if typ == "l1":
            st.session_state.ready1 = [mse, float(end - start), fig]
        if typ == "l2":
            st.session_state.ready2 = [mse, float(end - start), fig]
    except Exception as ex:
        print(ex)
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")



# @st.cache_data(show_spinner="Робимо прогнозування...")
def submit_data_TN(datafra, iter, horizon, rarety, inp, typ):
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        rarety = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(rarety)
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)
    try:
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                TimesNet(h=horizon,
                         input_size=int(round(inp, 0)),
                         # output_size=horizon,
                         max_steps=iter,
                         scaler_type='standard',
                         start_padding_enabled=True
                         ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        start = time.time()
        fcst.fit(df=Y_train_df)
        end = time.time()
        forecasts = fcst.predict(futr_df=Y_test_df)
        mse = mean_squared_error(Y_test_df["y"], forecasts["TimesNet"])
        print(len(forecasts["TimesNet"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["TimesNet"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["TimesNet"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        fig = go.Figure()

        # Plot the data except the last seven days
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["real"],
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))

        # Plot the last seven days in a different color
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["pred"],
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
        print(dpred)
        if typ == "l1":
            st.session_state.ready1 = [mse, float(end - start), fig]
        if typ == "l2":
            st.session_state.ready2 = [mse, float(end - start), fig]
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")



# @st.cache_data(show_spinner="Робимо прогнозування...")
def submit_data_TM(datafra, iter, horizon, rarety, inp, typ):
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        rarety = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(rarety)
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)
    try:
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                TimeMixer(h=horizon,
                          input_size=int(round(inp, 0)),
                          # output_size=horizon,
                          max_steps=iter,
                          scaler_type='standard',
                          start_padding_enabled=True,
                          n_series=1
                          ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        start = time.time()
        fcst.fit(df=Y_train_df)
        end = time.time()
        forecasts = fcst.predict(futr_df=Y_test_df)
        mse = mean_squared_error(Y_test_df["y"], forecasts["TimeMixer"])

        print(len(forecasts["TimeMixer"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["TimeMixer"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["TimeMixer"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        fig = go.Figure()

        # Plot the data except the last seven days
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["real"],
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))

        # Plot the last seven days in a different color
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["pred"],
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
        print(dpred)
        if typ == "l1":
            st.session_state.ready1 = [mse, float(end - start), fig]
        if typ == "l2":
            st.session_state.ready2 = [mse, float(end - start), fig]
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


# @st.cache_data(show_spinner="Робимо прогнозування...")
def submit_data_PTST(datafra, iter, horizon, rarety, inp, typ):
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        rarety = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(rarety)
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)
    try:
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                PatchTST(h=horizon,
                         input_size=int(round(inp, 0)),
                         # output_size=horizon,
                         max_steps=iter,
                         scaler_type='standard',
                         start_padding_enabled=True
                         ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        start = time.time()
        fcst.fit(df=Y_train_df)
        end = time.time()
        forecasts = fcst.predict(futr_df=Y_test_df)
        mse = mean_squared_error(Y_test_df["y"], forecasts["PatchTST"])
        print(len(forecasts["PatchTST"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["PatchTST"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["PatchTST"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        fig = go.Figure()

        # Plot the data except the last seven days
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["real"],
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))

        # Plot the last seven days in a different color
        fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["pred"],
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
        print(dpred)
        if typ == "l1":
            st.session_state.ready1 = [mse, float(end - start), fig]
        if typ == "l2":
            st.session_state.ready2 = [mse, float(end - start), fig]
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


# @st.cache_data(show_spinner="Робимо прогнозування...")
def submit_data_NBx(datafra, iter, horizon, rarety, inp, typ):
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        rarety = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(rarety)
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)

    # try:
    q = int(round(len(datafra) * 0.1, 0))
    fcst = NeuralForecast(
        models=[
            NBEATSx(h=100000,
                    input_size=89000,
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
        ],
        freq=rarety
    )

    Y_train_df = datafra[:-horizon]
    Y_test_df = datafra[-horizon:]
    start = time.time()
    fcst.fit(df=Y_train_df)
    end = time.time()
    forecasts = fcst.predict(futr_df=Y_test_df)
    mse = mean_squared_error(Y_test_df["y"], forecasts["NBEATSx"])
    print(len(forecasts["NBEATSx"]), len(Y_test_df["y"]))
    print(forecasts.columns)
    forecasts = forecasts.reset_index(drop=True)
    print(forecasts.columns.values)
    print(forecasts["NBEATSx"].values.tolist())
    dpred = pd.DataFrame()
    dpred["real"] = Y_test_df["y"]
    dpred["pred"] = forecasts["NBEATSx"].values.tolist()
    dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
    # Create distplot with custom bin_size
    fig = go.Figure()

    # Plot the data except the last seven days
    fig.add_trace(go.Scatter(
        x=dpred["unique_id"],
        y=dpred["real"],
        mode='lines',
        name='Дані',
        line=dict(color='blue')
    ))

    # Plot the last seven days in a different color
    fig.add_trace(go.Scatter(
        x=dpred["unique_id"],
        y=dpred["pred"],
        mode='lines',
        name='Прогноз',
        line=dict(color='green')
    ))
    print(dpred)
    if typ == "l1":
        st.session_state.ready1 = [mse, float(end - start), fig]
    if typ == "l2":
        st.session_state.ready2 = [mse, float(end - start), fig]
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")


# @st.cache_data(show_spinner="Робимо порівняння...")
def save(l1, l2):
    print(l1[0])
    print(l2[0])
    if l1[0] == "KAN":
        submit_data_KAN(l1[1], l1[2], l1[3], l1[4], l1[5], "l1")
    if l1[0] == "NBEATSx":
        submit_data_NBx(l1[1], l1[2], l1[3], l1[4], l1[5], "l1")
    if l1[0] == "TimeMixer":
        submit_data_TM(l1[1], l1[2], l1[3], l1[4], l1[5], "l1")
    if l1[0] == "TimesNet":
        submit_data_TN(l1[1], l1[2], l1[3], l1[4], l1[5], "l1")
    if l1[0] == "SNN":
        print(l1)
        submit_data_SNN(l1[1], l1[2], l1[3], l1[4], l1[5], "l1")
    if l1[0] == "PatchTST":
        submit_data_PTST(l1[1], l1[2], l1[3], l1[4], l1[5], "l1")
        #__________________________________________________________________________________________________________________________
    if l2[0] == "KAN":
        print(l2)
        submit_data_KAN(l2[1], l2[2], l2[3], l2[4], l2[5], "l2")
    if l2[0] == "NBEATSx":
        submit_data_NBx(l2[1], l2[2], l2[3], l2[4], l2[5], "l2")
    if l2[0] == "TimeMixer":
        submit_data_TM(l2[1], l2[2], l2[3], l2[4], l2[5], "l2")
    if l2[0] == "TimesNet":
        submit_data_TN(l2[1], l2[2], l2[3], l2[4], l2[5], "l2")
    if l2[0] == "SNN":
        submit_data_SNN(l2[1], l2[2], l2[3], l2[4], l2[5], "l2")
    if l2[0] == "PatchTST":
        submit_data_PTST(l2[1], l2[2], l2[3], l2[4], l2[5], "l2")







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
        st.title("Порівняти моделі")

    c1, c2 = st.columns(2)
    with c1:
        option = st.selectbox(
            "Оберіть модель",
            ["KAN", "TimesNet", "NBEATSx", "TimeMixer", "PatchTST", "SNN"],
            key="one"
        )
        # try:
        if option == "KAN":
            st.markdown("## Ви обрали модель KAN")
            st.markdown(
                "### KAN — це нейронна мережа, що застосовує апроксимаційну теорему Колмогорова-Арнольда, яка стверджує, що сплайни можуть апроксимувати складніші функції.")
            st.divider()
            horizon = st.select_slider(
                "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                options=[i for i in range(1, 101)],
                key="one1"
            )
            iter = st.select_slider(
                "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                options=[i for i in range(5, 101)],
                key="one2"

            )
            inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                  max_value=150, key="one3")

            st.button(label="Підтвердити", key="one4", on_click=save1,
                      args=("KAN", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        # except Exception as ex:
        #     print(ex)
        #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")

        try:
            if option == "NBEATSx":
                st.markdown("## Ви обрали модель NBEATSx")
                st.markdown(
                    "### NBEATSx — це глибока нейронна архітектура на основі MLP, яка використовує прямі та зворотні залишкові зв'язки.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="one5"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="one6"
                )

                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="one7")
                st.button(label="Підтвердити", key="one16", on_click=save1,
                          args=("NBEATSx", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option == "TimesNet":
                st.markdown("## Ви обрали модель TimesNet")

                st.markdown(
                    "### TimesNet — це модель на основі CNN, яка ефективно вирішує завдання моделювання як внутрішньоперіодних, так і міжперіодних змін у часових рядах.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="one8"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="one9"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="one10")
                st.button(label="Підтвердити", key="one", on_click=save1,
                          args=("TimesNet", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option == "TimeMixer":
                st.markdown("## Ви обрали модель TimeMixer")
                st.markdown(
                    "### TimeMixer - модель, яка поєднує елементи архітектури Transformers і CNN для досягнення високої точності в прогнозах, обробляючи залежності як в просторі, так і в часі.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="one11"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="one12"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="one13")
                st.button(label="Підтвердити", key="one15", on_click=save1,
                          args=("TimeMixer", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option == "PatchTST":
                st.markdown("## Ви обрали модель PatchTST")
                st.markdown(
                    "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="one14"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="one17"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="one18")
                st.button(label="Підтвердити", key="one19", on_click=save1,
                          args=("PatchTST", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option == "SNN":
                st.markdown("## Ви обрали модель SNN")
                st.markdown(
                    "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="one20"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="one21"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="one22")
                st.button(label="Підтвердити", key="one23", on_click=save1,
                          args=("SNN", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")

        st.write("You selected:", option)
    with c2:
        option2 = st.selectbox(
            "Оберіть модель",
            ["KAN", "TimesNet", "NBEATSx", "TimeMixer", "PatchTST", "SNN"],
            key="two"
        )
        # try:
        if option2 == "KAN":
            st.markdown("## Ви обрали модель KAN")
            st.markdown(
                "### KAN — це нейронна мережа, що застосовує апроксимаційну теорему Колмогорова-Арнольда, яка стверджує, що сплайни можуть апроксимувати складніші функції.")
            st.divider()
            horizon = st.select_slider(
                "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                options=[i for i in range(1, 101)], key="two1"
            )
            iter = st.select_slider(
                "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                options=[i for i in range(5, 101)], key="two2"
            )
            inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                  max_value=150, key="two3")

            st.button(label="Підтвердити", key="two4", on_click=save2,
                      args=("KAN", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        # except Exception as ex:
        #     print(ex)
        #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")

        try:
            if option2 == "NBEATSx":
                st.markdown("## Ви обрали модель NBEATSx")
                st.markdown(
                    "### NBEATSx — це глибока нейронна архітектура на основі MLP, яка використовує прямі та зворотні залишкові зв'язки.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="two5"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="two6"
                )

                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="two7")
                st.button(label="Підтвердити", key="two8", on_click=save2,
                          args=("NBEATSx", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option2 == "TimesNet":
                st.markdown("## Ви обрали модель TimesNet")

                st.markdown(
                    "### TimesNet — це модель на основі CNN, яка ефективно вирішує завдання моделювання як внутрішньоперіодних, так і міжперіодних змін у часових рядах.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="two9"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="two0"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="two11")
                st.button(label="Підтвердити", key="two12", on_click=save2,
                          args=("TimesNet", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option2 == "TimeMixer":
                st.markdown("## Ви обрали модель TimeMixer")
                st.markdown(
                    "### TimeMixer - модель, яка поєднує елементи архітектури Transformers і CNN для досягнення високої точності в прогнозах, обробляючи залежності як в просторі, так і в часі.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="two13"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="two14"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="two15")
                st.button(label="Підтвердити", key="two16", on_click=save2,
                          args=("TimeMixer", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option2 == "PatchTST":
                st.markdown("## Ви обрали модель PatchTST")
                st.markdown(
                    "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="two17"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="two18"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="two19")
                st.button(label="Підтвердити", key="two20", on_click=save2,
                          args=("PatchTST", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if option2 == "SNN":
                st.markdown("## Ви обрали модель SNN")
                st.markdown(
                    "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)], key="two21"
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)], key="two22"
                )
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                      max_value=150, key="two23")
                st.button(label="Підтвердити", key="two24", on_click=save2,
                          args=("SNN", ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")

        st.write("You selected:", option2)

    if st.session_state.sel1 != [] and st.session_state.sel2 != []:
        st.divider()
        st.markdown("### ")
        with st.container():
            st.title("Порівняння")
            st.button(label="Порівняти", key="drt", on_click=save,
                      args=(st.session_state.sel1, st.session_state.sel2))

        if st.session_state.ready1 != [] and st.session_state.ready2 != []:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### Середньоквадратичне відхилення обраної моделі(MSE): {st.session_state.ready1[0]}")
                st.markdown(f"### Час навчання: {st.session_state.ready1[1]}")
                st.plotly_chart(st.session_state.ready1[2], use_container_width=True)
                sum1 = st.session_state.ready1[0]*0.9 + st.session_state.ready1[1]*0.1
            with c2:
                st.markdown(f"### Середньоквадратичне відхилення обраної моделі(MSE): {st.session_state.ready2[0]}")
                st.markdown(f"### Час навчання: {st.session_state.ready2[1]}")
                st.plotly_chart(st.session_state.ready2[2], use_container_width=True)
                sum2 = st.session_state.ready2[0] * 0.9 + st.session_state.ready2[1] * 0.1

            if sum1 > sum2:
                c1, c2, c3 = st.columns(3)
                with c2:
                    print(sum1)
                    print(sum2)
                    st.markdown(f"### Краща модель: {option2}")
            if sum1 < sum2:
                c1, c2, c3 = st.columns(3)
                with c2:
                    print(sum1)
                    print(sum2)
                    st.markdown(f"### Краща модель: {option}")
    # try:
    # if model == "KAN":
    #     st.markdown("## Ви обрали модель KAN")
    #     st.markdown(
    #         "### KAN — це нейронна мережа, що застосовує апроксимаційну теорему Колмогорова-Арнольда, яка стверджує, що сплайни можуть апроксимувати складніші функції.")
    #     st.divider()
    #     horizon = st.select_slider(
    #         "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #         options=[i for i in range(1, 1051)]
    #     )
    #     iter = st.select_slider(
    #         "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #         options=[i for i in range(5, 101)]
    #     )
    #     inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
    #                           max_value=150)
    #
    #     st.button(label="Підтвердити", key="kan", on_click=submit_data_KAN,
    #               args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    # # except Exception as ex:
    # #     print(ex)
    # #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    #
    # try:
    #     if model == "NBEATSx":
    #         st.markdown("## Ви обрали модель NBEATSx")
    #         st.markdown(
    #             "### NBEATSx — це глибока нейронна архітектура на основі MLP, яка використовує прямі та зворотні залишкові зв'язки.")
    #         st.divider()
    #         horizon = st.select_slider(
    #             "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #             options=[i for i in range(1, 1051)]
    #         )
    #         iter = st.select_slider(
    #             "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #             options=[i for i in range(5, 101)]
    #         )
    #
    #         inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
    #                               max_value=2050)
    #         st.button(label="Підтвердити", key="kan", on_click=submit_data_NBx,
    #                   args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    # try:
    #     if model == "TimesNet":
    #         st.markdown("## Ви обрали модель TimesNet")
    #
    #         st.markdown(
    #             "### TimesNet — це модель на основі CNN, яка ефективно вирішує завдання моделювання як внутрішньоперіодних, так і міжперіодних змін у часових рядах.")
    #         st.divider()
    #         horizon = st.select_slider(
    #             "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #             options=[i for i in range(1, 151)]
    #         )
    #         iter = st.select_slider(
    #             "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #             options=[i for i in range(5, 101)]
    #         )
    #         inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
    #                               max_value=150)
    #         st.button(label="Підтвердити", key="kan", on_click=submit_data_TN,
    #                   args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    # try:
    #     if model == "TimeMixer":
    #         st.markdown("## Ви обрали модель TimeMixer")
    #         st.markdown(
    #             "### TimeMixer - модель, яка поєднує елементи архітектури Transformers і CNN для досягнення високої точності в прогнозах, обробляючи залежності як в просторі, так і в часі.")
    #         st.divider()
    #         horizon = st.select_slider(
    #             "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #             options=[i for i in range(1, 151)]
    #         )
    #         iter = st.select_slider(
    #             "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #             options=[i for i in range(5, 101)]
    #         )
    #         inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
    #                               max_value=150)
    #         st.button(label="Підтвердити", key="kan", on_click=submit_data_TM,
    #                   args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    # try:
    #     if model == "PatchTST":
    #         st.markdown("## Ви обрали модель PatchTST")
    #         st.markdown(
    #             "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
    #         st.divider()
    #         horizon = st.select_slider(
    #             "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #             options=[i for i in range(1, 151)]
    #         )
    #         iter = st.select_slider(
    #             "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #             options=[i for i in range(5, 101)]
    #         )
    #         inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
    #                               max_value=150)
    #         st.button(label="Підтвердити", key="kan", on_click=submit_data_PTST,
    #                   args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    # try:
    #     if model == "SNN":
    #         st.markdown("## Ви обрали модель SNN")
    #         st.markdown(
    #             "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
    #         st.divider()
    #         horizon = st.select_slider(
    #             "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #             options=[i for i in range(1, 151)]
    #         )
    #         iter = st.select_slider(
    #             "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #             options=[i for i in range(5, 101)]
    #         )
    #         inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
    #                               max_value=150)
    #         st.button(label="Підтвердити", key="kan", on_click=submit_data_SNN,
    #                   args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    # try:
    #     if model == "Авто-вибір":
    #         st.markdown("## Ви обрали Авто-вибір")
    #         st.markdown(
    #             "### Тут обирається модель, яка найкраще може працювати з Вашими даними та налаштовуються гіперпараметри для моделі.")
    #         st.divider()
    #         horizon = st.select_slider(
    #             "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
    #             options=[i for i in range(1, 151)]
    #         )
    #         iter = st.select_slider(
    #             "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
    #             options=[i for i in range(5, 101)]
    #         )
    #         st.button(label="Підтвердити", key="kan", on_click=submit_data_auto,
    #                   args=(ds_for_pred, iter, horizon, means[st.session_state.freq]))
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")
    # st.divider()
    # if st.session_state.fig is not None:
    #     if st.session_state.inst_name != "Авто-вибір":
    #         st.markdown(
    #             f"## Середньоквадратичне відхилення обраної моделі ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")
    #
    #         st.session_state.fig.update_layout(
    #             xaxis_title='',
    #             yaxis_title='Значення'
    #         )
    #         st.plotly_chart(st.session_state.fig, use_container_width=True)
    #     else:
    #         st.markdown(
    #             f"## Середньоквадратичне відхилення обраної моделі авто-вибором ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")
    #         st.plotly_chart(st.session_state.fig, use_container_width=True)

    # st.button("Train", type="primary", on_click=save_performance, args=((model, k)))
    #
    # with st.expander("See full dataset"):
    #     st.write(wine_df)
    #
    # if len(st.session_state['score']) != 0:
    #     st.subheader(f"The model has an F1-Score of: {st.session_state['score'][-1]}")
else:
    st.warning('Для початки роботи з моделями, оберіть дані', icon="⚠️")