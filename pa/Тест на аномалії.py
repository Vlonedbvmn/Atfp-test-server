import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import numpy as np
import io
import numpy as np
import datetime
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
# st.set_page_config(
#     page_title="Аналіз аномалій",
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
if 'fig_a' not in st.session_state:
    st.session_state.fig_a = None
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
if 'datanom' not in st.session_state:
    st.session_state.datanom = None
if 'df_anom' not in st.session_state:
    st.session_state.df_anom = None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


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

    dates = df_as_np[:, 0]  

    middle_matrix = df_as_np[:, 1:-hor]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -hor:]

    return dates, X.astype(np.float32), Y.astype(np.float32)


def anomal(datafra, freqs):
    with st.spinner('Проводимо тестування...'):
        if st.session_state.date_not_n:
            start_date = pd.to_datetime('2024-01-01')
            freqs = "D"
            datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, freqs)

        dafaf = datafra
        datafra['ds'] = pd.to_datetime(datafra['ds'])
        datafra = datafra.drop_duplicates(subset=['ds'])
        datafra = datafra.set_index('ds').asfreq(freqs)
        datafra = datafra.reset_index()
        datafra['y'] = datafra['y'].interpolate()
        datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
        print("s;kgfoshdisdifsdf")

        print(datafra)

        print(datafra)
        print(datafra["ds"].tolist()[7])

        windowed_df = df_to_windowed_df(datafra,
                                        datafra["ds"].tolist()[10],
                                        datafra["ds"].tolist()[-2],
                                        n=2,
                                        hor=40)
        print("jhgut")
        dates, X, Y = windowed_df_to_date_X_y(windowed_df, hor=40)

        q_80 = int(len(dates) * .8)
        q_90 = int(len(dates) * .9)

        dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]

        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
        dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]

        # scaler = RobustScaler()
        # orig_shape = X_train.shape
        # print(X_train.reshape(-1, X_train.shape[-1]))
        # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(orig_shape)
        # X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        # st.session_state.scaler = scaler
        # X_train = torch.tensor(X_train, dtype=torch.float32)
        # X_val = torch.tensor(X_val, dtype=torch.float32)
        # X_test = torch.tensor(X_test, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.float32)
        # y_val = torch.tensor(y_val, dtype=torch.float32)
        # y_test = torch.tensor(y_test, dtype=torch.float32)
        #
        # if len(y_train.shape) == 1:
        #     y_train = y_train.unsqueeze(1)
        # if len(y_val.shape) == 1:
        #     y_val = y_val.unsqueeze(1)
        # if len(y_test.shape) == 1:
        #     y_test = y_test.unsqueeze(1)
        #
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        # st.session_state.device = device
        #
        # def create_reservoir(input_dim, reservoir_size, spectral_radius=0.9):
        #     W_in = torch.randn(reservoir_size, input_dim) * 0.1
        #     W_res = torch.randn(reservoir_size, reservoir_size)
        #
        #     eigenvalues = torch.linalg.eigvals(W_res)
        #     max_eigenvalue = torch.max(torch.abs(eigenvalues))
        #     W_res = W_res * (spectral_radius / max_eigenvalue)
        #
        #     return W_in.to(device), W_res.to(device)
        #
        # input_dim = X_train.shape[1]
        # print(input_dim)
        # reservoir_size = 400
        #
        # W_in, W_res = create_reservoir(input_dim, reservoir_size)
        # st.session_state.reser_size = reservoir_size
        # st.session_state.win = W_in
        # st.session_state.wres = W_res
        # # SNN parameters
        # beta = 0.5
        # time_steps = 150
        # spike_grad = surrogate.fast_sigmoid()
        import requests
        import json
        # URL to your forecast endpoint (adjust domain/IP and port as needed)
        url = "https://207fvaz3j39ovs-8000.proxy.runpod.net/"

        dafaf['ds'] = dafaf['ds'].astype(str)
        inp = 2
        horizon = 40
        payload = {
            "data": dafaf.to_dict(orient='records'),
            "inp": inp,  # Example integer value for inp
            "horiz": horizon,  # Example integer value for horiz
            "iter": 45,  # Example integer value for iter
        }

        # Define the URL of your forecasting endpoint

        # Send a POST request with the JSON payload.
        response = requests.post(url, json=payload)

        # Check and print the response.
        data = ""
        if response.status_code == 200:
            data = response.json().get("predictions")
            # print("Forecast predictions:", data)
        else:
            print("Error:", response.status_code, response.text)

        # data = json.loads(json_data)

        # --- Reconstruct the PyTorch model ---
        # Instantiate a new model using the provided class and initialization parameters.
        # Rebuild the state dictionary: convert lists back into tensors.
        model_state_serialized = data["model_state"]
        model_state = {k: torch.tensor(v) for k, v in model_state_serialized.items()}
        # Load the state dictionary into the model.

        # --- Reconstruct the RobustScaler ---
        scaler_data = data["robust_scaler"]
        # Create a new RobustScaler instance using the saved parameters.
        robust_scaler = RobustScaler(**scaler_data["params"])
        # Set the fitted attributes if available.
        attributes = scaler_data.get("attributes", {})
        if "center_" in attributes:
            robust_scaler.center_ = np.array(attributes["center_"])
        if "scale_" in attributes:
            robust_scaler.scale_ = np.array(attributes["scale_"])

        # --- Retrieve and reconstruct the additional tensor values and integer ---
        tensor_data = data["int_values"]
        # Convert the lists back into PyTorch tensors.
        W_in = torch.tensor(tensor_data["W_in"])
        W_res = torch.tensor(tensor_data["W_res"])
        reservoir_size = tensor_data["reser"]  # This remains an integer
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
                    nn.Conv1d(in_channels=reservoir_size, out_channels=256, kernel_size=3,
                              padding=1),
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
        model = SNNRegression(reservoir_size, output_dim).to("cpu")
        criterion = nn.MSELoss()
        model.load_state_dict(model_state)

        st.session_state.device = "cpu"
        st.session_state.reser_size = reservoir_size
        st.session_state.win = W_in
        st.session_state.wres = W_res
        st.session_state.scaler = robust_scaler
        # # Data loaders
        # batch_size = 32
        # train_dataset = TensorDataset(X_train, y_train)
        # val_dataset = TensorDataset(X_val, y_val)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # Training loop
        # epochs = 40
        # train_losses = []
        # val_losses = []
        #
        # for epoch in range(epochs):
        #     model.train()
        #     train_loss = 0
        #
        #     for X_batch, y_batch in train_loader:
        #         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        #
        #         # Reservoir computation
        #         reservoir_state = []
        #         for x in X_batch:
        #             x = x.unsqueeze(0)  # Ensure x has a batch dimension
        #             res_state = torch.tanh(
        #                 W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
        #             reservoir_state.append(res_state.squeeze(1))
        #         reservoir_state = torch.stack(reservoir_state).to(device)
        #
        #         output = model(reservoir_state)
        #         loss = criterion(output, y_batch)
        #
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_loss += loss.item()
        #
        #     train_loss /= len(train_loader)
        #     train_losses.append(train_loss)
        #
        #     model.eval()
        #     val_loss = 0
        #     with torch.no_grad():
        #         for X_batch, y_batch in val_loader:
        #             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        #
        #             reservoir_state = []
        #             for x in X_batch:
        #                 x = x.unsqueeze(0)
        #                 res_state = torch.tanh(
        #                     W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
        #                 reservoir_state.append(res_state.squeeze(1))
        #             reservoir_state = torch.stack(reservoir_state).to(device)
        #
        #             output = model(reservoir_state)
        #             loss = criterion(output, y_batch)
        #             val_loss += loss.item()
        #
        #     val_loss /= len(val_loader)
        #     val_losses.append(val_loss)
        #     scheduler.step(val_loss)
        #
        #     print(
        #         f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        #
        # X_test, y_test = X_test.to(device), y_test.to(device)
        # model.eval()
        # with torch.no_grad():
        #     reservoir_state = []
        #     for x in X_test:
        #         x = x.unsqueeze(0)
        #         res_state = torch.tanh(
        #             W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
        #         reservoir_state.append(res_state.squeeze(1))
        #     reservoir_state = torch.stack(reservoir_state).to(device)
        #
        #     predictions = model(reservoir_state)
        #     test_loss = criterion(predictions, y_test)
        #     loses = []
        #     for i in range(len(predictions)):
        #         loses.append(criterion(predictions[i], y_test[i]))
        #     print(f"{min(loses):.4f}")
        #     ind = 0
        #     for i in range(len(loses)):
        #         if min(loses) == loses[i]: ind = i
        #
        # qu = int(round(len(datafra) * 0.1, 0))
        # print(2)


        # .reshape(orig_shape)
        def make_prediction(input_values):

            input_values = st.session_state.scaler.transform(input_values)
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(input_values, dtype=torch.float32).to(
                    st.session_state.device).unsqueeze(0)
                reservoir_state = torch.tanh(
                    st.session_state.win @ input_tensor.T + st.session_state.wres @ torch.rand(
                        st.session_state.reser_size, 1).to(st.session_state.device))
                reservoir_state = reservoir_state.T
                reservoir_state = reservoir_state.unsqueeze(2)
                prediction = model(reservoir_state).cpu().numpy()
            return prediction

        countr = (-len(datafra["y"].tolist()))
        vals = datafra["y"].tolist()[:2]
        for i in range(len(datafra["y"].tolist())):
            try:
                new_sample = datafra["y"].tolist()[countr:countr+2]
                new_sample = np.array(new_sample).reshape(-1, 1)
                print(new_sample)
                result = make_prediction(new_sample)
                print("Prediction for new sample:", result)
                for i in result.tolist()[0]:
                    vals.append(i)
                countr += 40
            except: break
        for i in range(397):
            vals.pop()
        print(vals)
        datafra['NBEATSx'] = vals
        datafra['residuals'] = np.abs(datafra['y'] - datafra['NBEATSx'])

        # Set anomaly threshold (adjust based on domain knowledge)
        threshold = 4 * datafra['residuals'].std()
        datafra['anomaly'] = datafra['residuals'] > threshold

        # # Plot actual, predicted values, and anomalies using plotly

        st.session_state.datanom = datafra.drop(['unique_id', 'residuals'], axis=1)
        if st.session_state.date_not_n == True:
            st.session_state.datanom["ds"] = [i for i in range(1, len(st.session_state.datanom) + 1)]
        print("preds", "-" * 100)
        print(st.session_state.datanom)


# if __name__ == "__main__":

if st.session_state.df is not None:
    # st.session_state.datanom = None
    st.session_state.df_anom = pd.DataFrame()
    st.session_state.df_anom["y"] = st.session_state.df[st.session_state.target]
    try:
        st.session_state.df_anom["ds"] = st.session_state.df[st.session_state.date]
        st.session_state.date_not_n = False
        st.session_state.df_anom['ds'] = pd.to_datetime(st.session_state.df_anom['ds'])
    except:
        st.session_state.date_not_n = True
        st.session_state.df_anom['ds'] = [i for i in range(1, len(ds_for_pred) + 1)]


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
    print(st.session_state.df_anom)
    # st.session_state.df_forpred = ds_for_pred

    with st.container():
        if st.session_state.lang == "ukr":
            st.title("Тестування часового ряду на аномалії")
            # st.markdown("### ")
            st.markdown("#### Тестування часових рядів на виявлення аномалій є важливим етапом аналізу, що дозволяє ідентифікувати нехарактерні або несподівані зміни в даних, які можуть свідчити про значущі події або проблеми у функціонуванні системи. Аномалії можуть бути ознакою технічних несправностей, системних збоїв або навіть випадків шахрайства у фінансових даних. Вчасне виявлення таких відхилень сприяє запобіганню критичним помилкам та мінімізації ризиків.")
            st.markdown("### ")
            # fr = st.selectbox("Оберіть частоту запису даних в ряді:",
            #                   ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"]
            print(means[st.session_state.freq])
            st.button(label="Провести тестування", key="anom", on_click=anomal,
                      args=(st.session_state.df_anom, means[st.session_state.freq]))
        else:
            st.title("Time series anomaly detection testing")
            # st.markdown("### ")
            st.markdown("#### Time series anomaly detection testing is a crucial stage of analysis that allows for the identification of atypical or unexpected changes in data, which may indicate significant events or issues in system operation. Anomalies can signal technical failures, system malfunctions, or even instances of fraud in financial data. Timely detection of such deviations helps prevent critical errors and minimize risks.")

            st.markdown("### ")
            # fr = st.selectbox("Оберіть частоту запису даних в ряді:",
            #                   ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"]
            print(means[st.session_state.freq])
            st.button(label="Conduct testing", key="anom", on_click=anomal,
                            args=(st.session_state.df_anom, means[st.session_state.freq]))


    st.divider()
    if st.session_state.datanom is not None:
        if st.session_state.lang == "ukr":
            st.markdown("# Результат проведення тестування")
            datafra = st.session_state.datanom.rename(columns={"NBEATSx": "preds"})
            datafra = datafra.reset_index()
            print("preds", "-" * 100)
            print(datafra)
            col3, col4 = st.columns(2)
            with col3:
                with st.expander("Подивитись тест даних на аномалії:"):
                    st.write(st.session_state.datanom)
            with col4:
                st.download_button(
                    label="Завантажити тест як файл .csv",
                    data=st.session_state.datanom.to_csv().encode("utf-8"),
                    file_name="anomaly.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Завантажити тест як файл .xlsx",
                    data=to_excel(st.session_state.datanom),
                    file_name="anomaly.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.divider()
            sl = st.select_slider(
                "Оберіть горизонт даних:",
                options=[i for i in range(1, len(datafra))]
            )
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(x=datafra[:sl]['ds'], y=datafra[:sl]['y'], mode='lines', name='Дані', line=dict(color='blue')))

            # Add predicted values
            print("jkghhdfgihdfiopjajkiopdfjlkbjklopkdjklfbjkaopwkwdjkbn jkiopaibhn dhjfiopijiahbe rjiofvh adjiofvh jiobhfv ghiojHVJ DFHJIObhwv hdiou0fohvgHIOJOHBJVFHIOUPshGCDFUIOSVJGHDVUIFPHGVGCAGHUSIDHFGVGGUIASDGVGFUIOPAHVGUIF8HVAGUIDDCFHUIOSDGHVGFUOAHGSHVDFUIOHGV")
            print(datafra[:sl]['preds'])
            fig.add_trace(
                go.Scatter(x=datafra[:sl]['ds'], y=datafra[:sl]['preds'], mode='lines', name='Прогнозовано', line=dict(color='green')))

            # Highlight anomalies
            anomalies = datafra[:sl][datafra['anomaly'] == True]
            fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Аномалія',
                                     marker=dict(color='red', size=8)))

            # Add title and labels
            fig.update_layout(
                title='Графік аномалій',
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
            )

            # Show the plot
            st.session_state.fig_a = fig
            co1, co2 = st.columns([1, 4])
            with co2:
                st.plotly_chart(st.session_state.fig_a, use_container_width=True)
            anomalie_count = 0
            for i in datafra[:sl]['anomaly'].values.tolist():
                if i:
                    anomalie_count += 1
            with co1:
                st.markdown("# ")
                st.metric(label="К-ть аномалій", value=anomalie_count)
        else:
            st.markdown("# Test results")
            datafra = st.session_state.datanom.rename(columns={"NBEATSx": "preds"})
            datafra = datafra.reset_index()
            print("preds", "-" * 100)
            print(datafra)
            col3, col4 = st.columns(2)
            with col3:
                with st.expander("Подивитись тест даних на аномалії:"):
                    st.write(st.session_state.datanom)
            with col4:
                st.download_button(
                    label="Download the test as a .csv",
                    data=st.session_state.datanom.to_csv().encode("utf-8"),
                    file_name="anomaly.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download the test as a .xlsx",
                    data=to_excel(st.session_state.datanom),
                    file_name="anomaly.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.divider()
            sl = st.select_slider(
                "Choose data horizon:",
                options=[i for i in range(1, len(datafra))]
            )
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(x=datafra[:sl]['ds'], y=datafra[:sl]['y'], mode='lines', name='Data',
                           line=dict(color='blue')))

            # Add predicted values
            print(
                "jkghhdfgihdfiopjajkiopdfjlkbjklopkdjklfbjkaopwkwdjkbn jkiopaibhn dhjfiopijiahbe rjiofvh adjiofvh jiobhfv ghiojHVJ DFHJIObhwv hdiou0fohvgHIOJOHBJVFHIOUPshGCDFUIOSVJGHDVUIFPHGVGCAGHUSIDHFGVGGUIASDGVGFUIOPAHVGUIF8HVAGUIDDCFHUIOSDGHVGFUOAHGSHVDFUIOHGV")
            print(datafra[:sl]['preds'])
            fig.add_trace(
                go.Scatter(x=datafra[:sl]['ds'], y=datafra[:sl]['preds'], mode='lines', name='Forecast',
                           line=dict(color='green')))

            # Highlight anomalies
            anomalies = datafra[:sl][datafra['anomaly'] == True]
            fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Anomaly',
                                     marker=dict(color='red', size=8)))

            # Add title and labels
            fig.update_layout(
                title='Графік аномалій',
                xaxis_title='Date',
                yaxis_title='Values',
                template='plotly_white'
            )

            # Show the plot
            st.session_state.fig_a = fig
            co1, co2 = st.columns([1, 4])
            with co2:
                st.plotly_chart(st.session_state.fig_a, use_container_width=True)
            anomalie_count = 0
            for i in datafra[:sl]['anomaly'].values.tolist():
                if i:
                    anomalie_count += 1
            with co1:
                st.markdown("# ")
                st.metric(label="Number of anomalies", value=anomalie_count)
    # st.button("Train", type="primary", on_click=save_performance, args=((model, k)))
    #
    # with st.expander("See full dataset"):
    #     st.write(wine_df)
    #
    # if len(st.session_state['score']) != 0:
    #     st.subheader(f"The model has an F1-Score of: {st.session_state['score'][-1]}")
else:
    if st.session_state.lang == "ukr":
        st.warning('Для проведення тесту на аномалії, оберіть дані', icon="⚠️")
    else:
        st.warning('To perform an anomaly test, select the data.', icon="⚠️")

