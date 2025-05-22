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


# @st.cache_data(show_spinner="Loading data...")
# def load_data():
#     wine_data = load_wine()
#     wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
#
#     wine_df['target'] = wine_data.target
#
#     return wine_df
#
# wine_df = load_data()
#
# @st.cache_data
# def split_data(df):
#
#     X = df.drop(['target'], axis=1)
#     y = df['target']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)
#
#     return X_train, X_test, y_train, y_test
#
# X_train, X_test, y_train, y_test = split_data(wine_df)
#
# @st.cache_data()
# def select_features(X_train, y_train, X_test, k):
#     selector = SelectKBest(mutual_info_classif, k=k)
#     selector.fit(X_train, y_train)
#
#     sel_X_train = selector.transform(X_train)
#     sel_X_test = selector.transform(X_test)
#
#     return sel_X_train, sel_X_test
#
# @st.cache_data(show_spinner="Training and evaluating model")
# def fit_and_score(model, k):
#
#     if model == "Baseline":
#         clf = DummyClassifier(strategy="stratified", random_state=42)
#     elif model == "Decision Tree":
#         clf = DecisionTreeClassifier(random_state=42)
#     elif model == "Random Forest":
#         clf = RandomForestClassifier(random_state=42)
#     else:
#         clf = GradientBoostingClassifier(random_state=42)
#
#     sel_X_train, sel_X_test = select_features(X_train, y_train, X_test, k)
#
#     clf.fit(sel_X_train, y_train)
#
#     preds = clf.predict(sel_X_test)
#
#     score = round(f1_score(y_test, preds, average='weighted'),3)
#
#     return score
#
# def save_performance(model, k):
#
#     score = fit_and_score(model, k)
#
#     st.session_state['model'].append(model)
#     st.session_state['num_features'].append(k)
#     st.session_state['score'].append(score)

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


def submit_data_auto(datafra, iter, horizon, rarety):
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
    st.session_state.df_forpred = datafra
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.01, 0))
        fcst = NeuralForecast(
            models=[
                KAN(h=horizon,
                    input_size=horizon * q,

                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
                TimesNet(h=horizon,
                         input_size=horizon * q,
                         max_steps=iter,
                         scaler_type='standard',
                         start_padding_enabled=True
                         ),
                TimeMixer(h=horizon,
                          input_size=horizon * q,
                          max_steps=iter,
                          scaler_type='standard',
                          n_series=1
                          ),
                PatchTST(h=horizon,
                         input_size=horizon * q,
                         max_steps=iter,
                         scaler_type='standard',
                         start_padding_enabled=True
                         ),
                NBEATSx(h=horizon,
                        input_size=horizon * q,
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
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        print(forecasts)
        results = {}
        for i in ["KAN", "TimesNet", "TimeMixer", "PatchTST", "NBEATSx"]:
            results[i] = mean_squared_error(Y_test_df["y"], forecasts[i])

        key_with_min_value = min(results, key=results.get)

        if key_with_min_value == "KAN":
            fcst = NeuralForecast(
                models=[
                    KAN(h=horizon,
                        input_size=horizon * q,
                        max_steps=iter,
                        scaler_type='standard',
                        start_padding_enabled=True
                        )

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["KAN"]
            st.session_state.inst_name = "KAN"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["KAN"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["KAN"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["KAN"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            st.session_state.fig = go.Figure()

            if st.session_state.lang == "ukr":
                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Дані',
                    line=dict(color='blue')
                ))

                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green')
                ))
            else:
                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Data',
                    line=dict(color='blue')
                ))

                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
            print(dpred)
        if key_with_min_value == "TimesNet":
            fcst = NeuralForecast(
                models=[
                    TimesNet(h=horizon,
                             input_size=horizon * q,
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
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["TimesNet"]
            st.session_state.inst_name = "TimesNet"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["TimesNet"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["TimesNet"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["TimesNet"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            st.session_state.fig = go.Figure()

            if st.session_state.lang == "ukr":

                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Дані',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green')
                ))
            else:
                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Data',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
        ))
            print(dpred)
        if key_with_min_value == "TimeMixer":
            fcst = NeuralForecast(
                models=[
                    TimeMixer(h=horizon,
                              input_size=horizon * q,

                              max_steps=iter,
                              scaler_type='standard',

                              n_series=1
                              )

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["TimeMixer"]
            st.session_state.inst_name = "TimeMixer"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["TimeMixer"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["TimeMixer"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["TimeMixer"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

            st.session_state.fig = go.Figure()

            if st.session_state.lang == "ukr":

                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Дані',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green')
                ))
            else:
                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Data',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
            print(dpred)
        if key_with_min_value == "PatchTST":
            fcst = NeuralForecast(
                models=[
                    PatchTST(h=horizon,
                             input_size=horizon * q,
                             # output_size=horizon,
                             max_steps=iter,
                             scaler_type='standard',
                             start_padding_enabled=True
                             )

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["PatchTST"]
            st.session_state.inst_name = "PatchTST"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["PatchTST"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["PatchTST"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["PatchTST"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

            st.session_state.fig = go.Figure()

            if st.session_state.lang == "ukr":

                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Дані',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green')
                ))
            else:
                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Data',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
            print(dpred)

        if key_with_min_value == "NBEATSx":
            fcst = NeuralForecast(
                models=[
                    NBEATSx(h=horizon,
                            input_size=horizon * q,
                            max_steps=iter,
                            scaler_type='standard',
                            start_padding_enabled=True
                            ),

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.inst_name = "NBEATSx"
            st.session_state.mse = results["NBEATSx"]
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["NBEATSx"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["NBEATSx"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["NBEATSx"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

            st.session_state.fig = go.Figure()
            if st.session_state.lang == "ukr":

                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Дані',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green')
                ))
            else:
                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["real"],
                    mode='lines',
                    name='Data',
                    line=dict(color='blue')
                ))


                st.session_state.fig.add_trace(go.Scatter(
                    x=dpred["unique_id"],
                    y=dpred["pred"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
            print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")

    # st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["KAN"])
    # st.session_state.model_forecast = fcst
    # print(f'Mean Squared Error: {st.session_state.mse}')
    # print(len(forecasts["KAN"]), len(Y_test_df["y"]))
    # print(forecasts.columns)
    # forecasts = forecasts.reset_index(drop=True)
    # print(forecasts.columns.values)
    # print(forecasts["KAN"].values.tolist())
    # dpred = pd.DataFrame()
    # dpred["real"] = Y_test_df["y"]
    # dpred["pred"] = forecasts["KAN"].values.tolist()
    # dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
    # # Create distplot with custom bin_size
    # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
    # print(dpred)


def submit_data_KAN(datafra, iter, horizon, rarety, inp):
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
    st.session_state.df_forpred = datafra
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.05, 0))
        fcst = NeuralForecast(
            models=[
                KAN(h=horizon,
                    input_size=int(round(inp, 0)),

                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["KAN"])

        st.session_state.inst_name = "KAN"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["KAN"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["KAN"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["KAN"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

        st.session_state.fig = go.Figure()

        if st.session_state.lang == "ukr":

            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Дані',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Прогноз',
                line=dict(color='green')
            ))
        else:
            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Data',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Forecast',
                line=dict(color='green')
            ))
        # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
    except Exception as ex:
        print(ex)
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


def submit_data_SNN(datafra, iter, horizon, rarety, inp, bs):
    # if st.session_state.date_not_n:
    print("no date")
    print(datafra)
    dafaf = datafra

    datafra['ds'] = [i for i in range(1, len(datafra) + 1)]
    start_date = pd.to_datetime('2024-01-01')
    rarety = "D"
    datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    st.session_state.inp = inp
    datafra['ds'] = pd.to_datetime(datafra['ds'])

    datafra = datafra.drop_duplicates(subset=['ds'])
    # datafra['y'] = datafra['y'].interpolate()
    mean = datafra.mean()
    #
    # datafra["y"].replace({"NaN": 10})


    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    datafra.fillna(mean, inplace=True)

    st.session_state.df_forpred = datafra
    dafaf['ds'] = dafaf['ds'].astype(str)
    # datafra = datafra.set_index('ds').asfreq(rarety)
    # datafra = datafra.reset_index()
    #
    # datafra.fillna(mean, inplace=True)

    print(datafra)

    print(datafra["ds"].tolist()[inp])
    print(datafra["ds"].tolist()[-(horizon + 1)])

    windowed_df = df_to_windowed_df(datafra,
                                    datafra["ds"].tolist()[inp],
                                    datafra["ds"].tolist()[-(horizon + 1)],
                                    n=inp,
                                    hor=horizon)

    dates, X, Y = windowed_df_to_date_X_y(windowed_df, hor=horizon)

    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]

    print("__________________________________" * 1000)
    st.session_state.df_forpred = st.session_state.df_forpred.reset_index()
    print(st.session_state.df_forpred)

    # try:

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
    #

    
    if bs:
        import requests
        import json

        url = "https://nxwekahcj0m93m-8000.proxy.runpod.net/forecast"


        payload = {
            "data": dafaf.to_dict(orient='records'),
            "inp": inp,  
            "horiz": horizon,  
            "iter": iter,  
        }


        response = requests.post(url, json=payload)

        data = ""
        data = response.json().get("predictions")
        # print("Forecast predictions:", data)

        # data = json.loads(json_data)



        model_state_serialized = data["model_state"]
        model_state = {k: torch.tensor(v) for k, v in model_state_serialized.items()}
        # Load the state dictionary into the model.


        scaler_data = data["robust_scaler"]
        robust_scaler = RobustScaler(**scaler_data["params"])
        attributes = scaler_data.get("attributes", {})
        if "center_" in attributes:
            robust_scaler.center_ = np.array(attributes["center_"])
        if "scale_" in attributes:
            robust_scaler.scale_ = np.array(attributes["scale_"])


        tensor_data = data["int_values"]

        W_in = torch.tensor(tensor_data["W_in"])
        W_res = torch.tensor(tensor_data["W_res"])
        reservoir_size = tensor_data["reser"]  
        beta = 0.5
        time_steps = 150
        spike_grad = surrogate.fast_sigmoid()


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
        model = SNNRegression(reservoir_size, output_dim).to("cpu")
        criterion = nn.MSELoss()
        model.load_state_dict(model_state)

        st.session_state.device = "cpu"
        st.session_state.reser_size = reservoir_size
        st.session_state.win = W_in
        st.session_state.wres = W_res
        st.session_state.scaler = robust_scaler
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
        #
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
        #                                                        verbose=True)
        #
        # # Data loaders
        # batch_size = 32
        # train_dataset = TensorDataset(X_train, y_train)
        # val_dataset = TensorDataset(X_val, y_val)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # Training loop
        # epochs = iter
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
        #             res_state = torch.tanh(W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
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

        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for X_batch, y_batch in val_loader:
        #         X_batch, y_batch = X_batch.to("cpu"), y_batch.to("cpu")
        #
        #         reservoir_state = []
        #         for x in X_batch:
        #             x = x.unsqueeze(0)
        #             res_state = torch.tanh(W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
        #             reservoir_state.append(res_state.squeeze(1))
        #         reservoir_state = torch.stack(reservoir_state).to(device)
        #
        #         output = model(reservoir_state)
        #         loss = criterion(output, y_batch)
        #         val_loss += loss.item()
        #
        # val_loss /= len(val_loader)
        # val_losses.append(val_loss)
        # scheduler.step(val_loss)
        #
        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        orig_shape = X_train.shape
        print(X_train.reshape(-1, X_train.shape[-1]))
        X_test = robust_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        st.session_state.scaler = robust_scaler
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        X_test, y_test = X_test.to("cpu"), y_test.to("cpu")
        model.eval()
        with torch.no_grad():
            reservoir_state = []
            for x in X_test:
                x = x.unsqueeze(0)
                res_state = torch.tanh(W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to("cpu"))
                reservoir_state.append(res_state.squeeze(1))
            reservoir_state = torch.stack(reservoir_state).to("cpu")

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

        st.session_state.horiz = horizon

        st.session_state.inst_name = "SNN"
        st.session_state.model_forecast = model
        st.session_state.mse = float(f"{min(loses):.4f}")
        sample_idx = ind
        single_prediction = predictions[sample_idx].cpu().numpy()
        single_y_test = y_test[sample_idx].cpu().numpy()
        times = [i for i in range(1, len(single_prediction) - 1)]

        st.session_state.fig = go.Figure()

        st.session_state.fig.add_trace(go.Scatter(
            x=times,
            y=single_y_test,
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))


        st.session_state.fig.add_trace(go.Scatter(
            x=times,
            y=single_prediction,
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
    else:
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

        st.session_state.device = device

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
        st.session_state.reser_size = reservoir_size
        st.session_state.win = W_in
        st.session_state.wres = W_res

        beta = 0.5
        time_steps = 150
        spike_grad = surrogate.fast_sigmoid()


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


        batch_size = 32
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        epochs = iter
        train_losses = []
        val_losses = []

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

        st.session_state.horiz = horizon

        st.session_state.inst_name = "SNN"
        st.session_state.model_forecast = model
        st.session_state.mse = float(f"{min(loses):.4f}")
        sample_idx = ind
        single_prediction = predictions[sample_idx].cpu().numpy()
        single_y_test = y_test[sample_idx].cpu().numpy()
        times = [i for i in range(1, len(single_prediction) - 1)]
        st.session_state.fig = go.Figure()


        st.session_state.fig.add_trace(go.Scatter(
            x=times,
            y=single_y_test,
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))

        st.session_state.fig.add_trace(go.Scatter(
            x=times,
            y=single_prediction,
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
        # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        # except Exception as ex:
        #     print(ex)
        #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")


def submit_data_TN(datafra, iter, horizon, rarety, inp):
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
    st.session_state.df_forpred = datafra
    try:
        st.session_state.horiz = horizon
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
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["TimesNet"])
        st.session_state.inst_name = "TimesNet"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["TimesNet"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["TimesNet"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["TimesNet"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

        st.session_state.fig = go.Figure()


        if st.session_state.lang == "ukr":

            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Дані',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Прогноз',
                line=dict(color='green')
            ))
        else:
            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Data',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Forecast',
                line=dict(color='green')
            ))
        print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


def submit_data_TM(datafra, iter, horizon, rarety, inp):
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
    st.session_state.df_forpred = datafra
    try:
        st.session_state.horiz = horizon
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
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["TimeMixer"])
        st.session_state.inst_name = "TimeMixer"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["TimeMixer"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["TimeMixer"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["TimeMixer"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

        st.session_state.fig = go.Figure()

        if st.session_state.lang == "ukr":

            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Дані',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Прогноз',
                line=dict(color='green')
            ))
        else:
            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Data',
                line=dict(color='blue')
            ))

            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Forecast',
                line=dict(color='green')
            ))
        print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


def submit_data_PTST(datafra, iter, horizon, rarety, inp):
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
    st.session_state.df_forpred = datafra
    try:
        st.session_state.horiz = horizon
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
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["PatchTST"])
        st.session_state.inst_name = "PatchTST"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["PatchTST"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["PatchTST"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["PatchTST"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        st.session_state.fig = go.Figure()

        if st.session_state.lang == "ukr":

            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Дані',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Прогноз',
                line=dict(color='green')
            ))
        else:
            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["real"],
                mode='lines',
                name='Data',
                line=dict(color='blue')
            ))


            st.session_state.fig.add_trace(go.Scatter(
                x=dpred["unique_id"],
                y=dpred["pred"],
                mode='lines',
                name='Forecast',
                line=dict(color='green')
            ))
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


def submit_data_NBx(datafra, iter, horizon, rarety, inp):
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
    st.session_state.df_forpred = datafra

    st.session_state.horiz = horizon
    q = int(round(len(datafra) * 0.1, 0))
    fcst = NeuralForecast(
        models=[
            NBEATSx(h=horizon,
                    input_size=inp,
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
    fcst.fit(df=Y_train_df)
    forecasts = fcst.predict(futr_df=Y_test_df)
    st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["NBEATSx"])
    st.session_state.inst_name = "NBEATSx"
    st.session_state.model_forecast = fcst
    print(f'Mean Squared Error: {st.session_state.mse}')
    print(len(forecasts["NBEATSx"]), len(Y_test_df["y"]))
    print(forecasts.columns)
    forecasts = forecasts.reset_index(drop=True)
    print(forecasts.columns.values)
    print(forecasts["NBEATSx"].values.tolist())
    dpred = pd.DataFrame()
    dpred["real"] = Y_test_df["y"]
    dpred["pred"] = forecasts["NBEATSx"].values.tolist()
    dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]

    st.session_state.fig = go.Figure()
    if st.session_state.lang == "ukr":

        st.session_state.fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["real"],
            mode='lines',
            name='Дані',
            line=dict(color='blue')
        ))


        st.session_state.fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["pred"],
            mode='lines',
            name='Прогноз',
            line=dict(color='green')
        ))
    else:
        st.session_state.fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["real"],
            mode='lines',
            name='Data',
            line=dict(color='blue')
        ))


        st.session_state.fig.add_trace(go.Scatter(
            x=dpred["unique_id"],
            y=dpred["pred"],
            mode='lines',
            name='Forecast',
            line=dict(color='green')
        ))
    print(dpred)
    # except:
    #     st.warning('Надано не коректні гіперпараметри', icon="⚠️")


# if __name__ == "__main__":

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
        if st.session_state.lang == "ukr":
            st.title("Обрання та налашутвання моделі прогнозування")
        else: st.title("Selection and configuration of the forecasting model")
    if st.session_state.lang == "ukr":
         model = option_menu("Оберіть модель для передбачення",
                           ["KAN", "TimesNet", "NBEATSx", "TimeMixer", "PatchTST", "SNN", "Auto-choose"],
                           # icons=['gear', 'gear', 'gear', 'gear', 'gear', 'gear'],
                           menu_icon="no",
                           orientation="horizontal")
    else:
         model = option_menu("Choose model for forecasting",
                           ["KAN", "TimesNet", "NBEATSx", "TimeMixer", "PatchTST", "SNN", "Auto-choose"],
                           # icons=['gear', 'gear', 'gear', 'gear', 'gear', 'gear'],
                           menu_icon="no",
                           orientation="horizontal")

    try:
        if model == "KAN":
            if st.session_state.lang == "ukr":
                st.markdown("## Ви обрали модель KAN")
                st.markdown(
                    "### KAN — це нейронна мережа, що застосовує апроксимаційну теорему Колмогорова-Арнольда, яка стверджує, що сплайни можуть апроксимувати складніші функції.")
                st.divider()
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

                st.button(label="Підтвердити", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
            else:
                st.markdown("## You have selected the KAN model")
                st.markdown(
                    "### KAN is a neural network that applies the Kolmogorov-Arnold approximation theorem, which states that splines can approximate more complex functions.")
                st.divider()
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

                st.button(label="Submit", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    except:
        print(ex)
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")

    try:
        if model == "NBEATSx":
            if st.session_state.lang == "ukr":
                st.markdown("## Ви обрали модель NBEATSx")
                st.markdown(
                    "### NBEATSx — це глибока нейронна архітектура на основі MLP, яка використовує прямі та зворотні залишкові зв'язки.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)]
                )

                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:", step=1, min_value=5,
                                    max_value=2050)
                st.button(label="Підтвердити", key="kan", on_click=submit_data_NBx,
                        args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
            else:
                st.markdown("## You have selected the NBEATSx model")
                st.markdown(
                    "### NBEATSx is a deep neural architecture based on MLP that uses direct and reverse residual connections.")
                st.divider()
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

                st.button(label="Submit", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
            
    except:
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")
    try:
        if model == "TimesNet":
            if st.session_state.lang == "ukr":
                st.markdown("## Ви обрали модель TimesNet")
                st.markdown(
                    "### TimesNet — це модель на основі CNN, яка ефективно вирішує завдання моделювання як внутрішньоперіодних, так і міжперіодних змін у часових рядах.")
                st.divider()
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
                st.button(label="Підтвердити", key="kan", on_click=submit_data_TN,
                        args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
            else:
                st.markdown("## You have selected the TimesNet model")
                st.markdown(
                    "### TimesNet is a CNN-based model that effectively addresses the task of modeling both intra-period and inter-period changes in time series.")
                st.divider()
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

                st.button(label="Submit", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    except:
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")
    try:
        if model == "TimeMixer":
            if st.session_state.lang == "ukr":
                st.markdown("## Ви обрали модель TimeMixer")
                st.markdown(
                    "### TimeMixer - модель, яка поєднує елементи архітектури Transformers і CNN для досягнення високої точності в прогнозах, обробляючи залежності як в просторі, так і в часі.")
                st.divider()
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
                st.button(label="Підтвердити", key="kan", on_click=submit_data_TM,
                        args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
            else:
                st.markdown("## You have selected the TimeMixer model")
                st.markdown(
                    "### TimeMixer is a model that combines elements of Transformer and CNN architectures to achieve high accuracy in predictions by processing dependencies in both space and time.")
                st.divider()
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

                st.button(label="Submit", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    except:
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")
    try:
        if model == "PatchTST":
            if st.session_state.lang == "ukr":
                st.markdown("## Ви обрали модель PatchTST")
                st.markdown(
                    "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
                st.divider()
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
                st.button(label="Підтвердити", key="kan", on_click=submit_data_PTST,
                        args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
            else:
                st.markdown("## You have selected the PatchTST model")
                st.markdown(
                    "### PatchTST is a highly efficient Transformer-based model designed for multivariate time series forecasting.")
                st.divider()
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

                st.button(label="Submit", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    except:
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")
    try:
        if model == "SNN":
            import requests

            x = requests.get('https://nxwekahcj0m93m-8000.proxy.runpod.net/')
            print(x.status_code)
            if x.status_code == 200:
                if st.session_state.lang == "ukr":
                    st.markdown("## Ви обрали модель SNN")
                    st.markdown(
                        "### SNN — це розроблена ШНМ, призначена для прогнозування часових рядів з використанням спайкових нейронних мереж та резервуарних обчислень.")
                    st.divider()
                    st.success("Зауважте, зараз інстанс моделі SNN розгорнуто на сервері з більш потужним NVIDIA A40 GPU. Пришвидшене навчання доступне.")
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
                    
                    boosted_vers = st.checkbox("Використати пришвидшене навчання")

                    st.button(label="Підтвердити", key="kan", on_click=submit_data_SNN,
                            args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp, boosted_vers))
                else:
                    st.markdown("## You have selected the SNN model")
                    st.markdown(
                        "### SNN is a developed Spiking Neural Network (SNN) designed for time series forecasting using spiking neural networks and reservoir computing.")
                    st.divider()
                    st.success("Please note, the SNN model instance is now deployed on a server with a more powerful NVIDIA A40 GPU. Boosted training is now available.")
                    
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

                    boosted_vers = st.checkbox("Use boosted training")

                    st.button(label="Submit", key="kan", on_click=submit_data_SNN,
                            args=(ds_for_pred, iter, horizon, "D", inp, boosted_vers))
            else: 
                if st.session_state.lang == "ukr":
                    st.markdown("## Ви обрали модель SNN")
                    st.markdown(
                        "### SNN — це розроблена ШНМ, призначена для прогнозування часових рядів з використанням спайкових нейронних мереж та резервуарних обчислень.")
                    st.divider()
                    st.warning("Зауважте, зараз інстанс моделі SNN не розгорнуто на сервері з більш потужним gpu. Нажаль, пришвидшене навчання не доступне.")
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
                    

                    st.button(label="Підтвердити", key="kan", on_click=submit_data_SNN,
                            args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp, False))
                else:
                    st.markdown("## You have selected the SNN model")
                    st.markdown(
                        "### SNN is a developed Spiking Neural Network (SNN) designed for time series forecasting using spiking neural networks and reservoir computing.")
                    st.divider()
                    st.warning("Please note, the SNN model instance is not deployed on a server with a more powerful GPU. Unfortunately, boosted training is not available at the moment.")
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

                    st.button(label="Submit", key="kan", on_click=submit_data_SNN,
                            args=(ds_for_pred, iter, horizon, "D", inp, False))
    except:
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")
    try:
        if model == "Auto-choose":
            if st.session_state.lang == "ukr":
                st.markdown("## Ви обрали Авто-вибір")
                st.markdown(
                    "### Тут обирається модель, яка найкраще може працювати з Вашими даними та налаштовуються гіперпараметри для моделі.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення (на скільки вперед буде проводитись передбачення):",
                    options=[i for i in range(1, 151)]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше та точніше):",
                    options=[i for i in range(5, 101)]
                )
                st.button(label="Підтвердити", key="kan", on_click=submit_data_auto,
                        args=(ds_for_pred, iter, horizon, means[st.session_state.freq]))
            else:
                st.markdown("## You have selected the Auto-choose")
                st.markdown(
                    "### Here, a model is selected that can work best with your data, and the hyperparameters for the model are adjusted.")
                st.divider()
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

                st.button(label="Submit", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[st.session_state.freq], inp))
    except:
        if st.session_state.lang == "ukr":
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        else:
            st.warning('Incorrect hyperparameters have been provided', icon="⚠️")
    if st.session_state.lang == "ukr":
        st.warning(
            'Будь-які прогнози базуються на доступних даних та ймовірнісних моделях, тому вони не можуть гарантувати абсолютну точність. Реальні результати можуть відрізнятися через непередбачувані фактори та змінні, які неможливо повністю врахувати.',
            icon="⚠️")
    else:
        st.warning(
            'Any forecasts are based on available data and probabilistic models, so they cannot guarantee absolute accuracy. Actual results may vary due to unpredictable factors and variables that cannot be fully accounted for.',
            icon="⚠️")
    st.divider()
    if st.session_state.fig is not None:
        if st.session_state.inst_name != "Авто-вибір":
            if st.session_state.lang == "ukr":
                st.markdown(
                    f"## Середньоквадратичне відхилення обраної моделі ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")

                st.session_state.fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Значення'
                )
                st.plotly_chart(st.session_state.fig, use_container_width=True)
            else:
                st.markdown(
                    f"## The MSE of the selected model ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")

                st.session_state.fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Values'
                )
                st.plotly_chart(st.session_state.fig, use_container_width=True)
        else:
            if st.session_state.lang == "ukr":
                st.markdown(
                    f"## Середньоквадратичне відхилення обраної моделі авто-вибором ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")
                st.plotly_chart(st.session_state.fig, use_container_width=True)
            else:
                st.markdown(
                    f"## The MSE of the selected model by auto-selection ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")
                st.plotly_chart(st.session_state.fig, use_container_width=True)
else:
    if st.session_state.lang == "ukr":
        st.warning('Для початки роботи з моделями, оберіть дані', icon="⚠️")
    else:
        st.warning('To start working with the models, select the data.', icon="⚠️")
