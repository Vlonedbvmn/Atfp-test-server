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
import os
import random
import time
from groq import Groq
import datetime
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


means = {"Місяць": "M",
         "Година": "h",
         "Рік": "Y",
         "Хвилина": "T",
         "Секунда": "S",
         "День": "D",
         }


client = Groq(api_key="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")



if st.session_state.lang == "ukr":
    st.session_state.messages1 = [{"role": "user", "content": "Здійсни прогнозування на наступні 7 днів"},
                                  {"role": "user", "content": "Здійсни тестування датасету на аномалії"},
                                  {"role": "user", "content": "Здійсни прогнозування на 2 тижні за допомогою моделі TimeMixer"}]
else:
    st.session_state.messages1 = [{"role": "user", "content": "Make forecast for next 7 days"},
                                  {"role": "user", "content": "Do the anomaly analyzing"},
                                  {"role": "user", "content": "Make forecast on next 2 weeks using TimeMixer model"}]


if "no_d" not in st.session_state:
    st.session_state.no_d = None


if "messages" not in st.session_state:
    st.session_state.messages = []

if "fig_b" not in st.session_state:
    st.session_state.fig_b = None
if "dataai" not in st.session_state:
    st.session_state.dataai = None
if "m1" not in st.session_state:
    st.session_state.m1 = None
if "m2" not in st.session_state:
    st.session_state.m2 = None


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


def response_1(chr):
    response = chr
    for word in response.split():
                yield word + " "
                time.sleep(0.1)


def response_generator(datafra, res):
    if st.session_state.lang == "ukr":
        my_bar = st.progress(0, text="Статус відповіді")
        my_bar.progress(33, "Запит отримано")
    else:
        my_bar = st.progress(0, text="Answer status")
        my_bar.progress(33, "Got your prompt")
    frd = st.session_state.freq
    st.session_state.fig_b = None
    st.session_state.dataai = None
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        st.session_state.freq = "День"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, means[st.session_state.freq])

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    datafrsnn = datafra.copy()
    datafra = datafra.set_index('ds').asfreq(means[st.session_state.freq])
    datafra = datafra.reset_index()
    print("s;kgfoshdisdifsdf")
    print(datafra)

    # cs = requests.post(
    #     url="https://openrouter.ai/api/v1/chat/completions",
    #     headers={
    #         "Authorization": f"Bearer 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",
    #     },
    #     data=json.dumps({
    #         "model": "nousresearch/hermes-3-llama-3.1-405b:free",  # Optional
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": f"З цього тексту '{res}' знайди потрібні ключові слова та видай мені відповідь за таким шаблоном:'model:  horizon:  input_size:  task:  '."
    #                            f"model може бути із тільки цього списку: [NBEATSx, KAN, PatchTST, TimeMixer, TimesNet, Авто-вибір, None]; task: [Anomaly, Forecasting, None]; а horizon пиши лише значення без пояснення частоти.  В разі якщо нема якогось компоненту на його місці пиши 'None'. Якщо нема жодного пиши тільки 'уточніть запит'. При наданні відповіді стого слідуй інструкції, тобто відповідай строго за шаблоном. Не пиши зайвого тексту!!! Не пиши коми!!! Горизонт повинен бути лише числом!!!",
    #             }
    #         ]
    #
    #     })
    # )

    # clientt = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key="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",
    # )
    #
    # completion = clientt.chat.completions.create(
    # model = "nousresearch/hermes-3-llama-3.1-405b:free",
    # messages = [
    #     {
    #         "role": "user",
    #         "content": f"З цього тексту '{res}' знайди потрібні ключові слова та видай мені відповідь за таким шаблоном:'model:  horizon:  input_size:  task:  '."
    #                    f"model може бути із тільки цього списку: [NBEATSx, KAN, PatchTST, TimeMixer, TimesNet, Авто-вибір, None]; task: [Anomaly, Forecasting, None]; а horizon пиши лише значення без пояснення частоти.  В разі якщо нема якогось компоненту на його місці пиши 'None'. Якщо нема жодного пиши тільки 'уточніть запит'. При наданні відповіді стого слідуй інструкції, тобто відповідай строго за шаблоном. Не пиши зайвого тексту!!! Не пиши коми!!! Горизонт повинен бути лише числом!!!",
    #     }
    # ]
    # )
    # print(completion.choices[0].message.content)

    # print("cs")
    # print(cs.text)

    chatco = client.chat.completions.create(
        messages=[
            {
               "role": "user",
                "content": f"З цього тексту '{res}' знайди потрібні ключові слова та видай мені відповідь за таким шаблоном:'model:  horizon:  input_size:  task:  '."
                           f"model може бути із тільки цього списку: [NBEATSx, KAN, PatchTST, TimeMixer, SNN, TimesNet, Авто-вибір, None]; task: [Anomaly, Forecasting, None]; а horizon пиши лише значення без пояснення частоти.  В разі якщо нема якогось компоненту на його місці пиши 'None'. Якщо нема жодного пиши тільки 'уточніть запит'. При наданні відповіді стого слідуй інструкції, тобто відповідай строго за шаблоном. Не пиши зайвого тексту!!! Не пиши коми!!! Горизонт повинен бути лише числом!!!",
            }
        ],
        model="llama3-70b-8192"
    )
    
    respo = chatco.choices[0].message.content
    if st.session_state.lang == "ukr":
        response = "Уточніть, будь ласка, Ваш запит"
    else:
        response = "Уточніть, будь ласка, Ваш запит"
    try:
        print(respo)
        mdl = respo.split()[1]
        hrz = respo.split()[3]
        inp_sz = respo.split()[5]
        tsk = respo.split()[7]
        if hrz != "None" and st.session_state.date_not_n is False:

            chatcoc = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"З цього тексту '{res}' знайди потрібні ключові слова та видай мені відповідь за таким шаблоном:'horizon_forecast: '."
                                   f"Також проаналізуй чи горизонт у тих одиницях, що і частота запису даних в ряді({frd}). Якщо так, то впиши у шаблон лише число, а якщо ні то переведи в одиниці що задані та запиши в шаблон лише число. Отже, якщо прогноз на 3 тижні, а частота запису в ряді - день, то ти відповідаєш: 'horizon_forecast: 18'. Видавай відповідь тільки по шаблону та не пиши додаткових розділових знаків.",
                    }
                ],
                model="llama3-70b-8192"
            )   # print([i for i in means if means[i]==st.session_state.freq])
            hrz = chatcoc.choices[0].message.content.split()[1]
            print(mdl)
            print(hrz)
            print(inp_sz)
            print(tsk)

        q = int(round(len(datafra) * 0.01, 0))

        if tsk == "Forecasting":
            if st.session_state.lang == "ukr":
                my_bar.progress(50, "Модель навчається")
                st.session_state.m1 = "Ось табличка з данми Вашого прогнозу"
                st.session_state.m2 = "Та також графік Вашого прогнозу. Синім зображені дані до, а червоним зображено сам прогноз"
                response = "Дякую за Ваш запит, ось результати прогнозування"
            else:
                my_bar.progress(50, "Model is training")
                st.session_state.m1 = "Here is the table with your forecast results"
                st.session_state.m2 = "And also a plot of your forecast. Blue shows the data before, and red shows the forecast itself"
                response = "Thank you for your prompt, here are the forecast results"
            if mdl == "KAN":
                if hrz == "None":
                    if st.session_state.lang == "ukr":
                        response = "Уточніть на скільки вперед робити прогноз"
                    else:
                        response = "Specify how far in advance to make the forecast"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                KAN(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "KAN": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"KAN": st.session_state.target}).drop(["unique_id"], axis=1)
                    else:
                        fcst = NeuralForecast(
                            models=[
                                KAN(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "KAN": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"KAN": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "NBEATSx":
                if hrz == "None":
                    if st.session_state.lang == "ukr":
                        response = "Уточніть на скільки вперед робити прогноз"
                    else:
                        response = "Specify how far in advance to make the forecast"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                    else:
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "TimesNet":
                if hrz == "None":
                    if st.session_state.lang == "ukr":
                        response = "Уточніть на скільки вперед робити прогноз"
                    else:
                        response = "Specify how far in advance to make the forecast"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                TimesNet(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimesNet": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)

                    else:
                        fcst = NeuralForecast(
                            models=[
                                TimesNet(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimesNet": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)

                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "TimeMixer":
                if hrz == "None":
                    if st.session_state.lang == "ukr":
                        response = "Уточніть на скільки вперед робити прогноз"
                    else:
                        response = "Specify how far in advance to make the forecast"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                TimeMixer(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    # start_padding_enabled=True,
                                    n_series=1
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimeMixer": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)

                    else:
                        fcst = NeuralForecast(
                            models=[
                                TimeMixer(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    # start_padding_enabled=True,
                                    n_series=1
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimeMixer": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "SNN":
                if hrz == "None":
                    if st.session_state.lang == "ukr":
                        response = "Уточніть на скільки вперед робити прогноз"
                    else:
                        response = "Specify how far in advance to make the forecast"
                else:
                    if inp_sz == "None":
                        print("jhgut")
                        print(datafra)
                        print(datafra["ds"].tolist()[7])


                        windowed_df = df_to_windowed_df(datafra,
                                                        datafra["ds"].tolist()[7],
                                                        datafra["ds"].tolist()[-(int(hrz) + 1)],
                                                        n=7,
                                                        hor=int(hrz))
                        print("jhgut")
                        dates, X, Y = windowed_df_to_date_X_y(windowed_df, hor=int(hrz))

                        q_80 = int(len(dates) * .8)
                        q_90 = int(len(dates) * .9)

                        dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]

                        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
                        dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]



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
                        model = SNNRegression(reservoir_size, output_dim).to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                               patience=5,
                                                                               verbose=True)


                        batch_size = 32
                        train_dataset = TensorDataset(X_train, y_train)
                        val_dataset = TensorDataset(X_val, y_val)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


                        epochs = 75
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
                                    res_state = torch.tanh(
                                        W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
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
                                        res_state = torch.tanh(
                                            W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
                                        reservoir_state.append(res_state.squeeze(1))
                                    reservoir_state = torch.stack(reservoir_state).to(device)

                                    output = model(reservoir_state)
                                    loss = criterion(output, y_batch)
                                    val_loss += loss.item()

                            val_loss /= len(val_loader)
                            val_losses.append(val_loss)
                            scheduler.step(val_loss)

                            print(
                                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                        X_test, y_test = X_test.to(device), y_test.to(device)
                        model.eval()
                        with torch.no_grad():
                            reservoir_state = []
                            for x in X_test:
                                x = x.unsqueeze(0)
                                res_state = torch.tanh(
                                    W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
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


                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        h = datafrsnn[-qu:]["y"].tolist()
                        new_sample = datafrsnn["y"].tolist()[-7:]
                        new_sample = np.array(new_sample).reshape(-1, 1)


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

                        print(new_sample)
                        result = make_prediction(new_sample)
                        print("Prediction for new sample:", result)
                        pr = []
                        for i in result.tolist()[0]:
                            pr.append(i)
                            h.append(i)

                        pr1dates = []
                        pr2dates = []
                        for i in range(1, len(h) + 1):
                            pr1dates.append(i)
                        for i in range(1, len(pr) + 1):
                            pr2dates.append(i)

                        predicted = pd.DataFrame({
                            st.session_state.date: pr1dates,
                            st.session_state.target: h
                        })
                        predicted2 = pd.DataFrame({
                            st.session_state.date: pr2dates,
                            st.session_state.target: pr
                        })
                        # if st.session_state.date_not_n == True:
                        #     pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predicted.tail(int(hrz))
                        rest_of_data = predicted.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = predicted2

                    else:
                        windowed_df = df_to_windowed_df(datafra,
                                                        datafra["ds"].tolist()[int(inp_sz)],
                                                        datafra["ds"].tolist()[-(int(hrz) + 1)],
                                                        n=int(inp_sz),
                                                        hor=int(hrz))

                        dates, X, Y = windowed_df_to_date_X_y(windowed_df, hor=int(hrz))

                        q_80 = int(len(dates) * .8)
                        q_90 = int(len(dates) * .9)

                        dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]

                        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
                        dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]

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
                        model = SNNRegression(reservoir_size, output_dim).to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                               patience=5,
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


                                reservoir_state = []
                                for x in X_batch:
                                    x = x.unsqueeze(0)  
                                    res_state = torch.tanh(
                                        W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
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
                                        res_state = torch.tanh(
                                            W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
                                        reservoir_state.append(res_state.squeeze(1))
                                    reservoir_state = torch.stack(reservoir_state).to(device)

                                    output = model(reservoir_state)
                                    loss = criterion(output, y_batch)
                                    val_loss += loss.item()

                            val_loss /= len(val_loader)
                            val_losses.append(val_loss)
                            scheduler.step(val_loss)

                            print(
                                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                        X_test, y_test = X_test.to(device), y_test.to(device)
                        model.eval()
                        with torch.no_grad():
                            reservoir_state = []
                            for x in X_test:
                                x = x.unsqueeze(0)
                                res_state = torch.tanh(
                                    W_in @ x.T + W_res @ torch.rand(reservoir_size, 1).to(device))
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

                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        h = datafrsnn[-qu:]["y"].tolist()
                        new_sample = datafrsnn["y"].tolist()[-int(inp_sz):]
                        new_sample = np.array(new_sample).reshape(-1, 1)

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

                        print(new_sample)
                        result = make_prediction(new_sample)
                        print("Prediction for new sample:", result)
                        pr = []
                        for i in result.tolist()[0]:
                            pr.append(i)
                            h.append(i)

                        pr1dates = []
                        pr2dates = []
                        for i in range(1, len(h) + 1):
                            pr1dates.append(i)
                        for i in range(1, len(pr) + 1):
                            pr2dates.append(i)

                        predicted = pd.DataFrame({
                            st.session_state.date: pr1dates,
                            st.session_state.target: h
                        })
                        predicted2 = pd.DataFrame({
                            st.session_state.date: pr2dates,
                            st.session_state.target: pr
                        })
                        # if st.session_state.date_not_n == True:
                        #     pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predicted.tail(int(hrz))
                        rest_of_data = predicted.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))


                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = predicted2
                        # Show the plot
            if mdl == "PatchTST":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                PatchTST(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "PatchTST": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)


                        chr = go.Figure()


                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))


                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"PatchTST": st.session_state.target}).drop(["unique_id"], axis=1)
                    else:
                        fcst = NeuralForecast(
                            models=[
                                PatchTST(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "PatchTST": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"PatchTST": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "Авто-вибір":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    fcst = NeuralForecast(
                        models=[
                            KAN(h=int(hrz),
                                input_size=int(hrz) * q,
                                # output_size=horizon,
                                max_steps=30,
                                scaler_type='standard',
                                start_padding_enabled=True
                                ),
                            TimesNet(h=int(hrz),
                                     input_size=int(hrz) * q,
                                     # output_size=horizon,
                                     max_steps=30,
                                     scaler_type='standard',
                                     start_padding_enabled=True
                                     ),
                            TimeMixer(h=int(hrz),
                                      input_size=int(hrz) * q,
                                      # output_size=horizon,
                                      max_steps=30,
                                      scaler_type='standard',
                                      # start_padding_enabled=True,
                                      n_series=1
                                      ),
                            PatchTST(h=int(hrz),
                                     input_size=int(hrz) * q,
                                     # output_size=horizon,
                                     max_steps=30,
                                     scaler_type='standard',
                                     start_padding_enabled=True
                                     ),
                            NBEATSx(h=int(hrz),
                                    input_size=int(hrz)* q,
                                    # output_size=horizon,
                                    max_steps=30,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),

                        ],
                        freq=means[st.session_state.freq]
                    )

                    Y_train_df = datafra[:-int(hrz)]
                    Y_test_df = datafra[-int(hrz):]
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
                                KAN(h=int(hrz),
                                    input_size=int(hrz)* q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "KAN": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"KAN": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "NBEATSx":
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         start_padding_enabled=True
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "PatchTST":
                        fcst = NeuralForecast(
                            models=[
                                PatchTST(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         start_padding_enabled=True
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "PatchTST": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        chr = go.Figure()

                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        my_bar.progress(100, "Надаю відповідь")
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"PatchTST": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "TimesNet":
                        fcst = NeuralForecast(
                            models=[
                                TimesNet(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         start_padding_enabled=True
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimesNet": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        if st.session_state.lang == "ukr":
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Дані',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Прогноз',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Дата',
                                yaxis_title='Значення'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Надаю відповідь")
                        else:
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Data',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Value'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Giving the answer")
                    if key_with_min_value == "TimeMixer":
                        fcst = NeuralForecast(
                            models=[
                                TimeMixer(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         # start_padding_enabled=True,
                                         n_series=1
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimeMixer": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        if st.session_state.lang == "ukr":
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Дані',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Прогноз',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Дата',
                                yaxis_title='Значення'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Надаю відповідь")
                        else:
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Data',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Value'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Giving the answer")
            if mdl == "None":
                if hrz == "None":
                    if st.session_state.lang == "ukr":
                        response = "Уточніть на скільки вперед робити прогноз"
                    else:
                        response = "Specify how far in advance to make the forecast"
                        
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                        input_size=int(hrz) * q,
                                        # output_size=horizon,
                                        max_steps=50,
                                        scaler_type='standard',
                                        start_padding_enabled=True
                                        ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)

                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        if st.session_state.date_not_n == True:
                            pred2[st.session_state.date] = [i for i in range(1, len(pred2) + 1)]
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)
                        if st.session_state.lang == "ukr":
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Дані',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Прогноз',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Дата',
                                yaxis_title='Значення'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Надаю відповідь")
                        else:
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Data',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Value'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Giving the answer")
                    else:
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                        input_size=int(inp_sz),
                                        # output_size=horizon,
                                        max_steps=50,
                                        scaler_type='standard',
                                        start_padding_enabled=True
                                        ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        if st.session_state.date_not_n == True:
                            predis[st.session_state.date] = [i for i in range(1, len(predis) + 1)]
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        if st.session_state.lang == "ukr":
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Дані',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Прогноз',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Дата',
                                yaxis_title='Значення'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Надаю відповідь")
                        else:
                            chr = go.Figure()

                            chr.add_trace(go.Scatter(
                                x=rest_of_data[st.session_state.date],
                                y=rest_of_data[st.session_state.target],
                                mode='lines',
                                name='Data',
                                line=dict(color='blue')
                            ))

                            chr.add_trace(go.Scatter(
                                x=last_days[st.session_state.date],
                                y=last_days[st.session_state.target],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red')
                            ))

                            chr.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Value'
                            )
                            st.session_state.fig_b = chr
                            st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                            my_bar.progress(100, "Giving the answer")
        elif tsk == "Anomaly":
            if st.session_state.lang == "ukr":
                my_bar.progress(50, "Модель навчається проведенню тестів на аномалії")
                st.session_state.m1 = "Ось табличка з даними після проведення тестування на аномалії"
                st.session_state.m2 = "Та також графік Вашого прогнозу. Синім зображені Ваші дані, зеленим - прогнозовані а червоним крапками місця з аномаліями"
                response = "Дякую за Ваш запит, ось резульати проведення тестування на аномалії"
            else:
                my_bar.progress(50, "Model is analyzing on anomalies")
                st.session_state.m1 = "Here is the data table after testing for anomalies"
                st.session_state.m2 = "And also a graph of your forecast. Blue shows your data, green shows the forecast, and red dots show areas with anomalies."
                response = "Thanks for your prompt, here is the result"
            model = NeuralForecast(
                models=[
                    NBEATSx(h=len(datafra),
                            input_size=30 * q,
                            # output_size=horizon,
                            max_steps=100,
                            scaler_type='standard',
                            start_padding_enabled=True
                            ),

                ],
                freq=means[st.session_state.freq]
            )
            model.fit(datafra)  

            predictions = model.predict(datafra.head(1))
            print(predictions)
            datafra['NBEATSx'] = predictions['NBEATSx']
            datafra['residuals'] = np.abs(datafra['y'] - datafra['NBEATSx'])

            threshold = 4 * datafra['residuals'].std()
            datafra['anomaly'] = datafra['residuals'] > threshold
            if st.session_state.date_not_n == True:
                datafra["ds"] = [i for i in range(1, len(datafra) + 1)]
            if st.session_state.lang == "ukr":
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=datafra['ds'], y=datafra['y'], mode='lines', name='Дані', line=dict(color='blue')))

                fig.add_trace(go.Scatter(x=datafra['ds'], y=datafra['NBEATSx'], mode='lines', name='Прогнозовано',
                                        line=dict(color='green')))


                anomalies = datafra[datafra['anomaly'] == True]
                fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Аномалія',
                                        marker=dict(color='red', size=8)))

                
                fig.update_layout(
                    title='Графік аномалій',
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white'
                )
                my_bar.progress(100, "Надаю відповідь")
                datafra = datafra.rename(columns={"NBEATSx": "preds"})
                # Show the plot
                st.session_state.fig_b = fig
                st.session_state.dataai = datafra.drop(['unique_id', 'residuals'], axis=1)
            else:
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=datafra['ds'], y=datafra['y'], mode='lines', name='Data', line=dict(color='blue')))

                fig.add_trace(go.Scatter(x=datafra['ds'], y=datafra['NBEATSx'], mode='lines', name='Forecasted',
                                        line=dict(color='green')))


                anomalies = datafra[datafra['anomaly'] == True]
                fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Anomaly',
                                        marker=dict(color='red', size=8)))

                
                fig.update_layout(
                    title='Anomaly plot',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_white'
                )
                my_bar.progress(100, "Giving the answer")
                datafra = datafra.rename(columns={"NBEATSx": "preds"})
                # Show the plot
                st.session_state.fig_b = fig
                st.session_state.dataai = datafra.drop(['unique_id', 'residuals'], axis=1)
        elif tsk == "None":
            # chatco = client.chat.completions.create(
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": f"{res}",
            #         }
            #     ],
            #     model="llama-3.1-70b-versatile"
            # )
            # response = chatco.choices[0].message.content
            if st.session_state.lang == "ukr":
                response = "Вибачте, але здається що Ви не вказали, що конкретно хочете робити"
            else:
                response = "Sorry, but it seems like, you didn't really mention what you want to do"

    except: pass
    if st.session_state.lang == "ukr":
        my_bar.progress(100, "Надаю відповідь")
    else:
        my_bar.progress(100, "Giving the answer")
    my_bar.empty()
    for word in response.split():
                yield word + " "
                time.sleep(0.1)




# st.set_page_config(
#     page_title="Аналіз аномалій",
#     layout="wide",
#     initial_sidebar_state="auto"
# )




# if __name__ == "__main__":
if st.session_state.df is not None:
    st.session_state.no_d = None
    print(st.session_state.fig_b)
    print(st.session_state.dataai)
    ds_for_pred = pd.DataFrame()
    ds_for_pred["y"] = st.session_state.df[st.session_state.target]
    try:
        st.session_state.date_not_n = False
        ds_for_pred["ds"] = st.session_state.df[st.session_state.date]
        ds_for_pred['ds'] = pd.to_datetime(ds_for_pred['ds'])
    except:
        st.session_state.date_not_n = True
        ds_for_pred['ds'] = [i for i in range(1, len(ds_for_pred) + 1)]
    if st.session_state.lang == "ukr":
        st.title("ШІ помічник")
        st.markdown(f"### Зараз ШІ помічник працює з набором даних: {st.session_state.name}")
        st.write(" ")
        st.markdown("## Приклади запитів до ШІ помічника:")
    else:
        st.title("AI assistant")
        st.markdown(f"### Currently AI assistant working with dataset: {st.session_state.name}")
        st.write(" ")
        st.markdown("## Examples of prompts to AI assistant:")
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            # if isinstance(message["content"],str):
            #     st.markdown(message["content"])
            # elif isinstance(message["content"],st.delta_generator.DeltaGenerator):
            #     st.plotly_chart(message["content"])
            # else:
            st.write(message["content"])
    st.write(" ")
    if st.session_state.lang == "ukr":
        st.markdown("## Чат")
    else: st.markdown("## Chat")
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # if isinstance(message["content"],str):
            #     st.markdown(message["content"])
            # elif isinstance(message["content"],st.delta_generator.DeltaGenerator):
            #     st.plotly_chart(message["content"])
            # else:
            st.write(message["content"])
    if st.session_state.lang == "ukr":
        if prompt := st.chat_input("Напишіть свій запит і отримайте відповідь"):

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # st.session_state.messages.append({"role": "assistant", "content": "Дякую за запитання, інтерпритую ваш запит до моделі прогнозування"})

            with st.chat_message("assistant"):
                gen = response_generator(ds_for_pred, prompt)
                response = st.write_stream(gen)
                st.session_state.messages.append({"role": "assistant", "content": response})
                if st.session_state.dataai is not None:
                    r1 = st.write_stream(response_1(st.session_state.m1))
                    dai = st.write(st.session_state.dataai)
                    r2 = st.write_stream(response_1(st.session_state.m2))
                    chart = st.plotly_chart(st.session_state.fig_b, use_container_width=True)
                    print(dai)
                    print("-"*1000)
                    st.session_state.messages.append({"role": "assistant", "content": r1})
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.dataai})
                    st.session_state.messages.append({"role": "assistant", "content": r2})
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.fig_b})
    else:
        if prompt := st.chat_input("Type in your prompt and get an answer"):

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # st.session_state.messages.append({"role": "assistant", "content": "Дякую за запитання, інтерпритую ваш запит до моделі прогнозування"})

            with st.chat_message("assistant"):
                gen = response_generator(ds_for_pred, prompt)
                response = st.write_stream(gen)
                st.session_state.messages.append({"role": "assistant", "content": response})
                if st.session_state.dataai is not None:
                    r1 = st.write_stream(response_1(st.session_state.m1))
                    dai = st.write(st.session_state.dataai)
                    r2 = st.write_stream(response_1(st.session_state.m2))
                    chart = st.plotly_chart(st.session_state.fig_b, use_container_width=True)
                    print(dai)
                    print("-"*1000)
                    st.session_state.messages.append({"role": "assistant", "content": r1})
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.dataai})
                    st.session_state.messages.append({"role": "assistant", "content": r2})
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.fig_b})


else:
    st.session_state.no_d = True
    if st.session_state.lang == "ukr":
        st.title("ШІ помічник")
        st.write(" ")
        st.markdown("## Приклади запитів до ШІ помічника:")
    else:
        st.title("AI assistant")
        st.write(" ")
        st.markdown("## Examples of prompts to AI assistant:")
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            # if isinstance(message["content"],str):
            #     st.markdown(message["content"])
            # elif isinstance(message["content"],st.delta_generator.DeltaGenerator):
            #     st.plotly_chart(message["content"])
            # else:
            st.write(message["content"])
    st.write(" ")
    st.markdown("## Чат")
    with st.chat_message("assistant"):
        if st.session_state.lang == "ukr":
            st.write_stream(response_1("Перед тим як працювати зі мною, оберіть дані з якими Ви будете працювати у розділі 'Дані'"))
        else:
            st.write_stream(response_1("Before working with me, please select the data to work with at 'Data' section "))
