import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
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

# Initialize session state variables
if 'predicted' not in st.session_state:
    st.session_state.predicted = None

if 'predicted2' not in st.session_state:
    st.session_state.predicted2 = None

if 'plotp' not in st.session_state:
    st.session_state.plotp = None

if 'plotp2' not in st.session_state:
    st.session_state.plotp2 = None

if 'bp' not in st.session_state:
    st.session_state.bp = None



# Set Streamlit page config
# st.set_page_config(
#     page_title="Прогноз",
#     layout="wide",
#     initial_sidebar_state="auto"
# )


# Define button click functions



def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data



def predd(datafre):

    if st.session_state.inst_name == "SNN":
        try:
            print(datafre)
            qu = int(round(len(datafre)*0.1, 0))
            h = datafre[-qu:]["y"].tolist()
            new_sample = datafre["y"].tolist()[-st.session_state.inp:]
            new_sample = np.array(new_sample).reshape(-1, 1)

            # .reshape(orig_shape)
            def make_prediction(input_values):

                input_values = st.session_state.scaler.transform(input_values)
                st.session_state.model_forecast.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(input_values, dtype=torch.float32).to(
                        st.session_state.device).unsqueeze(0)
                    reservoir_state = torch.tanh(
                        st.session_state.win @ input_tensor.T + st.session_state.wres @ torch.rand(
                            st.session_state.reser_size, 1).to(st.session_state.device))
                    reservoir_state = reservoir_state.T
                    reservoir_state = reservoir_state.unsqueeze(2)
                    prediction = st.session_state.model_forecast(reservoir_state).cpu().numpy()
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
            for i in range(1, len(h)+1):
                pr1dates.append(i)
            for i in range(1, len(pr)+1):
                pr2dates.append(i)

            st.session_state.predicted = pd.DataFrame({
                st.session_state.date: pr1dates,
                st.session_state.target: h
            })
            st.session_state.predicted2 = pd.DataFrame({
                st.session_state.date: pr2dates,
                st.session_state.target: pr
            })
            # preds = preds.drop(columns=['unique_id'], inplace=True)
            # print(preds)
            # print(-qu)
            # pred1 = datafre[-qu:]
            # pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
            #              inplace=True)
            # pred1.drop(columns=['unique_id'], inplace=True)
            # print("1 df")
            # print(pred1)
            # print("2 df")
            # print(preds)
            # st.session_state.predicted = pd.concat([pred1, preds], ignore_index=True)
            # print("finale df")
            # print(len(st.session_state.predicted))
            # print(st.session_state.predicted)
            # st.session_state.predicted2 = preds
        except:
            st.warning('Щоб зробити прогноз оберіть спочатку модель)', icon="⚠️")
    else:
        try:
            print(datafre)
            qu = int(round(len(datafre)*0.1, 0))
            model = st.session_state.model_forecast
            print(2)
            preds = model.predict(df=datafre)
            print(preds)
            preds.rename(columns={'ds': st.session_state.date, str(st.session_state.inst_name): st.session_state.target}, inplace=True)
            preds.reset_index(drop=True, inplace=True)
            # preds = preds.drop(columns=['unique_id'], inplace=True)
            print(preds)
            print(-qu)
            pred1 = datafre[-qu:]
            pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                         inplace=True)
            pred1.drop(columns=['unique_id'], inplace=True)
            print("1 df")
            print(pred1)
            print("2 df")
            print(preds)
            st.session_state.predicted = pd.concat([pred1, preds], ignore_index=True)
            print("finale df")
            print(len(st.session_state.predicted))
            print(st.session_state.predicted)
            st.session_state.predicted2 = preds
        except:
            st.warning('Щоб зробити прогноз оберіть спочатку модель)', icon="⚠️")

# Main function
# if __name__ == "__main__":
print(st.session_state.inst_name)
# try:
if st.session_state.inst_name is not None:
    if st.session_state.lang == "ukr":
        with st.container():
            st.title("Прогноз")

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        # Button for selecting experimental data
        with col1:
            st.markdown(f"### Модель: {st.session_state.inst_name}")
            dff = pd.DataFrame()
            print(st.session_state.df_forpred)
            dff["ds"] = st.session_state.df_forpred["ds"]
            dff["y"] = st.session_state.df_forpred["y"]
            dff["unique_id"] = st.session_state.df_forpred["unique_id"]
            print(dff)
            st.button(label="Прогноз", key="pr", on_click=predd,
                      args=(dff,))

        # Button for selecting own data
        with col2:
            st.markdown(f"### Дані:")
            with st.expander("Подивитися обраний датасет:"):
                st.write(st.session_state.df)



        st.divider()
        st.markdown(f"### Результати прогнозу")
    else:
        with st.container():
            st.title("Forecast")

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        # Button for selecting experimental data
        with col1:
            st.markdown(f"### Model: {st.session_state.inst_name}")
            dff = pd.DataFrame()
            print(st.session_state.df_forpred)
            dff["ds"] = st.session_state.df_forpred["ds"]
            dff["y"] = st.session_state.df_forpred["y"]
            dff["unique_id"] = st.session_state.df_forpred["unique_id"]
            print(dff)
            st.button(label="Forecast", key="pr", on_click=predd,
                      args=(dff,))

        # Button for selecting own data
        with col2:
            st.markdown(f"### Data:")
            with st.expander("Check chosen dataset:"):
                st.write(st.session_state.df)

        st.divider()
        st.markdown(f"### Forecast results")
    if st.session_state.predicted2 is not None:
        if st.session_state.lang == "ukr":
            col3, col4 = st.columns(2)
            with col3:
                with st.expander("Подивитись прогнозні значення:"):
                    st.write(st.session_state.predicted2)
            with col4:

                st.download_button(
                    label="Завантажити прогноз як файл .csv",
                    data=st.session_state.predicted2.to_csv().encode("utf-8"),
                    file_name="prediction.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Завантажити прогноз як файл .xlsx",
                    data=to_excel(st.session_state.predicted2),
                    file_name="prediction.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.divider()

            st.markdown(f"### Дашборд прогнозу")
            st.markdown("# ")
            if st.session_state.date_not_n == True:
                st.session_state.predicted[st.session_state.date] = [i for i in range(1, len(st.session_state.predicted) + 1)]
            else: pass
            last_days = st.session_state.predicted.tail(st.session_state.horiz)
            rest_of_data = st.session_state.predicted.iloc[:-st.session_state.horiz]

            val = len(last_days)



            cool1, cool2 = st.columns([2,5])

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
                if st.session_state.inst_name == "SNN":
                    st.write(last_days[:(slid)].describe().head(7), use_container_width=True)
                else:
                    st.write(last_days[:(slid)].describe().drop(["unique_id"], axis=1).head(7), use_container_width=True)

            with cool2:

                st.session_state.plotp = go.Figure()
                st.session_state.plotp.add_trace(go.Scatter(
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
                    max_value = max(upper_forecast.tolist())+100
                    min_value = min(lower_forecast.tolist()) - 100
                # First, add the upper bound trace (invisible line) to serve as the fill ceiling.
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date][:(slid)],
                    y=upper_forecast,
                    mode='lines',
                    line=dict(color='rgba(0,128,0,0)'),  # fully transparent line
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # Next, add the lower bound trace that fills the area up to the previous (upper bound) trace.
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date][:(slid)],
                    y=lower_forecast,
                    mode='lines',
                    fill='tonexty',  # fills the area between this trace and the one above
                    fillcolor='rgba(0,128,0,0.2)',  # adjust the color and transparency as needed
                    line=dict(color='rgba(0,128,0,0)'),  # transparent line to keep the focus on the fill
                    name='Діапазон можливих значень прогнозу'
                ))

                # Finally, add the median (50% quantile) forecast line on top.
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date][:(slid)],
                    y=q50,
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green')
                ))
                # Update layout (optional)
                st.session_state.plotp.update_layout(
                    xaxis_title='Дата',
                    # yaxis_title='Значення',
                    yaxis=dict(
                        # range=[min_value, max_value],
                        title='Спрогнозовані значення'  # Optional: add a title for clarity
                    ),
                    title="Графік прогнозу", # Increase the overall height
                )

                # Show the plot
                st.plotly_chart(st.session_state.plotp, use_container_width=True)

                # Plot the data except the last seven days

                st.session_state.bp = go.Figure()

                st.session_state.bp.add_trace(go.Bar(
                    x=last_days[st.session_state.date][:(slid)],
                    y=last_days[st.session_state.target][:(slid)],
                    name='Прогноз',
                    marker_color='green'
                ))

                # Customize layout
                st.session_state.bp.update_layout(
                    title='Барплот прогнозу',
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white'
                )

                 # Display the Plotly chart in Streamlit
                st.plotly_chart(st.session_state.bp, use_container_width=True)
        else:
            col3, col4 = st.columns(2)
            with col3:
                with st.expander("Check forecasted values:"):
                    st.write(st.session_state.predicted2)
            with col4:

                st.download_button(
                    label="Download forecast file as .csv",
                    data=st.session_state.predicted2.to_csv().encode("utf-8"),
                    file_name="prediction.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download forecast file as .xlsx",
                    data=to_excel(st.session_state.predicted2),
                    file_name="prediction.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.divider()

            st.markdown(f"### Forecast dashboard")
            st.markdown("# ")
            if st.session_state.date_not_n == True:
                st.session_state.predicted[st.session_state.date] = [i for i in
                                                                     range(1, len(st.session_state.predicted) + 1)]
            else:
                pass
            last_days = st.session_state.predicted.tail(st.session_state.horiz)
            rest_of_data = st.session_state.predicted.iloc[:-st.session_state.horiz]

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
                if st.session_state.inst_name == "SNN":
                    st.write(last_days[:(slid)].describe().head(7), use_container_width=True)
                else:
                    st.write(last_days[:(slid)].describe().drop(["unique_id"], axis=1).head(7),
                             use_container_width=True)

            with cool2:

                st.session_state.plotp = go.Figure()
                st.session_state.plotp.add_trace(go.Scatter(
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
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date][:(slid)],
                    y=upper_forecast,
                    mode='lines',
                    line=dict(color='rgba(0,128,0,0)'),  # fully transparent line
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # Next, add the lower bound trace that fills the area up to the previous (upper bound) trace.
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date][:(slid)],
                    y=lower_forecast,
                    mode='lines',
                    fill='tonexty',  # fills the area between this trace and the one above
                    fillcolor='rgba(0,128,0,0.2)',  # adjust the color and transparency as needed
                    line=dict(color='rgba(0,128,0,0)'),  # transparent line to keep the focus on the fill
                    name='Range of possible forecast values'
                ))

                # Finally, add the median (50% quantile) forecast line on top.
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date][:(slid)],
                    y=q50,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
                # Update layout (optional)
                st.session_state.plotp.update_layout(
                    xaxis_title='Date',
                    # yaxis_title='Значення',
                    yaxis=dict(
                        # range=[min_value, max_value],
                        title='Forecasted values'  # Optional: add a title for clarity
                    ),
                    title="Forecast plot",  # Increase the overall height
                )

                # Show the plot
                st.plotly_chart(st.session_state.plotp, use_container_width=True)

                # Plot the data except the last seven days

                st.session_state.bp = go.Figure()

                st.session_state.bp.add_trace(go.Bar(
                    x=last_days[st.session_state.date][:(slid)],
                    y=last_days[st.session_state.target][:(slid)],
                    name='Forecast',
                    marker_color='green'
                ))

                # Customize layout
                st.session_state.bp.update_layout(
                    title='Forecast barplot',
                    xaxis_title='Date',
                    yaxis_title='Values',
                    template='plotly_white'
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(st.session_state.bp, use_container_width=True)



else:
    if st.session_state.lang == "ukr":
        st.warning('Щоб зробити прогноз оберіть спочатку модель', icon="⚠️")
    else:
        st.warning('To make a prediction, first select a model', icon="⚠️")
