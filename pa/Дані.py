import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go




if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'date' not in st.session_state:
    st.session_state.date = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'name' not in st.session_state:
    st.session_state.name = None
if 'frq' not in st.session_state:
    st.session_state.frq = None



means = {"Місяць": "M",
         "Година": "h",
         "Рік": "Y",
         "Хвилина": "T",
         "Секунда": "S",
         "День": "D",
         }




st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 30px;
    }

    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
    }

    .button-container button {
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }

    .button-container button:hover {
        background-color: #45a049;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 10px;
    }

    .data-button-container {
        display: flex;
        justify-content: center;
        gap: 15px;
    }

    .data-button-container button {
        font-size: 16px;
        padding: 10px 20px;
        background-color: #333;
        color: #f0f0f0;
        border-radius: 5px;
        border: 1px solid #555;
    }

    .data-button-container button:hover {
        background-color: #444;
    }
    </style>
    """, unsafe_allow_html=True)



def click_button():
    st.session_state.clicked = True
    st.session_state.clicked2 = False
    st.session_state.submitted = False  


def click_button2():
    st.session_state.clicked = False
    st.session_state.clicked2 = True
    st.session_state.submitted = False  

def submit_data(dataframe, date_col, target_col, name, fr):
    st.session_state.df = dataframe
    st.session_state.date = date_col
    st.session_state.target = target_col
    st.session_state.name = name
    st.session_state.freq = fr
    st.session_state.submitted = True




with st.container():
    if st.session_state.lang == "ukr":
        st.title("Оберіть з якими даними Ви бажаєте працювати")
    else:
        st.title("Choose the data you would like to work with")


st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)


with col1:
    if st.session_state.lang == "ukr":
        st.button(label="Обрати тестувальні", on_click=click_button)
    else:
        st.button(label="Choose test data", on_click=click_button)
st.markdown('</div>', unsafe_allow_html=True)

with col4:
    if st.session_state.lang == "ukr":
        st.button(label="Обрати свої", on_click=click_button2)
    else:
        st.button(label="Choose your own data", on_click=click_button2)
    


if st.session_state.clicked:
    if st.session_state.lang == "ukr":
        st.markdown(
        "### Ви обрали тестові дані. Це набори даних, призначені для тестування, які дозволяють ознайомитися з функціональними можливостями проєкту та визначити, яка модель буде найбільш відповідною.")
        st.write("Оберіть тестовий набір даних:")
    else:
        st.markdown(
        "### You have selected test data. These are datasets designed for testing, which allow you to familiarize yourself with the features of the project and determine which model will be most appropriate.")
        st.write("Select the test dataset:")
    st.markdown('<div class="data-button-container">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    if st.session_state.lang == "ukr":
        with c1:
            t1 = st.button(label="Тестовий набір даних 1")
        with c2:
            t2 = st.button(label="Тестовий набір даних 2")
        with c3:
            t3 = st.button(label="Тестовий набір даних 3")
        with c4:
            t4 = st.button(label="Тестовий набір даних 4 (Тест на аномалії)")
    else:
        with c1:
            t1 = st.button(label="Test dataset 1")
        with c2:
            t2 = st.button(label="Test dataset 2")
        with c3:
            t3 = st.button(label="Test dataset 3")
        with c4:
            t4 = st.button(label="Test dataset 4 (Anomaly analysis)")
    st.markdown('</div>', unsafe_allow_html=True)

    if t1:
        dataframe = pd.read_csv("sales.csv")
        if st.session_state.lang == "ukr":
            st.markdown(
                "### Тестовий набір даних 1 - це штучно згенерозаний датасет, що представляє собою щоденну зміну акцій умовної компанії.")
            st.markdown(
                "[Посилання на датасет](https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features)")
        else:
            st.markdown(
                "### Test dataset 1 is an artificially generated dataset representing the daily change in the stock prices of a hypothetical company.")
            st.markdown(
                "[Dataset link](https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()
            if st.session_state.lang == "ukr":

                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані', line=dict(color='blue')))
    

                fig.update_layout(
                    title = "Тестовий набір даних 1",
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
 
                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Data', line=dict(color='blue')))
    

                fig.update_layout(
                    title = "Test dataset 1",
                    xaxis_title='Date',
                    yaxis_title='Values',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        if st.session_state.lang == "ukr":
            st.button(label="Підтвердити", key="submit1", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 1", "День"))
        else:
            st.button(label="Submit", key="submit1", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 1", "День"))

    if t2:
        dataframe = pd.read_csv("Weather_dataset.csv")
        if st.session_state.lang == "ukr":
            st.markdown(
            "### Тестовий набір даних 2 - це  датасет, що являє собою часовий ряд щогодинної зміни середньої температури в Німеччині.")
            st.markdown("[Посилання на датасет](https://www.kaggle.com/datasets/parthdande/timeseries-weather-dataset)")
        else:
            st.markdown(
                "### Test dataset 2 is a dataset representing a time series of the hourly change in the average temperature in Germany.")
            st.markdown("[Dataset link](https://www.kaggle.com/datasets/parthdande/timeseries-weather-dataset)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()
            if st.session_state.lang == "ukr":

                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані',
                               line=dict(color='blue')))
    

                fig.update_layout(
                    title="Тестовий набір даних 2",
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
 
                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Data', line=dict(color='blue')))
    

                fig.update_layout(
                    title = "Test dataset 2",
                    xaxis_title='Date',
                    yaxis_title='Values',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        if st.session_state.lang == "ukr":
            st.button(label="Підтвердити", key="submit2", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 2", "Година"))
        else:
            st.button(label="Submit", key="submit2", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 2", "Година"))
        



    if t3:
        dataframe = pd.read_csv("electricityConsumptionAndProductioction.csv")
        if st.session_state.lang == "ukr":
            st.markdown(
            "### Тестовий набір даних 3 - це датасет, що являє собою часовий ряд щогодинної зміни обсягу спожитої електроенергії в Румунії.")
            st.markdown(
            "[Посилання на датасет](https://www.kaggle.com/datasets/srinuti/residential-power-usage-3years-data-timeseries)")
        else:
            st.markdown(
                "### Test dataset 3 is a dataset representing a time series of the hourly change in the amount of electricity consumed in Romania.")
            st.markdown("[Dataset link](https://www.kaggle.com/datasets/srinuti/residential-power-usage-3years-data-timeseries)")
        
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()
            if st.session_state.lang == "ukr":

                fig.add_trace(
                go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані',
                           line=dict(color='blue')))


                fig.update_layout(
                title="Тестовий набір даних 3",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:

                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Data', line=dict(color='blue')))
    

                fig.update_layout(
                    title = "Test dataset 3",
                    xaxis_title='Date',
                    yaxis_title='Values',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

        if st.session_state.lang == "ukr":
            st.button(label="Підтвердити", key="submit3", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 3", "Година"))
        else:
            st.button(label="Submit", key="submit3", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 3", "Година"))

    if t4:
        dataframe = pd.read_csv("anomaly.csv")
        if st.session_state.lang == "ukr":
            st.markdown(
            "### Тестовий набір даних 4 - це копія тестового набору даних 1, але з доданими аномаліями для наочної демонстрації можливості тестування на аномалії .")
            st.markdown(
            "[Посилання на датасет](https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features)")
        else:
            st.markdown(
                "### Test dataset 4 is a copy of test dataset 1, but with added anomalies for a visual demonstration of anomaly testing capabilities.")
            st.markdown("[Dataset link](https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()
            if st.session_state.lang == "ukr":

                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані',
                               line=dict(color='blue')))
    

                fig.update_layout(
                    title="Тестовий набір даних 4",
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:

                fig.add_trace(
                    go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Data', line=dict(color='blue')))
    

                fig.update_layout(
                    title = "Test dataset 4",
                    xaxis_title='Date',
                    yaxis_title='Values',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
        if st.session_state.lang == "ukr":
            st.button(label="Підтвердити", key="submit4", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 4", "День"))
        else:
            st.button(label="Submit", key="submit4", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 4", "День"))


if st.session_state.clicked2:
    if st.session_state.lang == "ukr":
        st.markdown("### Ви обрали свої дані. Наразі основні вимоги до даних це:")
        st.markdown(
            "### • Завжди повинні бути 2 колонки: час та значення показника, для якого здійснюється прогнозування")
        st.markdown("### • Бажано оформити колонки з часом під формат timestamp, date, або datetime")
        st.markdown(
            "### • Бажано, щоб не було пропусків між записами значень, бо пропуски будуть заміщуватися, а отже якість прогнозування може погіршитися")
        st.markdown("###  ")
    else:
        st.markdown("### You have selected to choose your own data. Currently, the main requirements for the data are:")
        st.markdown(
            "### • There must always be 2 columns: time and the value of the indicator for which forecasting is being performed.")
        st.markdown("### • It is recommended to format the time columns as timestamp, date, or datetime.")
        st.markdown(
            "### • It is recommended that there be no gaps between the value entries, as gaps will be filled, which may degrade the quality of the forecasting.")
        st.markdown("###  ")
    if st.session_state.lang == "ukr":
        uploaded_file = st.file_uploader("Оберіть файл (only .csv and .xlsx formats are supported)", type=["csv", "xlsx"])
    else:
        uploaded_file = st.file_uploader("Choose file (Supported files are .csv та .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name[-4:] == "xlsx":
            print(uploaded_file)
            dataframe = pd.read_excel(uploaded_file)
            st.write(dataframe)
        else:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)
        if st.session_state.lang == "ukr":
            option = st.selectbox(
                'Оберіть назву колонки з данними про дати',
                tuple(dataframe.columns.values))
    
            option2 = st.selectbox(
                'Оберіть назву колонки з данними про значення які ви хочете передбачити',
                tuple(dataframe.columns.values))
            fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                              ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
            if st.checkbox("Налаштувати проміжок роботи"):
    
                ran = st.select_slider(
                    "Проміжок (Виділений червоним):",
                    value=[1, 2],
                    options=[i for i in range(1, len(dataframe[option].tolist()))]
                )
                val1 = ran[0]
                val2 = ran[1]
                fig = go.Figure()
    

                fig.add_trace(
                    go.Scatter(x=dataframe[option], y=dataframe[option2], mode='lines', name='Дані',
                               line=dict(color='blue')))
    
                start_range = dataframe[option].iloc[val1]  
                end_range = dataframe[option].iloc[val2]  
    

                y_min = dataframe[option2].min()
                y_max = dataframe[option2].max()
    

                fig.add_trace(
                    go.Scatter(
                        x=[start_range, start_range],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Start Range',
                        showlegend=False  
                    )
                )
    

                fig.add_trace(
                    go.Scatter(
                        x=[end_range, end_range],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='End Range',
                        showlegend=False
                    )
                )
    

                fig.update_layout(
                    title=f"{uploaded_file.name}",
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white',
                    shapes=[
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",  
                            x0=start_range,
                            y0=0,
                            x1=end_range,
                            y1=1,
                            fillcolor="rgba(255, 0, 0, 0.1)",  
                            line=dict(width=0),
                            layer="below"
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)
    
                st.button(label="Підтвердити", key="submit_own", on_click=submit_data,
                          args=(dataframe.iloc[val1:val2], option, option2, uploaded_file.name, fr))
            else:
                fig = go.Figure()
    

                fig.add_trace(
                    go.Scatter(x=dataframe[option], y=dataframe[option2], mode='lines', name='Дані',
                               line=dict(color='blue')))
    
    

                fig.update_layout(
                    title=f"{uploaded_file.name}",
                    xaxis_title='Дата',
                    yaxis_title='Значення',
                    template='plotly_white',
                )
                st.plotly_chart(fig, use_container_width=True)
    
                st.button(label="Підтвердити", key="submit_own", on_click=submit_data,
                          args=(dataframe, option, option2, uploaded_file.name, fr))
        else:
            option = st.selectbox(
                'Select the column name with the date data',
                tuple(dataframe.columns.values))
    
            option2 = st.selectbox(
                'Select the column name with the values you want to predict',
                tuple(dataframe.columns.values))
            fr = st.selectbox("Select the frequency of data entries in the series:",
                              ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
            if st.checkbox("Set the working interval"):
    
                ran = st.select_slider(
                    "Interval (Highlighted in red):",
                    value=[1, 2],
                    options=[i for i in range(1, len(dataframe[option].tolist()))]
                )
                val1 = ran[0]
                val2 = ran[1]
                fig = go.Figure()
    

                fig.add_trace(
                    go.Scatter(x=dataframe[option], y=dataframe[option2], mode='lines', name='Data',
                               line=dict(color='blue')))
    
                start_range = dataframe[option].iloc[val1]  
                end_range = dataframe[option].iloc[val2]  
    

                y_min = dataframe[option2].min()
                y_max = dataframe[option2].max()
    

                fig.add_trace(
                    go.Scatter(
                        x=[start_range, start_range],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Start Range',
                        showlegend=False  
                    )
                )
    

                fig.add_trace(
                    go.Scatter(
                        x=[end_range, end_range],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='End Range',
                        showlegend=False
                    )
                )
    

                fig.update_layout(
                    title=f"{uploaded_file.name}",
                    xaxis_title='Data',
                    yaxis_title='Values',
                    template='plotly_white',
                    shapes=[
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",  
                            x0=start_range,
                            y0=0,
                            x1=end_range,
                            y1=1,
                            fillcolor="rgba(255, 0, 0, 0.1)",  
                            line=dict(width=0),
                            layer="below"
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)
    
                st.button(label="Submit", key="submit_own", on_click=submit_data,
                          args=(dataframe.iloc[val1:val2], option, option2, uploaded_file.name, fr))
            else:
                fig = go.Figure()
    

                fig.add_trace(
                    go.Scatter(x=dataframe[option], y=dataframe[option2], mode='lines', name='Data',
                               line=dict(color='blue')))
    
    
                fig.update_layout(
                    title=f"{uploaded_file.name}",
                    xaxis_title='Data',
                    yaxis_title='Values',
                    template='plotly_white',
                )
                st.plotly_chart(fig, use_container_width=True)
    
                st.button(label="Submit", key="submit_own", on_click=submit_data,
                          args=(dataframe, option, option2, uploaded_file.name, fr))

if st.session_state.submitted:
    if st.session_state.lang == "ukr":
        st.success(
            f"Дані датасету {st.session_state.name} успішно завантажені! Тепер можете перейти до розділу 'Налаштування моделі'")
    else:
        st.success(
            f"The dataset {st.session_state.name} has been successfully uploaded! You can now proceed to the 'Model Settings' section")

st.divider()
if st.session_state.name is not None:
    if st.session_state.lang == "ukr":
        st.header(f"Наразі обраний датасет: {st.session_state.name}")
        with st.expander("Подивитися обраний датасет:"):
            st.write(st.session_state.df)
    else:
        st.header(f"Currently selected dataset: {st.session_state.name}")
        with st.expander("Check chosen dataset:"):
            st.write(st.session_state.df)
