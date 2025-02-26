import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import gettext

languages = {"English": "en", "Español": "es"}
selected_language = st.sidebar.selectbox("Choose your language", list(languages.keys()))
lang_code = languages[selected_language]

# Load the appropriate translation (assuming your locale files are in the 'locales' folder)
translation = gettext.translation('messages', localedir='locales', languages=[lang_code], fallback=True)
translation.install()
_ = translation.gettext

st.write(_("Welcome to my app!"))


# Initialize session state variables
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

# Set Streamlit page config
# st.set_page_config(
#     page_title="Дані",
#     layout="wide",
#     initial_sidebar_state="auto"
# )


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


# Define button click functions
def click_button():
    st.session_state.clicked = True
    st.session_state.clicked2 = False
    st.session_state.submitted = False  # Reset submitted state on new button click


def click_button2():
    st.session_state.clicked = False
    st.session_state.clicked2 = True
    st.session_state.submitted = False  # Reset submitted state on new button click


def submit_data(dataframe, date_col, target_col, name, fr):
    st.session_state.df = dataframe
    st.session_state.date = date_col
    st.session_state.target = target_col
    st.session_state.name = name
    st.session_state.freq = fr
    st.session_state.submitted = True


# Main function
# if __name__ == "__main__":

with st.container():
    st.title("Оберіть з якими даними Ви бажаєте працювати")

# Create two columns for buttons
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

# Button for selecting experimental data
with col1:
    st.button(label="Обрати тестувальні", on_click=click_button)
st.markdown('</div>', unsafe_allow_html=True)
# Button for selecting own data
with col4:
    st.button(label="Обрати свої", on_click=click_button2)

# If experimental data button is clicked, show additional options
if st.session_state.clicked:
    st.markdown(
        "### Ви обрали тестові дані. Це набори даних, призначені для тестування, які дозволяють ознайомитися з функціональними можливостями проєкту та визначити, яка модель буде найбільш відповідною.")
    st.write("Оберіть тестовий набір даних:")
    st.markdown('<div class="data-button-container">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t1 = st.button(label="Тестовий набір даних 1")
    with c2:
        t2 = st.button(label="Тестовий набір даних 2")
    with c3:
        t3 = st.button(label="Тестовий набір даних 3")
    with c4:
        t4 = st.button(label="Тестовий набір даних 4 (Тест на аномалії)")
    st.markdown('</div>', unsafe_allow_html=True)
    # Test 1: Load sales.csv and allow submission
    if t1:
        dataframe = pd.read_csv("sales.csv")
        st.markdown(
            "### Тестовий набір даних 1 - це штучно згенерозаний датасет, що представляє собою щоденну зміну акцій умовної компанії.")
        st.markdown(
            "[Посилання на датасет](https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані', line=dict(color='blue')))

            # Add title and labels
            fig.update_layout(
                title = "Тестовий набір даних 1",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.button(label="Підтвердити", key="submit1", on_click=submit_data,
              args=(dataframe, "date", "target", "Тестовий набір даних 1", "День"))

    # Test 2: Load AXISBANK-BSE.csv and allow submission
    if t2:
        dataframe = pd.read_csv("Weather_dataset.csv")
        st.markdown(
            "### Тест набір даних 2 - це  датасет, що являє собою часовий ряд щогодинної зміни середньої температури в Німеччині.")
        st.markdown("[Посилання на датасет](https://www.kaggle.com/datasets/parthdande/timeseries-weather-dataset)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані',
                           line=dict(color='blue')))

            # Add title and labels
            fig.update_layout(
                title="Тестовий набір даних 2",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.button(label="Підтвердити", key="submit2", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 2", "Година"))


    # Test 3: Load electricityConsumptionAndProduction.csv and allow submission
    if t3:
        dataframe = pd.read_csv("electricityConsumptionAndProductioction.csv")
        st.markdown(
            "### Тестовий набір даних 3 - це датасет, що являє собою часовий ряд щогодинної зміни обсягу спожитої електроенергії в Румунії.")
        st.markdown(
            "[Посилання на датасет](https://www.kaggle.com/datasets/srinuti/residential-power-usage-3years-data-timeseries)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані',
                           line=dict(color='blue')))

            # Add title and labels
            fig.update_layout(
                title="Тестовий набір даних 3",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.button(label="Підтвердити", key="submit3", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 3", "Година"))

    if t4:
        dataframe = pd.read_csv("anomaly.csv")
        st.markdown(
            "### Тестовий набір даних 4 - це копія тестового набору даних 1, але з доданими аномаліями для наочної демонстрації можливості тестування на аномалії .")
        st.markdown(
            "[Посилання на датасет](https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features)")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(dataframe)
        with c2:
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(x=dataframe['date'], y=dataframe['target'], mode='lines', name='Дані',
                           line=dict(color='blue')))

            # Add title and labels
            fig.update_layout(
                title="Тестовий набір даних 4",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.button(label="Підтвердити", key="submit4", on_click=submit_data,
                  args=(dataframe, "date", "target", "Тестовий набір даних 4", "День"))

# If own data button is clicked, allow file upload
if st.session_state.clicked2:
    st.markdown("### Ви обрали свої дані. Наразі основні вимоги до даних це:")
    st.markdown(
        "### • Завжди повинні бути 2 колонки: час та значення показника, для якого здійснюється прогнозування")
    st.markdown("### • Бажано оформити колонки з часом під формат timestamp, date, або datetime")
    st.markdown(
        "### • Бажано, щоб не було пропусків між записами значень, бо пропуски будуть заміщуватися, а отже якість прогнозування може погіршитися")
    st.markdown("###  ")
    uploaded_file = st.file_uploader("Оберіть файл (Підтриуються формати .csv та .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name[-4:] == "xlsx":
            print(uploaded_file)
            dataframe = pd.read_excel(uploaded_file)
            st.write(dataframe)
        else:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)

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

            # Add actual values
            fig.add_trace(
                go.Scatter(x=dataframe[option], y=dataframe[option2], mode='lines', name='Дані',
                           line=dict(color='blue')))

            start_range = dataframe[option].iloc[val1]  # example start index
            end_range = dataframe[option].iloc[val2]  # example end index

            # Get the y-range for the vertical lines (you can also use fixed values)
            y_min = dataframe[option2].min()
            y_max = dataframe[option2].max()

            # Add vertical red line at the start of the chosen range
            fig.add_trace(
                go.Scatter(
                    x=[start_range, start_range],
                    y=[y_min, y_max],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Start Range',
                    showlegend=False  # Hide legend entry if not needed
                )
            )

            # Add vertical red line at the end of the chosen range
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

            # Add title and labels
            fig.update_layout(
                title=f"{uploaded_file.name}",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white',
                shapes=[
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",  # 'paper' makes it span the full y-range of the plot
                        x0=start_range,
                        y0=0,
                        x1=end_range,
                        y1=1,
                        fillcolor="rgba(255, 0, 0, 0.1)",  # half-transparent red
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

            # Add actual values
            fig.add_trace(
                go.Scatter(x=dataframe[option], y=dataframe[option2], mode='lines', name='Дані',
                           line=dict(color='blue')))


            # Add title and labels
            fig.update_layout(
                title=f"{uploaded_file.name}",
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white',
            )
            st.plotly_chart(fig, use_container_width=True)

            st.button(label="Підтвердити", key="submit_own", on_click=submit_data,
                      args=(dataframe, option, option2, uploaded_file.name, fr))

# After submission, show the dataframe and success message
if st.session_state.submitted:
    st.success(
        f"Дані датасету {st.session_state.name} успішно завантажені! Тепер можете перейти до розділу 'Налаштування моделі'")

st.divider()
if st.session_state.name is not None:
    st.header(f"Наразі обраний датасет: {st.session_state.name}")
    with st.expander("Подивитися обраний датасет:"):
        st.write(st.session_state.df)
