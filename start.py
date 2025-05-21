import streamlit as st
import pandas as pd
import os

if "lang" not in st.session_state:
    st.session_state.lang = "ukr"
if "user" not in st.session_state:
    st.session_state.user = None

os.environ['NIXTLA_ID_AS_COL'] = '1'

if 'role' not in st.session_state:
    st.session_state.role = None

# st.logo("atfp-logo.png")
st.set_page_config(
    # page_title="ATFP",
    layout="wide",
    initial_sidebar_state="auto"
)

st.logo("atfp-logo-rgb.png")

st.html("""
  <style>
    [alt=Logo] {
      height: 3rem;
    }
  </style>
        """)

# st.title("ATFP - AI Timeseries Forecasting Platform")
#
# st.subheader(
#     "Вітаю, це сторінка проєкту, який є програмним застосунком для дослідників у сфері прогнозування часових рядів використовуючи методи машинного навчання. Для початку роботи натисніть розділ Дані, щоб програма отримала дані з якими ви хочете працювати та отримувати прогнози.")
# video_f = open("instruction_2.mp4", "rb")
# video_bytes = video_f.read()
# st.subheader(" ")
# st.divider()
# st.subheader("Відео інструкція користування застосунком")
# st.video(video_bytes)


# if st.session_state.role == "Аматор або професіонал":
selected_language = st.sidebar.selectbox("Choose language:", ["Українська", "English"])

try:
    if st.session_state.user is not None:
        st.sidebar.write(f"Logged in as {st.session_state.user}")
    else:
        st.sidebar.write("You are not logged in")
except:
    pass

if selected_language == "Українська":
    st.session_state.lang = "ukr"
else:
    st.session_state.lang = "eng"
# else:
# selected_language = st.sidebar.selectbox("Choose language:", ["Українська", "English"])
# if selected_language == "Українська":
#     st.session_state.lang = "ukr"
# else:
#     st.session_state.lang = "eng"


# Load the appropriate translation (assuming your locale files are in the 'locales' folder)
if st.session_state.lang == "ukr":
    p1 = st.Page(
        "pa/ATFP.py",
        title="Головна",
    )
    p2 = st.Page(
        "pa/Дані.py",
        title="Дані",
    )
    p3 = st.Page(
        "pa/Налаштування моделі.py",
        title="Налаштування моделі",
    )

    p4 = st.Page(
        "pa/Прогноз.py",
        title="Прогноз",
    )
    p5 = st.Page(
        "pa/Тест на аномалії.py",
        title="Тест на аномалії",
    )
    p6 = st.Page(
        "pa/ШІ помічник.py",
        title="ШІ помічник",
    )
    p7 = st.Page(
        "pa/Порівняти.py",
        title="Порівняти",
    )
    p8 = st.Page(
        "pa/Plugins.py",
        title="Плагіни (New)"
    )
    p9 = st.Page(
        "pa/curwork.py",
        title="Зворотній зв'язок"
    )
    p10 = st.Page(
        "pa/login.py",
        title="Ввійти"
    )
    pg = st.navigation({"": [p1, p2], "Для фахівців:": [p3, p4, p8, p5], "Для всіх:": [p6, p9, p10]})
    pg.run()
else:
    p1 = st.Page(
        "pa/ATFP.py",
        title="Home",
    )
    p2 = st.Page(
        "pa/Дані.py",
        title="Data",
    )
    p3 = st.Page(
        "pa/Налаштування моделі.py",
        title="Model",
    )

    p4 = st.Page(
        "pa/Прогноз.py",
        title="Forecast",
    )
    p5 = st.Page(
        "pa/Тест на аномалії.py",
        title="Anomaly analysis",
    )
    p6 = st.Page(
        "pa/ШІ помічник.py",
        title="AI assistant",
    )
    p7 = st.Page(
        "pa/Порівняти.py",
        title="Compare",
    )
    p8 = st.Page(
        "pa/Plugins.py",
        title="Plugins (New)"
    )
    p9 = st.Page(
        "pa/curwork.py",
        title="Feedback"
    )
    p10 = st.Page(
        "pa/login.py",
        title="Log in"
    )
    pg = st.navigation({"": [p1, p2], "Professional level:": [p3, p4, p8, p5], "Regular:": [p6, p9, p10]})
    pg.run()


# if st.session_state.role == "Новачок":
#     pg = st.navigation([p1, p2, p6])
#     pg.run()

# if st.session_state.role is None:
    # st.title("Перед користуванням цим застосунком, оберіть хто Ви у сфері прогнозування часових рядів")
    #
    # role = st.selectbox("Роль:", ["Аматор або професіонал", "Новачок"])
    # if st.button("Підтвердити"):
    #     st.session_state.role = role
    #     st.rerun()



