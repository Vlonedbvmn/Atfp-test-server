import streamlit as st
import pandas as pd
import os
from streamlit_js_eval import streamlit_js_eval



# st.set_page_config(
#     page_title="ATFP",
#     layout="wide",
#     initial_sidebar_state="auto"
# )


if all(key not in st.session_state.keys() for key in ('model', 'num_features', 'score')):
    st.session_state['num_features'] = []
    st.session_state['score'] = []
    st.session_state.clicked = False
    st.session_state.clicked2 = False
    st.session_state.df = None
    st.session_state.date = ""
    st.session_state.target = ""


def display_df():
    df = pd.DataFrame({"Model": st.session_state['model'],
                       "Number of features": st.session_state['num_features'],
                       "F1-Score": st.session_state['score']})

    sorted_df = df.sort_values(by=['F1-Score'], ascending=False).reset_index(drop=True)

    st.write(sorted_df)


st.title("ATFP - AI Timeseries Forecasting Platform")

if st.session_state.lang == "ukr":
    st.subheader(
    "Вітаємо! Ви знаходитеся на сторінці проєкту, що представляє собою платформу для дослідників у галузі прогнозування часових рядів із використанням методів машинного навчання. Для початку роботи перейдіть до розділу 'Дані', щоб завантажити дані або обрати із запропонованих наборів даних, для яких Ви будете здійснювати прогнозування.")
else:
    st.subheader(
    "Welcome! You are on the page of a project that represents a platform for researchers in the field of time series forecasting using machine learning methods. To get started, go to the 'Data' section to upload your own data or choose from the available datasets for which you will make forecasts.")
video_f = open("video_guide.mp4", "rb")
video_bytes = video_f.read()
# st.subheader(" ")
st.divider()
# st.subheader(
#     f"Зараз Ви обрали, що Ви '{st.session_state.role}' у сфері прогнозування часових рядів. Щоб змінити, просто оновіть сторінку")
# if st.button("Оновити"):
#     streamlit_js_eval(js_expressions="parent.window.location.reload()")
# st.divider()
if st.session_state.lang == "ukr":
    st.subheader("Відео інструкція користування застосунком")
else:
    st.subheader("Video tutorial for using the application")
st.video(video_bytes)
