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


st.subheader(
    "Вітаємо! Ви знаходитеся на сторінці проєкту, що представляє собою платформу для дослідників у галузі прогнозування часових рядів із використанням методів машинного навчання. Для початку роботи перейдіть до розділу 'Дані', щоб завантажити дані або обрати із запропонованих наборів даних, для яких Ви будете здійснювати прогнозування.")
video_f = open("instruction_2.mp4", "rb")
video_bytes = video_f.read()
# st.subheader(" ")
st.divider()
# st.subheader(
#     f"Зараз Ви обрали, що Ви '{st.session_state.role}' у сфері прогнозування часових рядів. Щоб змінити, просто оновіть сторінку")
# if st.button("Оновити"):
#     streamlit_js_eval(js_expressions="parent.window.location.reload()")
# st.divider()
st.subheader("Відео інструкція користування застосунком")
st.video(video_bytes)