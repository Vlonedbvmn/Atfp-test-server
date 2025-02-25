import streamlit as st
import pandas as pd
import os



st.set_page_config(
    page_title="ATFP",
    layout="wide",
    initial_sidebar_state="auto"
)

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


if __name__ == "__main__":
    st.title("ATFP - AI Timeseries Forecasting Platform")

    st.subheader(
        "Вітаю, це сторінка проєкту, який є програмним застосунком для дослідників у сфері прогнозування часових рядів використовуючи методи машинного навчання. Для початку роботи натисніть розділ Дані, щоб програма отримала дані з якими ви хочете працювати та отримувати прогнози.")
    video_f = open("instruction_2.mp4", "rb")
    video_bytes = video_f.read()
    st.subheader(" ")
    st.divider()
    st.subheader("Відео інструкція користування застосунком")
    st.video(video_bytes)