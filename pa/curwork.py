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


import smtplib
from email.message import EmailMessage
import imghdr

def send_form_email(text, uploaded_images):
    sender_email = "atfp.webplatform@gmail.com"
    sender_password = "wydv sclj qlvn uppf"
    smtp_server = "smtp.gmail.com"
    smtp_port = 465 


    msg = EmailMessage()
    msg["Subject"] = "Відгук від ATFP"
    msg["From"] = sender_email
    msg["To"] = "ancorufio@gmail.com"
    msg.set_content(text)


    for image in uploaded_images:
        image_data = image.read()  
        image_type = imghdr.what(None, h=image_data)
        if not image_type:
            image_type = "jpeg"
        msg.add_attachment(
            image_data,
            maintype="image",
            subtype=image_type,
            filename=image.name
        )


    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)



if st.session_state.lang == "ukr":
    st.markdown("# Зараз працюємо над...")
    st.markdown("#### • Переклад вебплатформи на англійську")
    st.markdown("#### • Алгоритм проведення аналізу на аномалії")
    st.markdown("#### • Додання розгорнутих пояснень та візуалізації процесу навчання")
    st.markdown("#### • Зв'язка з google drive та dropbox")
else:
    st.markdown("# Currently working on...")
    st.markdown("#### • Translation of webplatform to english language")
    st.markdown("#### • Anomaly detection alghoritm")
    st.markdown("#### • Adding extended training process log visuals")
    st.markdown("#### • Google drive and dropbox link up")

st.markdown("### ")
st.divider()
with st.container():
    if st.session_state.lang == "ukr":
        st.markdown("# Зворотній зв'язок")
    else:
        st.markdown("# Feedback") 

with st.form("my_form"):
    if st.session_state.lang == "ukr":
        st.write("Залишились питання чи зауваження? Напишіть його нам")
        text_input = st.text_input(
        "Відгук 👇",
        placeholder="Пишіть текст тут",)
        uploaded_images = st.file_uploader("Завантажити скріншот", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Підтвердити")
        if submitted:
            send_form_email(text_input, uploaded_images)
            st.write("Дякую за відгук!")
    else: 
        st.write("Still have some questions or ? Give us a feedback")
        text_input = st.text_input(
        "Feedback 👇",
        placeholder="Enter text here",)
        uploaded_images = st.file_uploader("Upload screenshot", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            send_form_email(text_input, uploaded_images)
            st.write("Thanks for your feedback!")
st.write("Outside the form")
    