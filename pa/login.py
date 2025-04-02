import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np
import datetime
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import streamlit as st
# import smtplib
# from email.message import EmailMessage
# import imghdr
if "logstate" not in st.session_state:
    st.session_state.logstate = True
if "regstate" not in st.session_state:
    st.session_state.regstate = False
if "finstate" not in st.session_state:
    st.session_state.finstate = False

if st.session_state.logstate:
    with st.form("my_form"):
        st.title("Ввійдіть у свій акаунт")
        username = st.text_input("Введіть ім'я користувача:", placeholder="Введіть тут...")
        password = st.text_input("Введіть пароль користувача:", placeholder="Введіть тут...")

        submitted = st.form_submit_button("Ввійти")
        if submitted:
            st.success("Ви ввійшли у свій аккаунт")
            st.write(f"Назар: {username}")