import streamlit as st
from sqlalchemy import text

if "page" not in st.session_state:
    st.session_state.page = "login"

conn = st.connection('mysql', type='sql')


def check_credentials(username: str, password: str) -> bool:
    """
    Returns True if there's a row in users with exactly this username & password.
    """

    df = conn.query(f"SELECT 1 FROM atfp WHERE username = '{username}' AND password = '{password}' LIMIT 1;")
    return not df.empty


def register_user(username: str, password: str) -> bool:
    """
    Registers a new user in the database.
    Returns True on success, False if the user already exists.
    """
    try:
        # Check if user already exists - using parameterized query for security
        query = text("SELECT 1 FROM atfp WHERE username = :username LIMIT 1")
        df = conn.query(query, {"username": username})

        # If we found any rows, the user exists
        if not df.empty:
            return False  # User already exists

        # Insert the new user with parameterized query
        with conn.session as session:
            insert_query = text("INSERT INTO atfp (username, password) VALUES (:username, :password)")
            session.execute(insert_query, {"username": username, "password": password})
            session.commit()
        return True

    except Exception as e:
        # Consider logging the error here
        print(f"Error registering user: {e}")
        return False


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()  



col1, col2 = st.columns([1, 1])
if st.session_state.lang == "ukr":
    with col1:
        if st.button("Увійти"):
            go_to("login")
    with col2:
        if st.button("Реєстрація"):
            go_to("register")
else:
    with col1:
        if st.button("Log in"):
            go_to("login")
    with col2:
        if st.button("Register"):
            go_to("register")


if st.session_state.page == "login":
    if st.session_state.lang == "ukr":
        with st.form("login_form"):
            st.title("Ввійдіть у свій акаунт")
            username = st.text_input("Ім'я користувача")
            password = st.text_input("Пароль", type="password")
            if st.form_submit_button("Ввійти"):
                if check_credentials(username, password):
                    st.session_state.user = username
                    st.success(f"Вітаємо, {username}!")
                    go_to("home")
                else:
                    st.warning("Невірний логін або пароль")
    else:
        with st.form("login_form"):
            st.title("Log in to your account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                if check_credentials(username, password):
                    st.session_state.user = username
                    st.success(f"Welcome, {username}!")
                    go_to("home")
                else:
                    st.warning("Incorrect username or password")

elif st.session_state.page == "register":
    if st.session_state.lang == "ukr":
        with st.form("register_form"):
            st.title("Реєструйтесь")
            username = st.text_input("Ім'я користувача")
            password = st.text_input("Пароль", type="password")
            confirm_password = st.text_input("Підтвердіть пароль", type="password")

            if st.form_submit_button("Зареєструватися"):
                if not username or not password:
                    st.warning("Заповніть усі поля!")
                elif password != confirm_password:
                    st.warning("Паролі не співпадають!")
                else:
                    if register_user(username, password):
                        st.success("Реєстрація успішна!")
                        go_to("login")
                    else:
                        st.warning("Користувач з таким ім'ям вже існує!")
    else:
        with st.form("register_form"):
                st.title("Register")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Repeat Password", type="password")

                if st.form_submit_button("Register"):
                    if not username or not password:
                        st.warning("Fill all the fields")
                    elif password != confirm_password:
                        st.warning("Passwords don't match!")
                    else:
                        if register_user(username, password):
                            st.success("Registration successful!")
                            go_to("login")
                        else:
                            st.warning("User with same username already exists!")

elif st.session_state.page == "home":
    if st.session_state.lang == "ukr":
        st.write(f"Ви в системі як {st.session_state.user}")
        if st.button("Вийти"):
            st.session_state.user = None
            go_to("login")
    else:
        st.write(f"You are logged in as {st.session_state.user}")
        if st.button("Log out"):
            st.session_state.user = None
            go_to("login")