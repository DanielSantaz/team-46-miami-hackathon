import streamlit as st

st.write('Hello world!')

st.title("Title")

menu = ["Image","Text"]
choice = st.sidebar.selectbox("Menu",menu)