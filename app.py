import streamlit as st
import pandas as pd
import numpy as np
from prediction import transform_predict

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Customer Support Bot Router')

columns = st.columns(2)
with columns[0]:
    first_name = st.text_input("First Name:", max_chars=90)
with columns[1]:
    last_name = st.text_input("Last Name:", max_chars=90)

txt = st.text_area("Type your question:")

if st.button("Send"):
    if txt:
        text = txt
        result = transform_predict(text)
        st.write(f'Dear {first_name} {last_name}. We will forward your request to the {result[0]} department.')
    else:
        st.write("Please fill the box with your request!")

st.image('assets/customer-support-flat.png')

url = "https://github.com/camilasp/customer_support_bot_router"
st.write("Know our [work](%s)" % url)
