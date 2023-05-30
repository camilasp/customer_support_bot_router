import streamlit as st
import pandas as pd
import numpy as np
from prediction import transform_predict

st.title('Customer Support Bot Router')

txt = st.text_area("Type your question:")

if st.button("Send"):
    text = txt
    result = transform_predict(text)
    st.write("We will forward your request to the " +result[0] + " department.")

