import streamlit as st
x = st.slider("Select a value")
st.title('제곱 구하기')
st.write(x, "squared is", x * x)