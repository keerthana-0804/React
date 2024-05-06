import streamlit as st

# Adding a style block to change the color of the first button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(204, 49, 49);
}
</style>""", unsafe_allow_html=True)

# Adding a style block to change the color of the second button
st.markdown("""
<style>
div.stButton > button:nth-child(2) {
    background-color: rgb(49, 204, 49); /* Change color to green */
}
</style>""", unsafe_allow_html=True)

# Creating the first button
b = st.button("test")

# Creating the second button
c = st.button("test2")
