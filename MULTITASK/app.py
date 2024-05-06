import streamlit as st
import subprocess

def run_script():
    # Call your Python script here
    process = subprocess.run(["python", "script.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Display the output
    st.write("Output:")
    st.code(process.stdout)
    st.write("Errors:")
    st.code(process.stderr)

# Streamlit UI
st.title("Sample UI")

# Define UI elements
st.write("Click the button below to run the script:")
if st.button("Run Script here"):
    run_script()
