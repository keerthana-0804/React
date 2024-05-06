import streamlit as st
import subprocess
import matplotlib.pyplot as plt
from PIL import Image

def run_script():
    # Call your Python script here
    subprocess.run(["python", "iamsamplingfinalprg.py"], capture_output=True)

def display_graph():
    # Load and display the graph image
    graph_image = Image.open("miragraph.png")
    st.image(graph_image, caption='Grid World Environment Graph', use_column_width=True)

def display_pygame():
    # Display Pygame window
    st.write("Pygame window will be displayed here.")

# Streamlit UI
st.title("Grid World Environment")

# Define UI elements
st.write("Welcome to the Grid World Environment!")
st.write("Click the buttons below to interact with the environment:")

# Button to run the script
if st.button("Run Script"):
    run_script()

# Button to display the graph
if st.button("Display Graph"):
    display_graph()

# Button to display the Pygame window
if st.button("Display Pygame Window"):
    display_pygame()
